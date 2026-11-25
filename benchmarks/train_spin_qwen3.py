"""Qwen3 adaptation of SpinQuant: optimize R1/R2 with official Cayley SGD.

Uses W16A4KV4 during rotation learning, then common searched W4 GPTQ.
All rotation-learning tokens come from the same WikiText train calibration set.
"""
from qwen3_models import *
import importlib.util,math,random,time,types
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb,repeat_kv
from torch.utils.checkpoint import checkpoint
from comparison_models import had
from search_ptq import sha

spec=importlib.util.spec_from_file_location('spin_official_optimizer',ROOT/'benchmarks/cache/SpinQuant/train_utils/optimizer.py')
optmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(optmod)


def ste(x,clip=.9):
    from duquant_cuda.decoder import qdq
    return x+(qdq(x.detach(),clip)-x).detach()


class RotLinear(torch.nn.Module):
    def __init__(self,w,r1,r2=None,output=False,down=False):
        super().__init__();self.register_buffer('weight',w);object.__setattr__(self,'r1',r1);object.__setattr__(self,'r2',r2)
        self.output,self.down=output,down
    def forward(self,x):
        w=self.weight.float()
        if self.output:w=self.r1.T@w
        else:w=w@self.r1
        if self.r2 is not None:
            if self.output:w=(w.reshape(-1,w.shape[1]//128,128)@self.r2).reshape(w.shape)
            else:w=(w.T.reshape(-1,w.shape[0]//128,128)@self.r2).reshape(w.T.shape).T
        if self.down:
            hk,k=had.get_hadK(x.shape[-1]);x=had.matmul_hadU_cuda(x.float(),hk,k).half()
        return F.linear(ste(x),w.half())


def attention(self,hidden_states,position_embeddings,attention_mask=None,**kwargs):
    shape=hidden_states.shape[:-1];hs=(*shape,-1,128)
    q=self.q_norm(self.q_proj(hidden_states).view(hs)).transpose(1,2)
    k=self.k_norm(self.k_proj(hidden_states).view(hs)).transpose(1,2)
    v=self.v_proj(hidden_states).view(hs).transpose(1,2)
    q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
    q=q@self.had;k=ste(k@self.had);v=ste(v,1.)
    a=F.scaled_dot_product_attention(q,repeat_kv(k,2),repeat_kv(v,2),is_causal=q.shape[2]>1)
    return self.o_proj(a.transpose(1,2).reshape(*shape,2048).contiguous()),None


def main():
    torch.manual_seed(0);random.seed(0);torch.backends.cuda.matmul.allow_tf32=False
    target=BASE/'spin_rotations.pt'
    if target.exists():raise FileExistsError(target)
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
    for p in model.parameters():p.requires_grad=False
    # Untie the original tied head before independently folding final RMSNorm.
    model.lm_head.weight=torch.nn.Parameter(model.lm_head.weight.detach().clone(),requires_grad=False)
    r1=torch.nn.Parameter(had.random_hadamard_matrix(2048,'cuda').float())
    r2=[torch.nn.Parameter(had.random_hadamard_matrix(128,'cuda').float()) for _ in range(28)]
    for i,layer in enumerate(model.model.layers):
        attn=layer.self_attn;mlp=layer.mlp
        for norm,modules in [(layer.input_layernorm,[attn.q_proj,attn.k_proj,attn.v_proj]),(layer.post_attention_layernorm,[mlp.gate_proj,mlp.up_proj])]:
            for m in modules:m.weight.data=(m.weight.double()*norm.weight.double()).half()
            norm.weight.data.fill_(1)
        d=mlp.down_proj.weight;hk,k=had.get_hadK(d.shape[1]);d.data=had.matmul_hadU_cuda(d.float(),hk,k).half()
        for name in ('q_proj','k_proj','v_proj','o_proj'):
            old=getattr(attn,name);setattr(attn,name,RotLinear(old.weight,r1,r2[i] if name in ('v_proj','o_proj') else None,output=name=='o_proj'))
        for name in ('gate_proj','up_proj','down_proj'):
            old=getattr(mlp,name);setattr(mlp,name,RotLinear(old.weight,r1,output=name=='down_proj',down=name=='down_proj'))
        from duquant_cuda.decoder import hadamard_matrix
        attn.register_buffer('had',hadamard_matrix('cuda'));attn.forward=types.MethodType(attention,attn)
    final=model.model.norm.weight.detach().clone();model.model.norm.weight.data.fill_(1)
    head=(model.lm_head.weight.double()*final.double()).half().detach();model.lm_head=None
    def checkpointed_forward(layer):
        old=layer.forward
        def forward(*args,**kwargs):
            return checkpoint(lambda x:old(x,**kwargs),args[0],use_reentrant=False)
        layer.forward=forward
    for layer in model.model.layers:checkpointed_forward(layer)
    tokens=torch.load(BASE/'train.pt',weights_only=True).cuda()
    optim=optmod.SGDG([r1,*r2],lr=1.5,momentum=.9,stiefel=True)
    logs=[];start=time.perf_counter();steps=800
    for step in range(steps):
        optim.zero_grad();ids=tokens[step%128:step%128+1]
        embed=(F.embedding(ids,model.model.embed_tokens.weight).float()@r1).half()
        hidden=model.model(inputs_embeds=embed,use_cache=False).last_hidden_state
        logits=F.linear(hidden,(head.float()@r1).half())
        loss=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1))
        loss.backward()
        for group in optim.param_groups:group['lr']=1.5*.5*(1+math.cos(math.pi*step/steps))
        optim.step();logs.append(dict(step=step,loss=loss.item(),seconds=time.perf_counter()-start))
        if step%10==0:print(logs[-1],flush=True)
        if (step+1)%100==0:
            torch.save(dict(R1=r1.detach().cpu(),R2=torch.stack([r.detach().cpu() for r in r2])),BASE/'spin_rotations.partial.pt')
        (REPORT/'spin_training_progress.json').write_text(json.dumps(logs[-1],indent=2)+'\n')
    gram=max((r.T@r-torch.eye(len(r),device='cuda')).abs().max().item() for r in [r1,*r2])
    assert gram<.01,gram
    torch.save(dict(R1=r1.detach().cpu(),R2=torch.stack([r.detach().cpu() for r in r2])),target)
    meta=dict(status='complete',steps=steps,learning_rate=1.5,schedule='cosine',momentum=.9,optimizer='Official SpinQuant SGDG Cayley',objective='Causal language-model loss with W16A4KV4 STE',training=json.loads((BASE/'train.json').read_text()),sample_schedule='128 fixed train samples cycled for 800 updates',max_orthogonality_error=gram,logs=logs,source_sha256=sha(Path(__file__)),rotation_sha256=sha(target),adaptation='Local Qwen3 adaptation; not an official released checkpoint')
    (REPORT/'spin_training.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':main()
