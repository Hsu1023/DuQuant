"""Train-only Qwen3 adaptation of DuQuant/QuaRot/SpinQuant plus common PTQ."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel
from qwen3_models import *
from comparison_models import identity_rotations,grouped_weights,had
from search_ptq import W_RATIOS,A_RATIOS,cpu_tree,sample_rows,CollectLinear,sha
from duquant_cuda.calibration import smoothing_scales,apply_smoothing,install_rotation_base,CalibratingLinear
from duquant_cuda.ptq import symmetric_rtn,search_activation,gptq
from duquant_cuda.backend import transform


def groups_block(w,rot,method,plan=None,mode='reference'):
    return wrap_basis(QwenBlock(w,rot,mode,plan,4),method)


def rotate(w,r1,r2,method):
    for norm,names in [('input_layernorm.weight',['self_attn.'+n+'_proj.weight' for n in ('q','k','v')]),('post_attention_layernorm.weight',['mlp.'+n+'_proj.weight' for n in ('gate','up')])]:
        for key in names:w[key]=((w[key].double()*w[norm].double()).half().double()@r1).half()
        w[norm]=torch.ones_like(w[norm])
    for key in ('self_attn.o_proj.weight','mlp.down_proj.weight'):w[key]=(r1.T@w[key].double()).half()
    v=w['self_attn.v_proj.weight'];w['self_attn.v_proj.weight']=(v.T.double().reshape(-1,8,128)@r2).reshape(v.T.shape).T.half().contiguous()
    o=w['self_attn.o_proj.weight'];w['self_attn.o_proj.weight']=(o.double().reshape(-1,16,128)@r2).reshape(o.shape).half()
    if method=='quarot':
        from fast_hadamard_transform import hadamard_transform
        o=w['self_attn.o_proj.weight'];w['self_attn.o_proj.weight']=hadamard_transform(o.float().reshape(-1,16,128).transpose(1,2).contiguous(),scale=.25).transpose(1,2).reshape(o.shape).half()
    d=w['mlp.down_proj.weight'];hk,k=had.get_hadK(d.shape[1]);w['mlp.down_proj.weight']=had.matmul_hadU_cuda(d.float(),hk,k).half()
    return w


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--method',required=True,choices=['duquant','quarot','spinquant']);args=p.parse_args();method=args.method
    folder=BASE/method;folder.mkdir(exist_ok=True)
    if (folder/'manifest.json').exists():raise FileExistsError('Choose fresh calibration')
    torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    tokens=torch.load(BASE/'train.pt',weights_only=True).cuda();ext=extras()
    r1=None
    if method=='quarot':r1=had.random_hadamard_matrix(2048,'cuda').double()
    if method=='spinquant':
        learned=torch.load(BASE/'spin_rotations.pt',map_location='cuda',weights_only=True);r1=learned['R1'].double()
    if r1 is not None:
        ext['embedding']=(ext['embedding'].cuda().double()@r1).half().cpu()
        ext['lm_head']=((ext['lm_head'].cuda().double()*ext['final_norm'].cuda().double()).half().double()@r1).half().cpu()
        ext['final_norm']=torch.ones_like(ext['final_norm'])
    torch.save(ext,folder/'extras.pt');hidden=F.embedding(tokens,ext['embedding'].cuda());h_rtn=hidden.clone();h_gptq=hidden.clone();rep=hidden.mean(0,keepdim=True)
    if method=='duquant':install_rotation_base()
    train=json.loads((BASE/'train.json').read_text())
    manifest=dict(status='running',method=method,model=str(MODEL),training=train,setting='W4A4KV4, Q16; symmetric per-channel W, asymmetric per-token A, per-head KV128',layers=[],gptq=dict(blocksize=128,damp=.01,act_order=False),weight_clip_grid=W_RATIOS,activation_kv_clip_grid=A_RATIOS,
        source_sha256={str(p):sha(p) for p in [Path(__file__),Path('benchmarks/qwen3_models.py'),Path('duquant_cuda/ptq.py')]})
    if method=='spinquant':manifest['rotation_training']=json.loads((REPORT/'spin_training.json').read_text())
    logs=[]
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for i in range(28):
            start=time.perf_counter();print(method,'layer',i+1,flush=True);w=load_weights(i)
            rot={n:identity_rotations(k,'cuda') for n,k in [('qkv',2048),('o',2048),('gate_up',2048),('down',6144)]}
            if method=='duquant':
                fp=QwenBlock(w);maxima={}
                def hook(name,x):
                    v=x.reshape(-1,x.shape[-1]).abs().amax(0);maxima[name]=torch.maximum(maxima[name],v) if name in maxima else v
                handles=[getattr(fp,n).register_forward_pre_hook(lambda m,inp,n=n:hook(n,inp[0])) for n in rot]
                cache=fp.cache(1,2048)
                for j in range(128):hidden[j:j+1]=fp(hidden[j:j+1],cache)
                for h in handles:h.remove()
                scales=smoothing_scales(w,maxima,.6);apply_smoothing(w,scales)
                block=QwenBlock(w);block.kv_bits=4
                for n,weight in grouped_weights(w).items():setattr(block,n,CalibratingLinear(weight,256))
                rep=block(rep,block.cache(1,2048));rot={n:getattr(block,n).exported_rotations for n in rot}
                del block,fp,cache,maxima
            else:
                r2=learned['R2'][i].double() if method=='spinquant' else had.matmul_hadU(torch.eye(128,dtype=torch.float64)).cuda()
                original=QwenBlock(w);probe=torch.randn(1,17,2048,device='cuda',dtype=torch.float16)*.1
                y=original(probe,original.cache(1,17));w=rotate(w,r1,r2,method)
                test=wrap_basis(QwenBlock(w),method);xp=(probe.double()@r1).half();actual=test(xp,test.cache(1,17));expect=(y.double()@r1).half()
                rel=((actual.float()-expect.float()).norm()/expect.float().norm()).item();assert rel<.01,(i,rel)
                del original,test,probe,y,xp,actual,expect
            ws=grouped_weights(w);block=groups_block(w,rot,method);block.k_samples=[];block.v_samples=[]
            for n,weight in ws.items():
                col=CollectLinear(weight,rot[n]);setattr(block,n,Basis(col,n,method) if method!='duquant' else col)
            cache=block.cache(1,2048)
            for j in range(128):block(h_rtn[j:j+1],cache)
            plan=dict(weights={},act_clips={},kv_clips=[])
            for n,weight in ws.items():
                col=getattr(block,n);col=col.inner if isinstance(col,Basis) else col
                plan['weights'][n],_=symmetric_rtn(transform(weight,*rot[n]),W_RATIOS)
                plan['act_clips'][n],_=search_activation(torch.cat(col.samples),A_RATIOS)
            for samples in (block.k_samples,block.v_samples):clip,_=search_activation(torch.cat(samples),A_RATIOS);plan['kv_clips'].append(clip)
            del block,cache
            block=groups_block(w,rot,method,plan,'cuda');cache=block.cache(1,2048)
            for j in range(128):h_rtn[j:j+1]=block(h_rtn[j:j+1],cache)
            for n,weight in ws.items():
                print(' GPTQ',n,flush=True);col=CollectLinear(weight,rot[n],act_clip=plan['act_clips'][n],hessian=True)
                setattr(block,n,Basis(col,n,method) if method!='duquant' else col)
                for j in range(128):block(h_gptq[j:j+1],cache)
                col.hessian.mul_(2./col.tokens);plan['weights'][n],_=gptq(transform(weight,*rot[n]),col.hessian,plan['weights'][n]['scale'])
                from duquant_cuda.backend import DuQuantLinearCUDA
                inner=DuQuantLinearCUDA(weight,*rot[n],quantized_weight=plan['weights'][n],act_clip=plan['act_clips'][n])
                setattr(block,n,Basis(inner,n,method) if method!='duquant' else inner)
                del col
            for j in range(128):h_gptq[j:j+1]=block(h_gptq[j:j+1],cache)
            target=folder/f'layer_{i:02d}.pt';torch.save(cpu_tree(dict(weights=w,rotations=rot,plan=plan)),target)
            manifest['layers'].append(dict(index=i,file=target.name,sha256=sha(target)))
            logs.append(dict(layer=i,seconds=time.perf_counter()-start))
            (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(REPORT/f'{method}_calibration.json').write_text(json.dumps(logs,indent=2)+'\n')
            print('Saved',i+1,'seconds',logs[-1]['seconds'],flush=True)
            del w,ws,block,cache,plan,rot;torch.cuda.empty_cache()
    manifest['status']='complete';(folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(REPORT/f'{method}_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
