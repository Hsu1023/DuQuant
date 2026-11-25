"""Shared clipping/GPTQ calibration of the official FlatQuant transforms."""
import hashlib,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel
from qwen3_flat_models import *
from search_ptq import W_RATIOS,A_RATIOS,cpu_tree,sha
from duquant_cuda.ptq import symmetric_rtn,search_activation,gptq
NAMES=('qkv','o','gate_up','down')


@torch.inference_mode()
def main():
    report=FLAT_REPORT
    if (FOLDER/'manifest.json').exists():raise FileExistsError('Existing calibration')
    for name in ('rotated','rtn','gptq'):(FOLDER/name).mkdir(parents=True,exist_ok=True)
    torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    target=BASE/'flatquant_training'
    training=json.loads((target/'manifest.json').read_text());assert training['status']=='complete'
    tokens=torch.load(BASE/'train.pt',weights_only=True).cuda();extra=extras()
    torch.save(extra,FOLDER/'extras.pt')
    h_rtn=F.embedding(tokens,extra['embedding'].cuda());h_gptq=h_rtn.clone()
    del tokens,extra
    manifest=dict(status='running',method='flatquant',model=str(MODEL),layers=[],
        training=json.loads((BASE/'train.json').read_text()),transformation_training_manifest_sha256=sha(target/'manifest.json'),
        weight_bits=4,activation_bits=4,kv_bits=4,query_bits=16,weight_symmetric=True,
        activation_symmetric=False,kv_symmetric=False,kv_groupsize=128,
        weight_clip_grid=W_RATIOS,activation_kv_clip_grid=A_RATIOS,
        gptq=dict(blocksize=128,damp=.01,act_order=False),
        source_sha256={str(p):sha(p) for p in [Path(__file__),Path('benchmarks/qwen3_flat_models.py'),Path('benchmarks/flatquant_models.py'),Path('benchmarks/qwen3_models.py'),Path('duquant_cuda/ptq.py')]})
    logs=[]
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for layer in range(28):
            start=time.perf_counter()
            print(f'FlatQuant layer {layer+1}/28',flush=True)
            original=load_weights(layer)
            entry=training['layers'][layer];assert sha(target/entry['file'])==entry['sha256']
            state=torch.load(target/entry['file'],weights_only=True,map_location='cuda')
            trans=transforms(state)
            inverse_errors=[]
            for key, matrix in state.items():
                if key.endswith(('.matrix_left', '.matrix_right', '.matrix')):
                    invkey=key+('_inv_t' if key.endswith('.matrix') else '_inv')
                    error=(matrix.double().T@state[invkey].double()-torch.eye(len(matrix),device='cuda')).abs().max().item()
                    inverse_errors.append(error)
            assert max(inverse_errors)<.001, inverse_errors
            w=transform_weights(original,state)
            ref=QwenBlock(original)
            ref.mode='reference'
            flat=make_block(w,trans,quantize_kv=False)
            probe=torch.randn(1,17,2048,device='cuda',dtype=torch.float16)*.1
            a,b=ref(probe,ref.cache(1,17)),flat(probe,flat.cache(1,17))
            rel=((a.float()-b.float()).norm()/a.float().norm()).item()
            assert rel<.01, (layer,rel)
            del ref,flat,probe,a,b,state,original
            path=FOLDER/'rotated'/f'layer_{layer:02d}.pt'
            torch.save(cpu_tree(dict(weights=w,transforms=trans)),path)
            ws=grouped_weights(w)
            block=make_block(w,trans)
            for name in NAMES:
                setattr(block,name,FlatLinear(ws[name],trans[name],collect=True))
            block.k_samples,block.v_samples=[],[]
            cache=block.cache(1,2048)
            for j in range(128):block(h_rtn[j:j+1],cache)
            plan=dict(weights={},act_clips={},kv_clips=[]);details={}
            for name in NAMES:
                plan['weights'][name],wr=symmetric_rtn(ws[name],W_RATIOS)
                plan['act_clips'][name],ar=search_activation(torch.cat(getattr(block,name).samples),A_RATIOS)
                details[name]=dict(weight={k:v for k,v in wr.items() if k!='clips'},activation=ar)
            for name,samples in [('k',block.k_samples),('v',block.v_samples)]:
                clip,detail=search_activation(torch.cat(samples),A_RATIOS)
                plan['kv_clips'].append(clip);details[name]=detail
            torch.save(cpu_tree(plan),FOLDER/'rtn'/f'layer_{layer:02d}.pt')
            del block,cache
            block=make_block(w,trans,plan,'cuda');cache=block.cache(1,2048)
            for j in range(128):h_rtn[j:j+1]=block(h_rtn[j:j+1],cache)
            for name in NAMES:
                print(f'  GPTQ {name}',flush=True)
                col=FlatLinear(ws[name],trans[name],act_clip=plan['act_clips'][name],hessian=True)
                setattr(block,name,col)
                for j in range(128):block(h_gptq[j:j+1],cache)
                col.hessian.mul_(2./col.tokens)
                quantized,detail=gptq(ws[name],col.hessian,plan['weights'][name]['scale'])
                plan['weights'][name]=quantized;details[name]['gptq']=detail
                setattr(block,name,FlatLinear(ws[name],trans[name],quantized,plan['act_clips'][name],'cuda'))
                del col
            for j in range(128):h_gptq[j:j+1]=block(h_gptq[j:j+1],cache)
            out=FOLDER/'gptq'/f'layer_{layer:02d}.pt';torch.save(cpu_tree(plan),out)
            manifest['layers'].append(dict(index=layer,rotated_sha256=sha(path),gptq_sha256=sha(out)))
            logs.append(dict(layer=layer,seconds=time.perf_counter()-start,
                float_invariance_relative_l2=rel,max_inverse_error=max(inverse_errors),details=details))
            (FOLDER/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            (report/'calibration.json').write_text(json.dumps(logs,indent=2)+'\n')
            print(f'Saved {layer+1}/28: {logs[-1]["seconds"]:.1f}s, float error {rel:.6g}',flush=True)
            del w,ws,plan,block,cache,trans
            torch.cuda.empty_cache()
    manifest['status']='complete';manifest['extras_sha256']=sha(FOLDER/'extras.pt')
    (FOLDER/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (report/'calibration_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
