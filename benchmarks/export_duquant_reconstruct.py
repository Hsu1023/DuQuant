"""Export learned folding/clipping with RTN and fixed-scale true-sequential GPTQ."""
from duquant_reconstruct import *
from search_ptq import sha,CollectLinear
from duquant_cuda.ptq import gptq
from duquant_cuda.backend import DuQuantLinearCUDA
from torch.nn.attention import SDPBackend,sdpa_kernel
import time


@torch.inference_mode()
def main():
    target=BASE/'duquant_reconstruct_training';training=json.loads((target/'manifest.json').read_text());assert training['status']=='complete'
    manifests={}
    for variant in ('rtn','gptq'):
        folder=FOLDER/variant;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'manifest.json').exists():raise FileExistsError(folder)
        manifests[variant]=dict(status='running',variant=variant,method='duquant_reconstruct',model=str(MODEL),layers=[],training=training['training'],
            training_manifest_sha256=sha(target/'manifest.json'),parent_manifest_sha256=sha(BASE/'duquant/manifest.json'),
            rotations='Unchanged parent rotations/permutation',quantization='W4 symmetric per-channel; A4 scalar-ratio asymmetric per-token; Q16; KV4 per-head128',
            gptq=dict(blocksize=128,damp=.01,act_order=False) if variant=='gptq' else None,
            source_sha256={str(p):sha(p) for p in [Path(__file__),Path('benchmarks/duquant_reconstruct.py'),Path('duquant_cuda/ptq.py')]})
    torch.backends.cuda.matmul.allow_tf32=False
    tokens=torch.load(BASE/'train.pt',weights_only=True).cuda();extra=extras();h_gptq=F.embedding(tokens,extra['embedding'].cuda());h_rtn=h_gptq.clone();logs=[]
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for i in range(28):
            start=time.perf_counter();print('Export',i+1,flush=True)
            entry=training['layers'][i];assert sha(target/entry['file'])==entry['sha256']
            params=torch.load(target/entry['file'],weights_only=True,map_location='cuda');w,rot=folded_base(i,params)
            ws=grouped_weights(w);plan=dict(weights={},act_clips=params['act_clips'],kv_clips=params['kv_clips'])
            for n,weight in ws.items():
                wt=transform(weight,*rot[n]);ratio=params['weight_clips'][n]
                scale=(wt.float().abs().amax(1)*ratio/7).clamp(1e-5,1e4).half()
                code=((wt.float()/scale[:,None]).round().clamp(-8,7)+8).byte()
                plan['weights'][n]=dict(codes=code,scale=scale,zero=torch.full_like(scale,8))
            torch.save(cpu_tree(dict(parameters=params,plan=plan)),FOLDER/'rtn'/f'layer_{i:02d}.pt')
            block=QwenBlock(w,rot,'cuda',plan,4);cache=block.cache(1,2048)
            for j in range(128):h_rtn[j:j+1]=block(h_rtn[j:j+1],cache)
            for n,weight in ws.items():
                col=CollectLinear(weight,rot[n],act_clip=plan['act_clips'][n],hessian=True);setattr(block,n,col)
                for j in range(128):block(h_gptq[j:j+1],cache)
                col.hessian.mul_(2./col.tokens)
                plan['weights'][n],_=gptq(transform(weight,*rot[n]),col.hessian,plan['weights'][n]['scale'])
                setattr(block,n,DuQuantLinearCUDA(weight,*rot[n],quantized_weight=plan['weights'][n],act_clip=plan['act_clips'][n]));del col
            for j in range(128):h_gptq[j:j+1]=block(h_gptq[j:j+1],cache)
            torch.save(cpu_tree(dict(parameters=params,plan=plan)),FOLDER/'gptq'/f'layer_{i:02d}.pt')
            for variant in manifests:
                path=FOLDER/variant/f'layer_{i:02d}.pt';m=manifests[variant]
                m['layers'].append(dict(index=i,file=path.name,sha256=sha(path)));(path.parent/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
            logs.append(dict(layer=i,seconds=time.perf_counter()-start));(TRIAL_REPORT/'export.json').write_text(json.dumps(logs,indent=2)+'\n')
            print('Saved',i+1,'seconds',logs[-1]['seconds'],flush=True)
            del w,ws,rot,params,plan,block,cache;torch.cuda.empty_cache()
    for variant,m in manifests.items():
        m['status']='complete';(FOLDER/variant/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');(TRIAL_REPORT/f'{variant}_manifest.json').write_text(json.dumps(m,indent=2)+'\n')

if __name__=='__main__':main()
