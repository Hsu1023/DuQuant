"""Apply identical searched symmetric-W/asymmetric-AKV GPTQ to two bases."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import types
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel
from datasets import Dataset
from transformers import AutoTokenizer
from comparison_models import (BASE,MODEL,QUAROT,NAMES,had,make_block,rotate_layer,
    rotated_extras,BasisLinear,grouped_weights)
from search_ptq import W_RATIOS,A_RATIOS,collect_kv,cpu_tree
from duquant_cuda.decoder import LlamaDecoderBlock,load_layer_weights
from duquant_cuda.ptq import symmetric_rtn,search_activation,gptq


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['quarot','spinquant'],required=True)
    args=p.parse_args();folder=BASE/args.method;report=Path('benchmarks/qa_compare')/args.method
    if (folder/'manifest.json').exists():raise FileExistsError('Use a fresh output; do not silently reuse comparator exports')
    for name in ('rotated','rtn','gptq'):(folder/name).mkdir(parents=True,exist_ok=True)
    report.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    cfg=json.loads((MODEL/'config.json').read_text())
    parent=Path('benchmarks/cache/calibrated_w4a4/manifest.json');meta=json.loads(parent.read_text())
    if sha(meta['source_arrow'])!=meta['source_arrow_sha256']:raise ValueError('Training data hash mismatch')
    tok=AutoTokenizer.from_pretrained(MODEL,use_fast=False,local_files_only=True)
    data=Dataset.from_file(meta['source_arrow'])
    ids=tok('\n\n'.join(data['text']),return_tensors='pt').input_ids
    tokens=torch.cat([ids[:,s:s+2048] for s in meta['sample_starts']])
    assert hashlib.sha256(tokens.numpy().tobytes()).hexdigest()==meta['token_ids_sha256']
    if args.method=='spinquant':
        checkpoint=BASE/'spinquant_llama2_w16a4kv4_R.bin'
        rotations=torch.load(checkpoint,map_location='cpu',weights_only=True)
        r1=rotations['R1'].cuda().double();rotation_source=dict(file=str(checkpoint),sha256=sha(checkpoint),
            official_file_id='193O_aCS6MaObW8JosZP5CASwHAVU259E',setting='Official W16A4KV4 rotation, seed0, used with GPTQ')
    else:
        r1=had.random_hadamard_matrix(4096,'cuda').double()
        r2=had.matmul_hadU(torch.eye(128,dtype=torch.float64)).cuda()
        rotation_source=dict(seed=0,type='QuaRot randomized residual Hadamard; fixed head Hadamard plus online across-head Hadamard')
    gram=r1.T@r1-torch.eye(4096,device='cuda',dtype=torch.float64)
    orth_error=gram.abs().max().item();assert orth_error<.002,orth_error
    extra=rotated_extras(r1);torch.save(extra,folder/'extras.pt')
    embedding=extra['embedding'].cuda()
    h_rtn=F.embedding(tokens.cuda(),embedding);h_gptq=h_rtn.clone()
    del embedding,extra,gram,tokens,ids,data
    manifest=dict(status='running',method=args.method,model=str(MODEL),layers=[],
        rotation_source=rotation_source,r1_max_orthogonality_error=orth_error,
        training_parent_manifest_sha256=sha(parent),training_token_ids_sha256=meta['token_ids_sha256'],
        nsamples=128,seqlen=2048,split='train',weight_bits=4,activation_bits=4,kv_bits=4,
        weight_symmetric=True,activation_symmetric=False,kv_symmetric=False,query_bits=16,
        w_groupsize=-1,a_groupsize=-1,kv_groupsize=128,weight_clip_grid=W_RATIOS,activation_kv_clip_grid=A_RATIOS,
        gptq=dict(blocksize=128,damp=.01,act_order=False,calibration='own quantized prefix; true-sequential A4 inputs'),
        source_sha256={str(x):sha(x) for x in [Path(__file__),Path('benchmarks/comparison_models.py'),
            Path('benchmarks/search_ptq.py'),Path('duquant_cuda/ptq.py'),QUAROT/'fake_quant/hadamard_utils.py']})
    logs=[]
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for layer in range(32):
            start=time.perf_counter();print(f'{args.method} layer {layer+1}/32',flush=True)
            original,_=load_layer_weights(MODEL,layer)
            if args.method=='spinquant':r2=rotations[f'model.layers.{layer}.self_attn.R2'].cuda().double()
            assert (r2.T@r2-torch.eye(128,device='cuda')).abs().max()<.002
            # Independent full-precision invariance test before introducing quantization.
            ref=LlamaDecoderBlock(original,{},'fp16',kv_bits=16)
            ref.mode='reference'
            probe=torch.randn(1,17,4096,device='cuda',dtype=torch.float16)*.1
            y=ref(probe,ref.cache(1,17))
            del ref
            w=rotate_layer(original,r1,r2,args.method)
            float_block=make_block(w,args.method,cfg,quantize_kv=False)
            rotated_input=(probe.double()@r1).half()
            yr=float_block(rotated_input,float_block.cache(1,17))
            expected=(y.double()@r1).half()
            rel=((yr.float()-expected.float()).norm()/expected.float().norm()).item()
            if rel>.01:raise AssertionError(f'Rotation invariance failed: layer={layer}, relative L2={rel}')
            del float_block,probe,y,yr,expected,rotated_input
            path=folder/'rotated'/f'layer_{layer:02d}.pt';torch.save(cpu_tree(w),path)
            ws=grouped_weights(w)
            block=make_block(w,args.method,cfg)
            for name in NAMES:setattr(block,name,BasisLinear(ws[name],name,args.method,collect=True))
            block.k_samples,block.v_samples=[],[]
            block._qkv_reference=types.MethodType(collect_kv,block)
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
            torch.save(cpu_tree(plan),folder/'rtn'/f'layer_{layer:02d}.pt')
            del block,cache
            block=make_block(w,args.method,cfg,plan,mode='cuda');cache=block.cache(1,2048)
            for j in range(128):h_rtn[j:j+1]=block(h_rtn[j:j+1],cache)
            for name in NAMES:
                print(f'  GPTQ {name}',flush=True)
                col=BasisLinear(ws[name],name,args.method,act_clip=plan['act_clips'][name],hessian=True)
                setattr(block,name,col)
                for j in range(128):block(h_gptq[j:j+1],cache)
                col.hessian.mul_(2./col.tokens)
                state,detail=gptq(ws[name],col.hessian,plan['weights'][name]['scale'])
                plan['weights'][name]=state;details[name]['gptq']=detail
                setattr(block,name,BasisLinear(ws[name],name,args.method,state,plan['act_clips'][name],mode='cuda'))
                del col
            for j in range(128):h_gptq[j:j+1]=block(h_gptq[j:j+1],cache)
            out=folder/'gptq'/f'layer_{layer:02d}.pt';torch.save(cpu_tree(plan),out)
            manifest['layers'].append(dict(index=layer,rotated_sha256=sha(path),gptq_sha256=sha(out)))
            logs.append(dict(layer=layer,seconds=time.perf_counter()-start,rotation_invariance_relative_l2=rel,details=details))
            (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            (report/'calibration.json').write_text(json.dumps(logs,indent=2)+'\n')
            print(f'Saved {layer+1}/32: {logs[-1]["seconds"]:.1f}s; rotation error {rel:.6g}',flush=True)
            del w,ws,plan,block,cache,original
            torch.cuda.empty_cache()
    manifest['status']='complete';manifest['extras_sha256']=sha(folder/'extras.pt')
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (report/'calibration_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
