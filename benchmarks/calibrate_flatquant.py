"""Shared clipping/GPTQ calibration of the official FlatQuant transforms."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from datasets import Dataset
from transformers import AutoTokenizer
from flatquant_models import (MODEL, FOLDER, CHECKPOINT, REPO, NAMES, transforms,
    transform_weights, make_block, FlatLinear, grouped_weights, original_extras)
from search_ptq import W_RATIOS, A_RATIOS, cpu_tree, sha
from duquant_cuda.decoder import LlamaDecoderBlock, load_layer_weights
from duquant_cuda.ptq import symmetric_rtn, search_activation, gptq


@torch.inference_mode()
def main():
    report = Path('benchmarks/flatquant_compare')
    if (FOLDER/'manifest.json').exists():
        raise FileExistsError('Existing calibration must not be silently overwritten')
    for name in ('rotated', 'rtn', 'gptq'):
        (FOLDER/name).mkdir(parents=True, exist_ok=True)
    report.mkdir(exist_ok=True)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    parent = Path('benchmarks/cache/calibrated_w4a4/manifest.json')
    meta = json.loads(parent.read_text())
    assert sha(meta['source_arrow']) == meta['source_arrow_sha256']
    assert sha(CHECKPOINT) == '7133f8a7bab8179ffa9cc0404c66d22e656f5b67a5abd31066d16bdd25d336ef'
    matrices = torch.load(CHECKPOINT, weights_only=True, map_location='cpu')
    assert len(matrices) == 32
    cfg = json.loads((MODEL/'config.json').read_text())
    data = Dataset.from_file(meta['source_arrow'])
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=False, local_files_only=True)
    ids = tokenizer('\n\n'.join(data['text']), return_tensors='pt').input_ids
    tokens = torch.cat([ids[:, s:s+2048] for s in meta['sample_starts']])
    assert hashlib.sha256(tokens.numpy().tobytes()).hexdigest() == meta['token_ids_sha256']
    extra = original_extras()
    torch.save(extra, FOLDER/'extras.pt')
    h_rtn = F.embedding(tokens.cuda(), extra['embedding'].cuda())
    h_gptq = h_rtn.clone()
    del extra, tokens, data, ids
    source = [Path(__file__), Path('benchmarks/flatquant_models.py'), Path('benchmarks/search_ptq.py'),
              Path('duquant_cuda/ptq.py'), *Path('duquant_cuda').glob('*.py')]
    manifest = dict(status='running', method='flatquant', model=str(MODEL), layers=[],
        transformation_source=dict(file=str(CHECKPOINT), sha256=sha(CHECKPOINT),
            url='https://huggingface.co/ruikangliu/FlatQuant/blob/main/llama-2-7b/w4a4/flat_matrices.pth',
            official_commit=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip(),
            setting='Official W4A4KV4 learned transforms and diagonals; released LWC/LAC replaced by common search'),
        training_parent_manifest_sha256=sha(parent), training_token_ids_sha256=meta['token_ids_sha256'],
        nsamples=128, seqlen=2048, split='train', weight_bits=4, activation_bits=4, kv_bits=4,
        weight_symmetric=True, activation_symmetric=False, kv_symmetric=False, query_bits=16,
        w_groupsize=-1, a_groupsize=-1, kv_groupsize=128,
        weight_clip_grid=W_RATIOS, activation_kv_clip_grid=A_RATIOS,
        gptq=dict(blocksize=128,damp=.01,act_order=False,calibration='own quantized prefix; true-sequential A4 inputs'),
        source_sha256={str(p):sha(p) for p in source})
    logs=[]
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for layer in range(32):
            start=time.perf_counter()
            print(f'FlatQuant layer {layer+1}/32',flush=True)
            original,_=load_layer_weights(MODEL,layer)
            state={k:v.cuda() for k,v in matrices[layer].items()}
            trans=transforms(state)
            inverse_errors=[]
            for key, matrix in state.items():
                if key.endswith(('.matrix_left', '.matrix_right', '.matrix')):
                    invkey=key+('_inv_t' if key.endswith('.matrix') else '_inv')
                    error=(matrix.double().T@state[invkey].double()-torch.eye(len(matrix),device='cuda')).abs().max().item()
                    inverse_errors.append(error)
            assert max(inverse_errors)<.001, inverse_errors
            w=transform_weights(original,state)
            ref=LlamaDecoderBlock(original,{},'fp16',kv_bits=16)
            ref.mode='reference'
            flat=make_block(w,trans,cfg,quantize_kv=False)
            probe=torch.randn(1,17,4096,device='cuda',dtype=torch.float16)*.1
            a,b=ref(probe,ref.cache(1,17)),flat(probe,flat.cache(1,17))
            rel=((a.float()-b.float()).norm()/a.float().norm()).item()
            assert rel<.01, (layer,rel)
            del ref,flat,probe,a,b,state,original
            path=FOLDER/'rotated'/f'layer_{layer:02d}.pt'
            torch.save(cpu_tree(dict(weights=w,transforms=trans)),path)
            ws=grouped_weights(w)
            block=make_block(w,trans,cfg)
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
            block=make_block(w,trans,cfg,plan,'cuda');cache=block.cache(1,2048)
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
            print(f'Saved {layer+1}/32: {logs[-1]["seconds"]:.1f}s, float error {rel:.6g}',flush=True)
            del w,ws,plan,block,cache,trans
            torch.cuda.empty_cache()
    manifest['status']='complete';manifest['extras_sha256']=sha(FOLDER/'extras.pt')
    (FOLDER/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (report/'calibration_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    main()
