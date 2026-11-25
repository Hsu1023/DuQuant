"""Evaluate DuQuant reconstruction using the unchanged six-task scorer and frozen requests."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from llama_reconstruct import BASE as FOLDER,REPORT as TRIAL_REPORT,load_trial,PARENT
from comparison_models import BASE
import eval_qa

from eval_qa import sha, batches, conditional_scores, summarize


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=['rtn','gptq'],required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['cuda','integer_reference'],default='cuda')
    p.add_argument('--limit-requests',type=int,default=0)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    folder=FOLDER/args.variant
    previous=json.loads(Path('benchmarks/qa_compare/results/duquant/run.json').read_text())
    common=previous['config']
    # Prove the actual shared code is still exactly the version used previously.
    for path,digest in common['source_sha256'].items():
        assert sha(path)==digest,('Shared evaluation/backend source changed',path)
    protocol=json.loads((BASE/'tasks.json').read_text())
    assert protocol==common['protocol']
    for name in ('requests','documents'):
        assert sha(BASE/f'{name}.jsonl')==protocol[f'{name}_sha256']
    requests=[json.loads(s) for s in (BASE/'requests.jsonl').read_text().splitlines()]
    docs=[json.loads(s) for s in (BASE/'documents.jsonl').read_text().splitlines()]
    if args.limit_requests:requests=requests[:args.limit_requests]
    meta=json.loads((folder/'manifest.json').read_text())
    assert meta['status']=='complete' and len(meta['layers'])==32
    assert sha(PARENT/'manifest.json')==meta['parent_manifest_sha256']
    for entry in meta['layers']:assert sha(folder/entry['file'])==entry['sha256']
    config=dict(method='duquant_reconstruct_'+args.variant,mode=args.mode,protocol=protocol,
        source_sha256=common['source_sha256'],
        extension_source_sha256={str(p):sha(p) for p in [Path(__file__),Path('benchmarks/llama_reconstruct.py'),Path('benchmarks/duquant_reconstruct.py')]},
        batch_cap=common['batch_cap'],token_budget=common['token_budget'],limit_requests=args.limit_requests,
        calibration_manifest_sha256=sha(folder/'manifest.json'),
        arithmetic=('True UINT4 GEMM with symmetric signed W4 encoded at zero=8; packed asymmetric KV4; DuQuant frozen transforms; trained folding scales/clipping'
                    if args.mode=='cuda' else 'Independent exact integer-dot reference with dense dequantized KV4'))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    scores=np.full(len(requests),np.nan,dtype=np.float64)
    if (out/'run.json').exists():
        if not args.resume:raise FileExistsError('Use --resume explicitly')
        old=json.loads((out/'run.json').read_text());assert old['config']==config
        scores=np.load(out/'scores.npy')
    record=dict(status='running',config=config,completed_requests=int(np.isfinite(scores).sum()))
    np.save(out/'scores.npy',scores)
    (out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    torch.backends.cuda.matmul.allow_tf32=False
    model=load_trial(args.mode,args.variant)
    print('Model loaded: DuQuant reconstruction',args.mode,flush=True)
    schedule=batches(requests,config['batch_cap'],config['token_budget'])
    caches=None;current_shape=None;start=time.perf_counter()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for i,(indices,width) in enumerate(schedule):
            if np.isfinite(scores[indices]).all():continue
            shape=(len(indices),width)
            if shape!=current_shape:
                del caches
                caches=model.caches(*shape);current_shape=shape
            scores[indices]=conditional_scores(model,[requests[j] for j in indices],width,caches)
            if not np.isfinite(scores[indices]).all():raise FloatingPointError('Nonfinite loglikelihood')
            if i%50==0 or i==len(schedule)-1:
                record.update(completed_requests=int(np.isfinite(scores).sum()),total_requests=len(requests),
                              elapsed_seconds=time.perf_counter()-start)
                np.save(out/'scores.npy',scores)
                (out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
                print(f"DuQuant reconstruction/{args.mode} {record['completed_requests']}/{len(requests)}, batch {i+1}/{len(schedule)}, {record['elapsed_seconds']:.1f}s",flush=True)
    if args.limit_requests:
        record['status']='diagnostic_complete'
    else:
        tasks,preds=summarize(docs,scores,protocol)
        record.update(status='complete',tasks=tasks,mean_accuracy=sum(t['primary'] for t in tasks.values())/6)
        (out/'predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in preds))
        print(json.dumps(dict(tasks=tasks,mean_accuracy=record['mean_accuracy'])),flush=True)
    record['scores_sha256']=sha(out/'scores.npy')
    (out/'run.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':
    main()
