"""Identical full zero-shot likelihood scoring for all comparison methods."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel
from comparison_models import load_model,BASE


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def request_length(r):return len(r['context'])+len(r['continuation'])-1


def batches(requests,cap=16,budget=8192):
    order=sorted(range(len(requests)),key=lambda i:(-request_length(requests[i]),i))
    result=[];start=0
    while start<len(order):
        width=math.ceil(request_length(requests[order[start]])/32)*32
        count=min(cap,max(1,budget//width),len(order)-start)
        result.append((order[start:start+count],width));start+=count
    return result


@torch.inference_mode()
def conditional_scores(model,requests,width,caches=None):
    """Score only continuation positions, including its first token exactly once."""
    device=model.embedding.device
    ids=torch.zeros((len(requests),width),device=device,dtype=torch.long)
    row,pos,target,lengths=[],[],[],[]
    for i,r in enumerate(requests):
        a,b=r['context'],r['continuation']
        assert a and b and request_length(r)<=width
        ids[i,:len(a)+len(b)-1]=torch.tensor((a+b)[:-1],device=device)
        row.extend([i]*len(b));pos.extend(range(len(a)-1,len(a)+len(b)-1));target.extend(b);lengths.append(len(b))
    if caches is None:caches=model.caches(len(requests),width)
    x=F.embedding(ids,model.embedding)
    for block,cache in zip(model.blocks,caches):x=block(x,cache)
    # Future right-padding cannot affect the causal positions selected here.
    x=x[torch.tensor(row,device=device),torch.tensor(pos,device=device)]
    x=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+model.eps)).half()*model.final_norm
    logits=F.linear(x,model.lm_head)
    logp=F.log_softmax(logits.float(),dim=-1).gather(1,torch.tensor(target,device=device)[:,None])[:,0]
    return torch.stack([chunk.double().sum() for chunk in logp.split(lengths)]).cpu().numpy()


def summarize(documents,scores,protocol):
    tasks={};predictions=[]
    for d in documents:
        values=scores[d['requests']]
        if not np.isfinite(values).all():raise ValueError('Incomplete scores')
        pred=(0 if values[0]>values[1] else 1) if d['task']=='boolq' else int(values.argmax())
        normalized=int((values/np.asarray(d['normalizers'])).argmax())
        row=dict(task=d['task'],index=d['index'],gold=d['gold'],prediction=pred,
                 prediction_norm=normalized,loglikelihood=values.tolist())
        predictions.append(row)
        t=tasks.setdefault(d['task'],dict(count=0,correct=0,correct_norm=0))
        t['count']+=1;t['correct']+=pred==d['gold'];t['correct_norm']+=normalized==d['gold']
    for name,t in tasks.items():
        t['acc']=t['correct']/t['count'];t['acc_norm']=t['correct_norm']/t['count']
        t['primary_metric']=protocol['primary_metrics'][name];t['primary']=t[t['primary_metric']]
        t['standard_error']=math.sqrt(t['primary']*(1-t['primary'])/(t['count']-1))
        assert t['count']==protocol['counts'][name]
    return tasks,predictions


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--method',required=True,choices=['fp16','duquant','quarot','spinquant'])
    p.add_argument('--mode',default='cuda',choices=['reference','cuda'])
    p.add_argument('--output',required=True);p.add_argument('--resume',action='store_true')
    p.add_argument('--limit-requests',type=int,default=0);p.add_argument('--batch-cap',type=int,default=16)
    args=p.parse_args()
    protocol=json.loads((BASE/'tasks.json').read_text())
    assert sha(BASE/'requests.jsonl')==protocol['requests_sha256']
    assert sha(BASE/'documents.jsonl')==protocol['documents_sha256']
    requests=[json.loads(s) for s in (BASE/'requests.jsonl').read_text().splitlines()]
    documents=[json.loads(s) for s in (BASE/'documents.jsonl').read_text().splitlines()]
    if args.limit_requests:requests=requests[:args.limit_requests]
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    source={str(p):sha(p) for p in [Path(__file__),Path('benchmarks/comparison_models.py'),*Path('duquant_cuda').glob('*.py')]}
    config=dict(method=args.method,mode=args.mode,protocol=protocol,source_sha256=source,
        batch_cap=args.batch_cap,token_budget=8192,limit_requests=args.limit_requests,
        arithmetic=('Original FP16 weights and activations, KV16, Flash SDPA' if args.method=='fp16' else
                    'FP16 dequantized GEMM, common decoder and Flash SDPA' if args.mode=='reference' else
                    'True UINT4 GEMM with symmetric signed W4 encoded at zero=8, packed KV4, Flash prefill'))
    if args.method!='fp16':
        path=Path('benchmarks/cache/ptq_llama2_final/sym_gptq/manifest.json') if args.method=='duquant' else BASE/args.method/'manifest.json'
        config['calibration_manifest_sha256']=sha(path)
        meta=json.loads(path.read_text());assert meta['status']=='complete'
        if args.method in ('quarot','spinquant'):
            for entry in meta['layers']:
                for kind in ('rotated','gptq'):
                    assert sha(path.parent/kind/f"layer_{entry['index']:02d}.pt")==entry[kind+'_sha256']
    scores=np.full(len(requests),np.nan,dtype=np.float64)
    if (out/'run.json').exists():
        if not args.resume:raise FileExistsError('Use --resume explicitly')
        previous=json.loads((out/'run.json').read_text())
        assert previous['config']==config,'Resume code or protocol changed'
        scores=np.load(out/'scores.npy')
    np.save(out/'scores.npy',scores)
    record=dict(status='running',config=config,completed_requests=int(np.isfinite(scores).sum()))
    (out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    torch.backends.cuda.matmul.allow_tf32=False
    model=load_model(args.method,args.mode)
    print('Model loaded',args.method,args.mode,flush=True)
    schedule=batches(requests,args.batch_cap);current_shape=None;caches=None;start=time.perf_counter()
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
                print(f"{args.method}/{args.mode} {record['completed_requests']}/{len(requests)} requests, batch {i+1}/{len(schedule)}, {record['elapsed_seconds']:.1f}s",flush=True)
    if args.limit_requests:
        record['status']='diagnostic_complete'
    else:
        tasks,preds=summarize(documents,scores,protocol)
        record.update(status='complete',tasks=tasks,mean_accuracy=sum(t['primary'] for t in tasks.values())/6)
        (out/'predictions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in preds))
        print(json.dumps(dict(tasks=tasks,mean_accuracy=record['mean_accuracy'])),flush=True)
    record['scores_sha256']=sha(out/'scores.npy')
    (out/'run.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__=='__main__':main()
