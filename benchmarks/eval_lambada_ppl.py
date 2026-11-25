"""Full LAMBADA OpenAI word accuracy/PPL and WikiText-2 test PPL."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks/cache/qwen3_deps'))
sys.path.insert(0,str(ROOT))
import argparse,hashlib,json,math,time
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel
from eval_qa import batches,sha

BASE=ROOT/'benchmarks/cache/lambada_compare'
REPORT=ROOT/'benchmarks/lambada_compare'


def prior_and_manifest(family,method):
    if family=='llama2':
        prior=Path('benchmarks/flatquant_compare/results/run.json') if method=='flatquant' else Path('benchmarks/qa_compare/results')/method/'run.json'
        if method=='duquant':manifest=Path('benchmarks/cache/ptq_llama2_final/sym_gptq/manifest.json')
        else:manifest=Path('benchmarks/cache/qa_compare')/method/'manifest.json'
    else:
        prior=Path('benchmarks/qwen3_compare/flatquant/results/run.json') if method=='flatquant' else Path('benchmarks/qwen3_compare/results')/('fp16_final' if method=='fp16' else method)/'run.json'
        manifest=Path('benchmarks/cache/qwen3_compare')/method/'manifest.json'
    record=json.loads(prior.read_text());assert record['status']=='complete'
    for section in ('source_sha256','extension_source_sha256'):
        for p,digest in record['config'].get(section,{}).items():assert sha(p)==digest,('Previous inference sources changed',p)
    if method=='fp16':return prior,None
    assert sha(manifest)==record['config']['calibration_manifest_sha256']
    meta=json.loads(manifest.read_text());assert meta['status']=='complete'
    for e in meta['layers']:
        if 'rotated_sha256' in e:
            for kind in ('rotated','gptq'):assert sha(manifest.parent/kind/f"layer_{e['index']:02d}.pt")==e[kind+'_sha256']
        elif 'file' in e:assert sha(manifest.parent/e['file'])==e['sha256']
    if 'extras_sha256' in meta:assert sha(manifest.parent/'extras.pt')==meta['extras_sha256']
    return prior,manifest


def load_model(family,method):
    if family=='llama2':
        if method=='flatquant':
            from flatquant_models import load_model as loader
            return loader('cuda')
        from comparison_models import load_model as loader
        return loader(method,'cuda')
    if method=='flatquant':
        from qwen3_flat_models import load_flat_model
        return load_flat_model('cuda')
    from qwen3_models import load_model as loader
    return loader(method,'cuda')


@torch.inference_mode()
def score_words(model,requests,width,caches):
    ids=torch.zeros((len(requests),width),device='cuda',dtype=torch.long)
    rows,positions,targets,lengths=[],[],[],[]
    for i,r in enumerate(requests):
        a,b=r['context'],r['continuation']
        ids[i,:len(a)+len(b)-1]=torch.tensor((a+b)[:-1],device='cuda')
        rows.extend([i]*len(b));positions.extend(range(len(a)-1,len(a)+len(b)-1));targets.extend(b);lengths.append(len(b))
    hidden=F.embedding(ids,model.embedding)
    for block,cache in zip(model.blocks,caches):hidden=block(hidden,cache)
    x=hidden[torch.tensor(rows,device='cuda'),torch.tensor(positions,device='cuda')]
    x=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+model.eps)).half()*model.final_norm
    logits=F.linear(x,model.lm_head)
    target=torch.tensor(targets,device='cuda');prediction=logits.argmax(-1)
    logp=logits.float().log_softmax(-1).gather(1,target[:,None])[:,0]
    sums=torch.stack([z.double().sum() for z in logp.split(lengths)]).cpu().numpy()
    greedy=torch.stack([z.all() for z in (prediction==target).split(lengths)]).cpu().numpy()
    pred=[z.cpu().tolist() for z in prediction.split(lengths)]
    return sums,greedy,pred


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--family',required=True,choices=['llama2','qwen3']);p.add_argument('--method',required=True,choices=['fp16','duquant','quarot','spinquant','flatquant']);p.add_argument('--resume',action='store_true');args=p.parse_args()
    folder=BASE/args.family;out=REPORT/args.family/args.method;out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((folder/'protocol.json').read_text());assert sha(folder/'requests.jsonl')==protocol['lambada']['requests_sha256']
    prior,manifest=prior_and_manifest(args.family,args.method)
    source_paths=[Path(__file__),Path('benchmarks/eval_qa.py'),Path('benchmarks/comparison_models.py'),Path('benchmarks/flatquant_models.py'),Path('benchmarks/qwen3_models.py'),Path('benchmarks/qwen3_flat_models.py'),*Path('duquant_cuda').glob('*.py')]
    config=dict(family=args.family,method=args.method,mode='fp16' if args.method=='fp16' else 'true_int4',kv_bits=16 if args.method=='fp16' else 4,
        protocol=protocol,source_sha256={str(p):sha(p) for p in source_paths},prior_qa_run_sha256=sha(prior),
        calibration_manifest_sha256=sha(manifest) if manifest else None,batch_cap=16,token_budget=8192)
    record=dict(status='running',config=config,lambada={},wikitext={})
    if (out/'run.json').exists():
        if not args.resume:raise FileExistsError('Explicit --resume required')
        record=json.loads((out/'run.json').read_text());assert record['config']==config
        if record['status']=='complete':print('Already complete');return
    (out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    tokens=torch.load(folder/'wikitext_tokens.pt',weights_only=True)
    assert hashlib.sha256(tokens.numpy().tobytes()).hexdigest()==protocol['wikitext']['token_ids_sha256']
    torch.backends.cuda.matmul.allow_tf32=False
    model=load_model(args.family,args.method)
    print('Loaded',args.family,args.method,flush=True)
    requests=[json.loads(s) for s in (folder/'requests.jsonl').read_text().splitlines()]
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        if record['lambada'].get('status')!='complete':
            ll=np.full(5153,np.nan);greedy=np.zeros(5153,dtype=bool);predictions=[None]*5153
            schedule=batches(requests,16,8192);caches=None;shape=None;start=time.perf_counter()
            for i,(indices,width) in enumerate(schedule):
                if shape!=(len(indices),width):
                    del caches;caches=model.caches(len(indices),width);shape=(len(indices),width)
                values,flags,pred=score_words(model,[requests[j] for j in indices],width,caches)
                ll[indices]=values;greedy[indices]=flags
                for j,predicted in zip(indices,pred):predictions[j]=predicted
                if i%50==0 or i==len(schedule)-1:print('LAMBADA',i+1,'/',len(schedule),'seconds',time.perf_counter()-start,flush=True)
            assert np.isfinite(ll).all()
            nll=-math.fsum(ll.tolist());total_tokens=sum(len(r['continuation']) for r in requests)
            np.savez_compressed(out/'lambada_arrays.npz',loglikelihood=ll,greedy=greedy)
            rows=[dict(index=i,target_word=r['target_word'],target_token_ids=r['continuation'],predicted_token_ids=predictions[i],loglikelihood=float(ll[i]),correct=bool(greedy[i])) for i,r in enumerate(requests)]
            (out/'lambada_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            record['lambada']=dict(status='complete',count=5153,correct=int(greedy.sum()),accuracy=float(greedy.mean()),word_nll_sum=nll,
                word_ppl=math.exp(nll/5153),target_token_count=total_tokens,target_token_ppl=math.exp(nll/total_tokens),elapsed_seconds=time.perf_counter()-start,
                arrays_sha256=sha(out/'lambada_arrays.npz'))
            (out/'run.json').write_text(json.dumps(record,indent=2)+'\n');del caches
        start=time.perf_counter();caches=model.caches(1,2048)
        blocks=record['wikitext'].get('blocks',[]);count=protocol['wikitext']['blocks']
        for i in range(len(blocks),count):
            x=tokens[:,i*2048:(i+1)*2048].cuda()
            logits=model(x,caches,last_token_only=False)
            losses=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),x[:,1:].reshape(-1),reduction='none')
            nll=losses.double().sum().item();assert math.isfinite(nll)
            blocks.append(dict(index=i,nll_sum=nll,scored_tokens=2047))
            del logits,losses,x
            if i%20==0 or i==count-1:
                record['wikitext']=dict(status='running',blocks=blocks)
                (out/'run.json').write_text(json.dumps(record,indent=2)+'\n');print('WikiText',i+1,'/',count,'PPL',math.exp(math.fsum(z['nll_sum'] for z in blocks)/(len(blocks)*2047)),flush=True)
        nll=math.fsum(z['nll_sum'] for z in blocks)
        record['wikitext']=dict(status='complete',blocks=blocks,block_count=count,scored_tokens=count*2047,nll_sum=nll,ppl=math.exp(nll/(count*2047)),elapsed_seconds=time.perf_counter()-start)
    record['status']='complete';(out/'run.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(lambada=record['lambada'],wikitext_ppl=record['wikitext']['ppl'])),flush=True)

if __name__=='__main__':main()
