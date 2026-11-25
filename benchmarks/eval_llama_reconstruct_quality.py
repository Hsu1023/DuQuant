"""Predeclared full LAMBADA and WikiText/C4 evaluation of both trial exports."""
from llama_reconstruct import *
TRIAL_REPORT=REPORT
FOLDER=BASE
import argparse,hashlib,math,time,numpy as np
from eval_qa import batches,sha
from eval_lambada_ppl import score_words
from torch.nn.attention import SDPBackend,sdpa_kernel

DATA=ROOT/'benchmarks/cache/lambada_compare/llama2'


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['rtn','gptq'],required=True);args=p.parse_args()
    out=TRIAL_REPORT/args.variant;out.mkdir(exist_ok=True)
    if (out/'quality.json').exists():raise FileExistsError('Do not overwrite evaluation')
    meta=json.loads((FOLDER/args.variant/'manifest.json').read_text());assert meta['status']=='complete'
    for e in meta['layers']:assert sha(FOLDER/args.variant/e['file'])==e['sha256']
    protocol=json.loads((DATA/'protocol.json').read_text());assert sha(DATA/'requests.jsonl')==protocol['lambada']['requests_sha256']
    torch.backends.cuda.matmul.allow_tf32=False;model=load_trial('cuda',args.variant)
    record=dict(status='running',variant=args.variant,protocol=protocol,calibration_manifest_sha256=sha(FOLDER/args.variant/'manifest.json'),
        source_sha256={str(p):sha(p) for p in [Path(__file__),Path('benchmarks/llama_reconstruct.py'),Path('benchmarks/duquant_reconstruct.py'),Path('benchmarks/eval_lambada_ppl.py'),*Path('duquant_cuda').glob('*.py')]})
    requests=[json.loads(s) for s in (DATA/'requests.jsonl').read_text().splitlines()]
    ll=np.full(5153,np.nan);flags=np.zeros(5153,dtype=bool);preds=[None]*5153;caches=None;shape=None;start=time.perf_counter()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for i,(indices,width) in enumerate(batches(requests,16,8192)):
            if shape!=(len(indices),width):
                del caches;caches=model.caches(len(indices),width);shape=(len(indices),width)
            values,greedy,pred=score_words(model,[requests[j] for j in indices],width,caches)
            ll[indices]=values;flags[indices]=greedy
            for j,v in zip(indices,pred):preds[j]=v
        assert np.isfinite(ll).all();nll=-math.fsum(ll.tolist());np.savez_compressed(out/'lambada_arrays.npz',loglikelihood=ll,greedy=flags)
        record['lambada']=dict(count=5153,accuracy=float(flags.mean()),correct=int(flags.sum()),word_ppl=math.exp(nll/5153),nll_sum=nll,
            elapsed_seconds=time.perf_counter()-start,arrays_sha256=sha(out/'lambada_arrays.npz'))
        (out/'lambada_predictions.jsonl').write_text(''.join(json.dumps(dict(index=i,target=r['continuation'],prediction=preds[i],correct=bool(flags[i]),loglikelihood=float(ll[i])))+'\n' for i,r in enumerate(requests)))
        (out/'quality.json').write_text(json.dumps(record,indent=2)+'\n');del caches
        for name,tokenfile,proto in [('wikitext','wikitext_tokens.pt',protocol['wikitext']),('c4','c4_tokens.pt',json.loads((DATA/'c4_protocol.json').read_text()))]:
            tokens=torch.load(DATA/tokenfile,weights_only=True);assert hashlib.sha256(tokens.numpy().tobytes()).hexdigest()==proto['token_ids_sha256']
            blocks=[];caches=model.caches(1,2048);start=time.perf_counter()
            for i in range(proto['blocks']):
                ids=tokens[:,i*2048:(i+1)*2048].cuda();logits=model(ids,caches,last_token_only=False)
                loss=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction='none')
                value=loss.double().sum().item();assert math.isfinite(value);blocks.append(dict(index=i,nll_sum=value,scored_tokens=2047));del ids,logits,loss
                if i%20==0:print(args.variant,name,i+1,'/',proto['blocks'],flush=True)
            nll=math.fsum(b['nll_sum'] for b in blocks)
            record[name]=dict(ppl=math.exp(nll/proto['scored_tokens']),nll_sum=nll,scored_tokens=proto['scored_tokens'],protocol=proto,blocks=blocks,elapsed_seconds=time.perf_counter()-start)
            (out/'quality.json').write_text(json.dumps(record,indent=2)+'\n');del caches
    record['status']='complete';(out/'quality.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k in ['status','variant','lambada']}),flush=True)

if __name__=='__main__':main()
