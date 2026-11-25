"""C4-new PPL with unchanged calibrated models and real INT4 backend."""
from eval_lambada_ppl import *

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--family',required=True,choices=['llama2','qwen3']);p.add_argument('--method',required=True,choices=['fp16','quarot','spinquant','duquant','flatquant']);p.add_argument('--resume',action='store_true');args=p.parse_args()
    folder=BASE/args.family;out=REPORT/args.family/args.method;out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((folder/'c4_protocol.json').read_text());prior,manifest=prior_and_manifest(args.family,args.method)
    tokens=torch.load(folder/'c4_tokens.pt',weights_only=True)
    assert hashlib.sha256(tokens.numpy().tobytes()).hexdigest()==protocol['token_ids_sha256']
    config=dict(family=args.family,method=args.method,protocol=protocol,mode='fp16' if args.method=='fp16' else 'true_int4',kv_bits=16 if args.method=='fp16' else 4,
        prior_qa_run_sha256=sha(prior),calibration_manifest_sha256=sha(manifest) if manifest else None,
        source_sha256={str(p):sha(p) for p in [Path(__file__),Path('benchmarks/eval_lambada_ppl.py'),Path('benchmarks/comparison_models.py'),Path('benchmarks/flatquant_models.py'),Path('benchmarks/qwen3_models.py'),Path('benchmarks/qwen3_flat_models.py'),*Path('duquant_cuda').glob('*.py')]})
    record=dict(status='running',config=config,blocks=[]);path=out/'c4.json'
    if path.exists():
        if not args.resume:raise FileExistsError('Explicit resume required')
        record=json.loads(path.read_text());assert record['config']==config
        if record['status']=='complete':return
    path.write_text(json.dumps(record,indent=2)+'\n')
    torch.backends.cuda.matmul.allow_tf32=False;model=load_model(args.family,args.method);caches=model.caches(1,2048);start=time.perf_counter()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for i in range(len(record['blocks']),protocol['blocks']):
            x=tokens[:,i*2048:(i+1)*2048].cuda();logits=model(x,caches,last_token_only=False)
            losses=F.cross_entropy(logits[:,:-1].float().reshape(-1,logits.shape[-1]),x[:,1:].reshape(-1),reduction='none')
            nll=losses.double().sum().item();assert math.isfinite(nll)
            record['blocks'].append(dict(index=i,nll_sum=nll,scored_tokens=2047));del logits,losses,x
            if i%20==0 or i==protocol['blocks']-1:
                path.write_text(json.dumps(record,indent=2)+'\n');print('C4',i+1,'/',protocol['blocks'],'PPL',math.exp(math.fsum(b['nll_sum'] for b in record['blocks'])/(len(record['blocks'])*2047)),flush=True)
    nll=math.fsum(b['nll_sum'] for b in record['blocks']);count=protocol['scored_tokens']
    record.update(status='complete',block_count=len(record['blocks']),scored_tokens=count,nll_sum=nll,ppl=math.exp(nll/count),elapsed_seconds=time.perf_counter()-start)
    path.write_text(json.dumps(record,indent=2)+'\n');print('Complete',args.family,args.method,record['ppl'],flush=True)

if __name__=='__main__':main()
