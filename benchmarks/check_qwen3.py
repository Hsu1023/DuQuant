"""Native-model scorer alignment and complete-model integer oracle checks."""
from qwen3_models import *
import numpy as np
from eval_qa import conditional_scores
from torch.nn.attention import SDPBackend,sdpa_kernel
from transformers import AutoModelForCausalLM
import torch.nn.functional as F

@torch.inference_mode()
def main():
    requests=[json.loads(s) for s in (BASE/'requests.jsonl').read_text().splitlines()]
    docs=[json.loads(s) for s in (BASE/'documents.jsonl').read_text().splitlines()]
    selected=[d for d in docs if d['index']==0]
    hf=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
    expected=[];indices=[]
    for d in selected:
        for idx in d['requests']:
            r=requests[idx];a,b=r['context'],r['continuation']
            logits=hf(torch.tensor([(a+b)[:-1]],device='cuda'),use_cache=False).logits[0,len(a)-1:].float()
            expected.append(logits.log_softmax(-1).gather(1,torch.tensor(b,device='cuda')[:,None]).double().sum().item());indices.append(idx)
    del hf;torch.cuda.empty_cache()
    custom=load_model('fp16');actual=[]
    for idx in indices:
        r=requests[idx];actual.extend(conditional_scores(custom,[r],len(r['context'])+len(r['continuation'])-1))
    error=float(np.max(np.abs(np.array(expected)-actual)))
    rows=[];start=0
    for d in selected:
        n=len(d['requests']);a=np.array(expected[start:start+n]);b=np.array(actual[start:start+n]);norm=np.array(d['normalizers']);start+=n
        rows.append(dict(task=d['task'],raw_agrees=bool(a.argmax()==b.argmax()),normalized_agrees=bool((a/norm).argmax()==(b/norm).argmax())))
    result=dict(native_check=dict(requests=len(indices),max_loglikelihood_difference=error,questions=rows),integer_checks={})
    del custom;torch.cuda.empty_cache()
    for method in ('duquant','quarot','spinquant'):
        model=load_model(method);r=[requests[i] for i in indices[:4]];width=max(len(x['context'])+len(x['continuation'])-1 for x in r)
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):a=conditional_scores(model,r,width)
        del model;torch.cuda.empty_cache();model=load_model(method,'integer_reference')
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):b=conditional_scores(model,r,width)
        result['integer_checks'][method]=dict(requests=len(r),max_loglikelihood_difference=float(np.max(abs(a-b))),exact=bool(np.array_equal(a,b)))
        del model;torch.cuda.empty_cache()
    (REPORT/'correctness.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
    assert all(x['raw_agrees'] and x['normalized_agrees'] for x in rows)
    assert all(x['exact'] for x in result['integer_checks'].values())

if __name__=='__main__':main()
