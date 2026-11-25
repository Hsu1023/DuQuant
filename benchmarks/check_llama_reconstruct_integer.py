"""Full-model exact integer check for newly learned scalar clips/scales."""
from llama_reconstruct import *
TRIAL_REPORT=REPORT
from comparison_models import BASE
from eval_qa import conditional_scores
from torch.nn.attention import SDPBackend,sdpa_kernel
import numpy as np

@torch.inference_mode()
def main():
    requests=[json.loads(s) for s in (BASE/'requests.jsonl').read_text().splitlines()][:32]
    width=max(len(r['context'])+len(r['continuation'])-1 for r in requests)
    results={}
    for variant in ('rtn','gptq'):
        values={}
        for mode in ('cuda','integer_reference'):
            model=load_trial(mode,variant)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):values[mode]=conditional_scores(model,requests,width)
            del model;torch.cuda.empty_cache()
        a,b=values['cuda'],values['integer_reference']
        results[variant]=dict(requests=32,max_absolute_loglikelihood_difference=float(np.max(abs(a-b))),exact=bool(np.array_equal(a,b)))
    (TRIAL_REPORT/'integer_check.json').write_text(json.dumps(results,indent=2)+'\n');print(results,flush=True)
    assert all(v['exact'] for v in results.values())

if __name__=='__main__':main()
