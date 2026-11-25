import json,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from torch.nn.attention import SDPBackend,sdpa_kernel
root=Path('benchmarks/cache/qa_compare')
docs=[json.loads(x) for x in (root/'documents.jsonl').read_text().splitlines()]
reqs=[json.loads(x) for x in (root/'requests.jsonl').read_text().splitlines()]
scores=np.load('benchmarks/qa_compare/results/fp16/scores.npy')
selected=[d for d in docs if d['index']<2]
model=AutoModelForCausalLM.from_pretrained('/mnt/public_models/llama2-7b-hf',torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
errors=[];rows=[]
with torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
 for d in selected:
  hs=[]
  for idx in d['requests']:
   r=reqs[idx];a,b=r['context'],r['continuation']
   logits=model(torch.tensor([(a+b)[:-1]],device='cuda'),use_cache=False).logits[0,len(a)-1:].float()
   value=logits.log_softmax(-1).gather(1,torch.tensor(b,device='cuda')[:,None]).double().sum().item()
   hs.append(value);errors.append(abs(value-scores[idx]))
  hs=np.asarray(hs);ss=scores[d['requests']];norm=np.asarray(d['normalizers'])
  rows.append(dict(task=d['task'],index=d['index'],hf=hs.tolist(),custom=ss.tolist(),raw_prediction_agrees=bool(hs.argmax()==ss.argmax()),normalized_prediction_agrees=bool((hs/norm).argmax()==(ss/norm).argmax())))
result=dict(max_absolute_continuation_loglikelihood_difference=max(errors),mean_absolute_difference=float(np.mean(errors)),all_raw_predictions_agree=all(x['raw_prediction_agrees'] for x in rows),all_normalized_predictions_agree=all(x['normalized_prediction_agrees'] for x in rows),questions=len(rows),requests=len(errors),rows=rows)
Path('benchmarks/qa_compare/hf_scorer_check.json').write_text(json.dumps(result,indent=2)+'\n')
print({k:v for k,v in result.items() if k!='rows'})
assert result['all_raw_predictions_agree'] and result['all_normalized_predictions_agree']
