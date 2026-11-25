"""Merge disjoint training shards then produce both fixed evaluation exports."""
from llama_reconstruct import *
TRIAL_REPORT=REPORT
FOLDER=BASE
from search_ptq import sha
import time,subprocess

def main():
    target=BASE/'training'
    while True:
        try:
            parts=[json.loads((target/f'manifest_shard{r}.json').read_text()) for r in range(3)]
            ready=all(p['status']=='complete' for p in parts)
        except (FileNotFoundError,json.JSONDecodeError):ready=False
        if ready:break
        time.sleep(5)
    entries=sorted([e for p in parts for e in p['layers']],key=lambda e:e['index']);assert [e['index'] for e in entries]==list(range(32))
    for e in entries:assert sha(target/e['file'])==e['sha256']
    for part in parts[1:]:
        for key in ('training','source_sha256','epochs','batch_size','optimizer'):assert part[key]==parts[0][key]
    merged=dict(parts[0]);merged['layers']=entries;merged.pop('shard',None)
    (target/'manifest.json').write_text(json.dumps(merged,indent=2)+'\n');(TRIAL_REPORT/'training_manifest.json').write_text(json.dumps(merged,indent=2)+'\n')
    with open(TRIAL_REPORT/'export.log','w') as f:subprocess.run([sys.executable,'benchmarks/export_llama_reconstruct.py'],stdout=f,stderr=subprocess.STDOUT,check=True)
    with open(TRIAL_REPORT/'gptq_qa.log','w') as f:subprocess.run([sys.executable,'benchmarks/eval_llama_reconstruct.py','--variant','gptq','--output',str(TRIAL_REPORT/'gptq/results')],stdout=f,stderr=subprocess.STDOUT,check=True)
    print('Export and GPTQ QA complete',flush=True)

if __name__=='__main__':main()
