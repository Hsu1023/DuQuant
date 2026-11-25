"""Merge disjoint layer training shards, then calibrate and evaluate."""
from qwen3_flat_models import *
from search_ptq import sha
import time,subprocess

def main():
    target=BASE/'flatquant_training'
    while True:
        try:
            parts=[json.loads((target/f'manifest_shard{r}.json').read_text()) for r in range(3)]
            ready=all(p['status']=='complete' for p in parts)
        except (FileNotFoundError,json.JSONDecodeError):ready=False
        if ready:break
        time.sleep(5)
    merged=dict(parts[0]);entries=sorted([e for p in parts for e in p['layers']],key=lambda e:e['index'])
    assert [e['index'] for e in entries]==list(range(28))
    for e in entries:assert sha(target/e['file'])==e['sha256']
    for p in parts[1:]:
        for key in ('training','source_sha256','epochs','batch_size'):assert p[key]==parts[0][key]
    merged['layers']=entries;merged['shards']=3;merged.pop('shard',None)
    (target/'manifest.json').write_text(json.dumps(merged,indent=2)+'\n');(FLAT_REPORT/'training_manifest.json').write_text(json.dumps(merged,indent=2)+'\n')
    for script,name in [('benchmarks/calibrate_flat_qwen3.py','calibration'),('benchmarks/eval_flat_qwen3.py','qa')]:
        with open(FLAT_REPORT/f'{name}.log','w') as f:subprocess.run([sys.executable,script],stdout=f,stderr=subprocess.STDOUT,check=True)
    print('FlatQuant Qwen3 full INT4 QA complete',flush=True)

if __name__=='__main__':main()
