"""Validate and summarize the full Qwen3 experiment."""
from qwen3_models import *
from eval_qa import sha
import numpy as np

def main():
    results={}
    for method in ('fp16','quarot','spinquant','duquant'):
        folder=REPORT/'results'/('fp16_final' if method=='fp16' else method)
        r=json.loads((folder/'run.json').read_text())
        assert r['status']=='complete' and r['completed_requests']==r['total_requests']==67106
        assert sha(folder/'scores.npy')==r['scores_sha256']
        assert np.isfinite(np.load(folder/'scores.npy')).all()
        assert len((folder/'predictions.jsonl').read_text().splitlines())==19965
        if results:
            for key in ('protocol','source_sha256','batch_cap','token_budget'):
                assert r['config'][key]==results['fp16']['config'][key],(method,key)
        if method!='fp16':
            m=json.loads((BASE/method/'manifest.json').read_text())
            assert m['status']=='complete' and len(m['layers'])==28
            assert sha(BASE/method/'manifest.json')==r['config']['calibration_manifest_sha256']
        results[method]=r
    checks=json.loads((REPORT/'correctness.json').read_text())
    assert all(x['exact'] for x in checks['integer_checks'].values())
    output=dict(status='complete',model=str(MODEL),setting='W4A4KV4; Q16',results=results,correctness=checks,
        caveats=['Qwen3 adapters are local implementations, not author-released Qwen3 model results.',
                 'SpinQuant uses learned R1/R2, 800 Cayley SGD updates on shared 128 train samples; original paper training data/batching differ.',
                 'Full causal prefill QA scoring does not establish cached-decode bitwise equivalence.',
                 'FP baseline uses FP16 conversion from original BF16 checkpoint; no chat template.',
                 'Wall times include compilation and Python batching, not controlled speed benchmarks.'])
    (REPORT/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    tasks=('piqa','arc_easy','arc_challenge','boolq','hellaswag','winogrande')
    lines=['| Method | PIQA | ARC-E | ARC-C | BoolQ | HellaSwag | WinoGrande | Mean |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for method,r in results.items():
        values=[r['tasks'][t]['primary']*100 for t in tasks]+[r['mean_accuracy']*100]
        lines.append('| '+method+' | '+' | '.join(f'{x:.2f}' for x in values)+' |')
    table='\n'.join(lines)+'\n';(REPORT/'table.md').write_text(table);print(table)

if __name__=='__main__':main()
