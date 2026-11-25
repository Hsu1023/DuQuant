"""Append validated FlatQuant Qwen3 results to the previous four-model table."""
from qwen3_flat_models import *
from eval_qa import sha
import numpy as np

def main():
    previous_path=REPORT/'summary.json';previous=json.loads(previous_path.read_text())
    path=FLAT_REPORT/'results/run.json';flat=json.loads(path.read_text());base=previous['results']['duquant']['config']
    assert flat['status']=='complete' and flat['completed_requests']==flat['total_requests']==67106
    assert flat['config']['mode']=='cuda' and flat['config']['limit_requests']==0
    for key in ('protocol','source_sha256','batch_cap','token_budget'):assert flat['config'][key]==base[key],key
    assert sha(FLAT_REPORT/'results/scores.npy')==flat['scores_sha256']
    scores=np.load(FLAT_REPORT/'results/scores.npy');assert scores.shape==(67106,) and np.isfinite(scores).all()
    assert len((FLAT_REPORT/'results/predictions.jsonl').read_text().splitlines())==19965
    m=json.loads((FOLDER/'manifest.json').read_text());assert m['status']=='complete' and len(m['layers'])==28
    assert sha(FOLDER/'manifest.json')==flat['config']['calibration_manifest_sha256']
    d=json.loads((BASE/'duquant/manifest.json').read_text())
    for key in ('training','weight_clip_grid','activation_kv_clip_grid','gptq'):assert m[key]==d[key],key
    check=json.loads((FLAT_REPORT/'integer_reference_check.json').read_text());assert check['all_exactly_equal']
    results={**previous['results'],'flatquant':flat}
    output=dict(status='complete',model=str(MODEL),setting='W4A4KV4; Q16',results=results,integer_reference_check=check,
        previous_summary_sha256=sha(previous_path),flatquant_minus_other_percentage_points={k:(flat['mean_accuracy']-v['mean_accuracy'])*100 for k,v in previous['results'].items()},
        caveats=previous['caveats']+['FlatQuant is a local Qwen3 adaptation trained for 15 epochs per layer using official SVD/quantizer modules; its learned clipping is replaced by common search before common GPTQ. Total optimization budgets differ.'])
    (FLAT_REPORT/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    tasks=('piqa','arc_easy','arc_challenge','boolq','hellaswag','winogrande')
    lines=['| Method | PIQA | ARC-E | ARC-C | BoolQ | HellaSwag | WinoGrande | Mean |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in results.items():
        vals=[r['tasks'][t]['primary']*100 for t in tasks]+[r['mean_accuracy']*100]
        lines.append('| '+name+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
    table='\n'.join(lines)+'\n';(FLAT_REPORT/'table.md').write_text(table);print(table)

if __name__=='__main__':main()
