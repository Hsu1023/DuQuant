"""Append a validated FlatQuant run without changing the previous experiment."""
import json
from pathlib import Path
import numpy as np
from eval_qa import sha
from flatquant_models import FOLDER

ROOT=Path('benchmarks/flatquant_compare')
TASKS=('piqa','arc_easy','arc_challenge','boolq','hellaswag','winogrande')


def main():
    prior_path=Path('benchmarks/qa_compare/summary.json')
    previous=json.loads(prior_path.read_text())
    run_path=ROOT/'results/run.json'
    flat=json.loads(run_path.read_text())
    assert flat['status']=='complete' and flat['config']['mode']=='cuda'
    assert flat['config']['limit_requests']==0
    assert flat['completed_requests']==flat['total_requests']==67106
    base=previous['results']['duquant']['config']
    for key in ('protocol','source_sha256','batch_cap','token_budget'):
        assert flat['config'][key]==base[key],key
    assert sha(ROOT/'results/scores.npy')==flat['scores_sha256']
    scores=np.load(ROOT/'results/scores.npy')
    assert scores.shape==(67106,) and np.isfinite(scores).all()
    manifest=json.loads((FOLDER/'manifest.json').read_text())
    duquant=json.loads(Path('benchmarks/cache/ptq_llama2_final/sym_gptq/manifest.json').read_text())
    assert sha(FOLDER/'manifest.json')==flat['config']['calibration_manifest_sha256']
    assert manifest['training_token_ids_sha256']==duquant['token_ids_sha256']
    for key in ('weight_symmetric','activation_symmetric','kv_symmetric','kv_groupsize',
                'weight_clip_grid','activation_kv_clip_grid'):
        assert manifest[key]==duquant['ptq'][key],key
    results={**previous['results'],'flatquant':flat}
    delta={m:(flat['mean_accuracy']-r['mean_accuracy'])*100 for m,r in previous['results'].items()}
    summary=dict(status='complete',model=manifest['model'],setting='W4A4KV4; Q16',
        results=results,flatquant_minus_other_percentage_points=delta,
        previous_summary_sha256=sha(prior_path),flatquant_run_sha256=sha(run_path),
        learned_transforms=manifest['transformation_source'],
        caveats=previous['caveats']+[
            'FlatQuant preserves released non-orthogonal linear/K/V transforms and diagonals. Its released learned clipping is replaced by the common MSE search and GPTQ.',
            'FlatQuant transformation training has its own original optimization budget; this is not a reproduction of its paper-default quantizer.',
            'FlatQuant uses FP16 Kronecker transforms and true integer GEMM, with packed asymmetric KV4. Online transform fusion is not optimized.'])
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['| Method | PIQA | ARC-E | ARC-C | BoolQ | HellaSwag | WinoGrande | Mean |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method in ('fp16','quarot','spinquant','duquant','flatquant'):
        r=results[method]
        vals=[r['tasks'][t]['primary']*100 for t in TASKS]+[r['mean_accuracy']*100]
        lines.append('| '+method+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
    table='\n'.join(lines)+'\n';(ROOT/'table.md').write_text(table)
    print(table)
    print('FlatQuant minus other methods, percentage points:',delta)


if __name__=='__main__':
    main()
