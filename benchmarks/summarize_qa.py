"""Validate complete, matched runs and write the six-task comparison."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path('benchmarks/qa_compare')
METHODS = ('fp16', 'quarot', 'spinquant', 'duquant')
TASKS = ('piqa', 'arc_easy', 'arc_challenge', 'boolq', 'hellaswag', 'winogrande')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    results = {}
    for method in METHODS:
        path = ROOT/'results'/method/'run.json'
        record = json.loads(path.read_text())
        assert record['status'] == 'complete', method
        assert record['config']['mode'] == 'cuda'
        assert record['config']['method'] == method
        assert record['config']['limit_requests'] == 0
        assert record['completed_requests'] == record['total_requests'] == 67106
        scores_path = path.parent/'scores.npy'
        assert sha(scores_path) == record['scores_sha256']
        scores = np.load(scores_path)
        assert scores.shape == (67106,) and np.isfinite(scores).all()
        if results:
            base = results['fp16']['config']
            for key in ('protocol', 'source_sha256', 'batch_cap', 'token_budget'):
                assert record['config'][key] == base[key], (method, key)
        results[method] = record
    duquant = json.loads(Path('benchmarks/cache/ptq_llama2_final/sym_gptq/manifest.json').read_text())
    for method in ('quarot', 'spinquant'):
        folder = Path('benchmarks/cache/qa_compare')/method
        metadata = json.loads((folder/'manifest.json').read_text())
        assert metadata['training_token_ids_sha256'] == duquant['token_ids_sha256']
        assert sha(folder/'manifest.json') == results[method]['config']['calibration_manifest_sha256']
        assert sha(folder/'extras.pt') == metadata['extras_sha256']
        for key in ('weight_symmetric', 'activation_symmetric', 'kv_symmetric',
                    'kv_groupsize', 'weight_clip_grid', 'activation_kv_clip_grid'):
            assert metadata[key] == duquant['ptq'][key]
    output = dict(status='complete', model=duquant['model'], setting='W4A4KV4; Q16',
                  evaluation='Full zero-shot continuation loglikelihood; true INT4 backend',
                  comparison='Shared quantizer, clipping search, GPTQ and evaluation; method-specific rotations',
                  caveats=['Official SpinQuant learned rotation has its own optimization data/budget; only subsequent GPTQ/clipping calibration is shared.',
                           'Controlled common-backend comparison, not a reproduction of original paper defaults.',
                           'Full causal prefill scoring; does not validate autoregressive KV4 cached-decode equivalence.',
                           'Elapsed evaluation times include compilation and batching overhead; not a controlled speed benchmark.'],
                  tasks=list(TASKS), results=results,
                  run_sha256={m:sha(ROOT/'results'/m/'run.json') for m in METHODS})
    (ROOT/'summary.json').write_text(json.dumps(output, indent=2)+'\n')
    lines=['| Method | PIQA | ARC-E | ARC-C | BoolQ | HellaSwag | WinoGrande | Mean |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for method in METHODS:
        r = results[method]
        values = [r['tasks'][t]['primary']*100 for t in TASKS]+[r['mean_accuracy']*100]
        lines.append('| '+method+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    table='\n'.join(lines)+'\n'
    (ROOT/'table.md').write_text(table)
    print(table)


if __name__ == '__main__':
    main()
