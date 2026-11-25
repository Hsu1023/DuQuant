"""Report every predeclared metric, preserving baseline and both exports."""
from llama_reconstruct import *
TRIAL_REPORT=REPORT
from eval_qa import sha
import numpy as np,math


def main():
    baseline=json.loads((ROOT/'benchmarks/flatquant_compare/summary.json').read_text())
    quality_baseline=json.loads((ROOT/'benchmarks/lambada_compare/summary.json').read_text())['families']['llama2']['results']
    results={}
    for name in ('fp16','duquant','flatquant'):
        results[name]=dict(qa=baseline['results'][name],quality=quality_baseline[name]['run'],c4_ppl=quality_baseline[name]['c4']['ppl'],source='previous unchanged checkpoint')
    check=json.loads((TRIAL_REPORT/'integer_check.json').read_text());assert all(v['exact'] for v in check.values())
    for variant in ('rtn','gptq'):
        folder=TRIAL_REPORT/variant;qa=json.loads((folder/'results/run.json').read_text());quality=json.loads((folder/'quality.json').read_text())
        assert qa['status']==quality['status']=='complete'
        assert qa['completed_requests']==67106 and qa['config']['protocol']==baseline['results']['duquant']['config']['protocol']
        assert sha(folder/'results/scores.npy')==qa['scores_sha256']
        assert np.isfinite(np.load(folder/'results/scores.npy')).all()
        assert quality['lambada']['count']==5153
        assert quality['wikitext']['scored_tokens']==339802 and quality['c4']['scored_tokens']==524032
        assert sha(BASE/variant/'manifest.json')==qa['config']['calibration_manifest_sha256']==quality['calibration_manifest_sha256']
        results['reconstruct_'+variant]=dict(qa=qa,quality=quality,c4_ppl=quality['c4']['ppl'],source='new learned folding scales/clipping; parent rotations/permutation frozen')
    rows=[]
    for name,r in results.items():
        qa=r['qa'];q=r['quality'];old6=qa['mean_accuracy'];la=q['lambada']['accuracy']
        m6=(sum(qa['tasks'][n]['primary'] for n in ('piqa','arc_easy','arc_challenge','hellaswag','winogrande'))+la)/6
        rows.append(dict(method=name,six_original_mean=old6,lambada_six_mean=m6,seven_mean=(old6*6+la)/7,
            lambada_accuracy=la,wikitext_ppl=q['wikitext']['ppl'],c4_ppl=r['c4_ppl']))
    output=dict(status='complete',model=str(MODEL),setting='W4A4KV4; Q16; learned extra scales folded offline; same kernel operations',rows=rows,results=results,integer_check=check,
        caveats=['FlatQuant baseline uses the author-released LLaMA-2 transforms with common clipping search/GPTQ; it is not its untouched author-default quantizer.',
                 'The DuQuant extension learns clipping and additional folding scales; this changes calibration strategy and adds optimization budget versus original DuQuant.',
                 'Both RTN/GPTQ exports and all predeclared metrics are reported; no test-based hyperparameter or epoch selection.',
                 'Layer reconstruction uses FP16 rounded forward values with FP32 STE intermediate gradients and grad norm clip10 to prevent nonfinite backward steps.'])
    (TRIAL_REPORT/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    lines=['| Method | Original six mean (%) | LAMBADA six mean (%) | LAMBADA acc (%) | WikiText PPL | C4 PPL |','|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['method']} | {r['six_original_mean']*100:.2f} | {r['lambada_six_mean']*100:.2f} | {r['lambada_accuracy']*100:.2f} | {r['wikitext_ppl']:.3f} | {r['c4_ppl']:.3f} |")
    table='\n'.join(lines)+'\n';(TRIAL_REPORT/'table.md').write_text(table);print(table)

if __name__=='__main__':main()
