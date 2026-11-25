"""Validate complete LAMBADA/PPL runs and append task-composition means."""
from pathlib import Path
import json,hashlib,math
import numpy as np
ROOT=Path('benchmarks/lambada_compare')
METHODS=('fp16','quarot','spinquant','duquant','flatquant')
FIVE=('piqa','arc_easy','arc_challenge','hellaswag','winogrande')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    all_results={};tables=[]
    for family,prior in [('llama2',Path('benchmarks/flatquant_compare/summary.json')),('qwen3',Path('benchmarks/qwen3_compare/flatquant/summary.json'))]:
        previous=json.loads(prior.read_text());results={};c4_reference=None
        for method in METHODS:
            folder=ROOT/family/method;r=json.loads((folder/'run.json').read_text())
            assert r['status']=='complete' and r['lambada']['count']==5153
            protocol=r['config']['protocol']
            if results:
                base=results['fp16']['run']['config']
                for key in ('protocol','source_sha256','batch_cap','token_budget'):assert r['config'][key]==base[key],(family,method,key)
            a=r['lambada'];w=r['wikitext']
            assert a['status']=='complete' and w['status']=='complete'
            assert sha(folder/'lambada_arrays.npz')==a['arrays_sha256']
            arrays=np.load(folder/'lambada_arrays.npz');ll=arrays['loglikelihood'];greedy=arrays['greedy']
            assert ll.shape==greedy.shape==(5153,) and np.isfinite(ll).all()
            assert a['correct']==int(greedy.sum()) and a['accuracy']==float(greedy.mean())
            predictions=[json.loads(s) for s in (folder/'lambada_predictions.jsonl').read_text().splitlines()]
            assert len(predictions)==5153 and [p['index'] for p in predictions]==list(range(5153))
            assert all(p['correct']==(p['target_token_ids']==p['predicted_token_ids']) for p in predictions)
            assert math.isclose(a['word_ppl'],math.exp(-math.fsum(ll.tolist())/5153),rel_tol=1e-12)
            assert w['block_count']==protocol['wikitext']['blocks']==len(w['blocks'])
            assert w['scored_tokens']==protocol['wikitext']['scored_tokens']
            assert [b['index'] for b in w['blocks']]==list(range(w['block_count']))
            assert math.isclose(w['ppl'],math.exp(math.fsum(b['nll_sum'] for b in w['blocks'])/w['scored_tokens']),rel_tol=1e-12)
            c4=json.loads((folder/'c4.json').read_text())
            assert c4['status']=='complete' and c4['block_count']==c4['config']['protocol']['blocks']==len(c4['blocks'])
            assert c4['scored_tokens']==c4['config']['protocol']['scored_tokens']
            assert c4['config']['prior_qa_run_sha256']==r['config']['prior_qa_run_sha256']
            assert c4['config']['calibration_manifest_sha256']==r['config']['calibration_manifest_sha256']
            if c4_reference is None:c4_reference=c4['config']
            else:
                for key in ('protocol','source_sha256'):assert c4['config'][key]==c4_reference[key],(family,method,key)
            assert math.isclose(c4['ppl'],math.exp(math.fsum(b['nll_sum'] for b in c4['blocks'])/c4['scored_tokens']),rel_tol=1e-12)
            old=previous['results'][method]
            assert old['status']=='complete'
            # Existing QA source/protocol is preserved. New LAMBADA follows full-text
            # token boundaries, unlike the old separately encoded choice strings.
            five=sum(old['tasks'][t]['primary'] for t in FIVE)
            results[method]=dict(run=r,c4=c4,c4_sha256=sha(folder/'c4.json'),run_sha256=sha(folder/'run.json'),prior_mean_six=old['mean_accuracy'],
                lambada_six_mean=(five+a['accuracy'])/6,
                seven_task_mean=(five+old['tasks']['boolq']['primary']+a['accuracy'])/7)
        all_results[family]=dict(prior_summary_sha256=sha(prior),results=results)
        lines=[f'### {family}', '', '| Method | LAMBADA acc (%) | LAMBADA word PPL | WikiText-2 PPL | C4 PPL | LAMBADA six-task mean (%) |',
               '|---|---:|---:|---:|---:|---:|']
        for method,x in results.items():
            r=x['run'];a=r['lambada'];w=r['wikitext']
            lines.append(f"| {method} | {a['accuracy']*100:.2f} | {a['word_ppl']:.3f} | {w['ppl']:.3f} | {x['c4']['ppl']:.3f} | {x['lambada_six_mean']*100:.2f} |")
        tables.append('\n'.join(lines))
    result=dict(status='complete',families=all_results,
        six_task_composition=list(FIVE)+['lambada_openai'],seven_task_composition=list(FIVE)+['boolq','lambada_openai'],
        caveats=['Task composition matches FlatQuant default list; this does not reproduce every official prompt/tokenizer/metric/quantization default.',
                 'Prior five QA scores retain their frozen legacy separate context/choice tokenization. LAMBADA uses a full-text/context-prefix suffix to avoid a spurious standalone SentencePiece space token.',
                 'LAMBADA word PPL only scores the target word; WikiText PPL scores shifted tokens in full disjoint 2048 blocks. These PPLs have different denominators.',
                 'Quantized rows are the unchanged real W4A4KV4 models; transformations/training budgets differ as documented in prior experiments.'])
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    table='\n\n'.join(tables)+'\n';(ROOT/'table.md').write_text(table);print(table)

if __name__=='__main__':main()
