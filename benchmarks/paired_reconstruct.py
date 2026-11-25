"""Paired question bootstrap for predeclared QA means; no model selection."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['qwen3','llama2'],required=True);args=p.parse_args()
    root=Path('benchmarks/qwen3_compare/duquant_reconstruct') if args.family=='qwen3' else Path('benchmarks/llama_reconstruct')
    flat=Path('benchmarks/qwen3_compare/flatquant/results') if args.family=='qwen3' else Path('benchmarks/flatquant_compare/results')
    base_rows=[json.loads(s) for s in (flat/'predictions.jsonl').read_text().splitlines()]
    base={(r['task'],r['index']):r for r in base_rows};assert len(base)==19965
    base_lam=np.load(Path('benchmarks/lambada_compare')/args.family/'flatquant/lambada_arrays.npz')['greedy'].astype(int)
    output=dict(family=args.family,replicates=4000,seed=0,interpretation='Paired bootstrap of questions independently within each fixed task; does not measure training-seed/model-family uncertainty.',comparisons={})
    for variant in ('rtn','gptq'):
        rows=[json.loads(s) for s in (root/variant/'results/predictions.jsonl').read_text().splitlines()]
        assert {(r['task'],r['index']) for r in rows}==set(base)
        diffs={}
        for r in rows:
            b=base[(r['task'],r['index'])];assert r['gold']==b['gold']
            metric='prediction_norm' if r['task'] in ('arc_challenge','hellaswag') else 'prediction'
            d=int(r[metric]==r['gold'])-int(b[metric]==b['gold']);diffs.setdefault(r['task'],[]).append(d)
        new_lam=np.load(root/variant/'lambada_arrays.npz')['greedy'].astype(int);assert new_lam.shape==base_lam.shape==(5153,)
        diffs['lambada_openai']=(new_lam-base_lam).tolist()
        rng=np.random.default_rng(0);replicates={};means={}
        for task,values in diffs.items():
            a=np.asarray(values);counts=np.array([(a==-1).sum(),(a==0).sum(),(a==1).sum()])
            draws=rng.multinomial(len(a),counts/len(a),size=4000)
            replicates[task]=(draws[:,2]-draws[:,0])/len(a);means[task]=float(a.mean())
        groups=dict(original_six=['piqa','arc_easy','arc_challenge','boolq','hellaswag','winogrande'],lambada_six=['piqa','arc_easy','arc_challenge','hellaswag','winogrande','lambada_openai'])
        groups['seven']=[*groups['original_six'],'lambada_openai'];groups['lambada_only']=['lambada_openai']
        result={}
        for name,tasks in groups.items():
            draws=sum(replicates[t] for t in tasks)/len(tasks)
            result[name]=dict(delta_percentage_points=sum(means[t] for t in tasks)/len(tasks)*100,
                paired_bootstrap_95ci_percentage_points=(np.quantile(draws,[.025,.975])*100).tolist())
        output['comparisons'][variant+'_minus_flatquant']=result
    (root/'paired_bootstrap.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))

if __name__=='__main__':main()
