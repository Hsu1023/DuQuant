"""Freeze six full zero-shot tasks using the repository's lm-eval templates.

Local source archives are read only. All methods consume the identical saved
requests. Raw and character-length-normalized accuracy are both retained.
"""
import hashlib
import json
from pathlib import Path
import re
import zipfile
import pyarrow.parquet as pq
from transformers import AutoTokenizer

SOURCE = Path('/mnt/yichen/wyc/QDLM/data/dataset/dataset')
OUT = Path('benchmarks/cache/qa_compare')
EXPECTED = dict(piqa=1838, arc_easy=2376, arc_challenge=1172, boolq=3270,
                hellaswag=10042, winogrande=1267)
PRIMARY = dict(piqa='acc', arc_easy='acc', arc_challenge='acc_norm', boolq='acc',
               hellaswag='acc_norm', winogrande='acc')


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    docs, sources = [], {}
    def source(p):
        sources[str(p)] = digest(p)
        return p
    def zipped(p, member):
        with zipfile.ZipFile(source(p)) as z:
            return z.read(member).decode().splitlines()
    def add(task, idx, contexts, endings, gold, norms=None):
        assert 0 <= gold < len(endings)
        docs.append(dict(task=task, index=idx, contexts=contexts, endings=endings,
                         gold=gold, normalizers=norms or [len(x.lstrip()) for x in endings]))
    p=SOURCE/'piqa/physicaliqa-train-dev.zip'
    rows=zipped(p,'physicaliqa-train-dev/dev.jsonl')
    labels=zipped(p,'physicaliqa-train-dev/dev-labels.lst')
    assert len(rows)==len(labels)
    for i,(raw,label) in enumerate(zip(rows,labels)):
        d=json.loads(raw); choices=[d['sol1'],d['sol2']]
        add('piqa',i,['Question: '+d['goal']+'\nAnswer:']*2,[' '+c for c in choices],int(label),list(map(len,choices)))
    for task,name in [('arc_easy','ARC-Easy'),('arc_challenge','ARC-Challenge')]:
        rows=pq.read_table(source(SOURCE/f'ai2_arc/{name}/test-00000-of-00001.parquet')).to_pylist()
        for i,d in enumerate(rows):
            choices=d['choices']['text']
            gold=d['choices']['label'].index(d['answerKey'])
            add(task,i,['Question: '+d['question']+'\nAnswer:']*len(choices),[' '+c for c in choices],gold,list(map(len,choices)))
    for i,raw in enumerate(zipped(SOURCE/'super_glue/BoolQ.zip','BoolQ/val.jsonl')):
        d=json.loads(raw)
        add('boolq',i,[f"{d['passage']}\nQuestion: {d['question']}?\nAnswer:"]*2,[' yes',' no'],0 if d['label'] else 1)
    def clean(s):
        return re.sub(r'\[.*?\]','',s.strip().replace(' [title]','. ')).replace('  ',' ')
    p=source(SOURCE/'hellaswag/hellaswag_val.jsonl')
    for i,raw in enumerate(p.read_text().splitlines()):
        d=json.loads(raw)
        ctx=clean(d['activity_label']+': '+d['ctx_a']+' '+d['ctx_b'].capitalize())
        choices=[clean(c) for c in d['endings']]
        add('hellaswag',i,[ctx]*len(choices),[' '+c for c in choices],int(d['label']),list(map(len,choices)))
    p=SOURCE/'winogrande/winogrande_1.1.zip'
    rows=zipped(p,'winogrande_1.1/dev.jsonl');labels=zipped(p,'winogrande_1.1/dev-labels.lst')
    assert len(rows)==len(labels)
    for i,(raw,label) in enumerate(zip(rows,labels)):
        d=json.loads(raw); left,right=d['sentence'].split('_')
        add('winogrande',i,[left+d['option1'],left+d['option2']],[' '+right.strip()]*2,int(label)-1)
    counts={t:sum(d['task']==t for d in docs) for t in EXPECTED}
    assert counts==EXPECTED,counts
    tokenizer=AutoTokenizer.from_pretrained('/mnt/public_models/llama2-7b-hf',use_fast=False,local_files_only=True)
    requests=[]
    for docid,doc in enumerate(docs):
        doc['requests']=[]
        for c,(ctx,end) in enumerate(zip(doc['contexts'],doc['endings'])):
            # Exactly the legacy lm-eval causal protocol: tokenize separately,
            # no BOS/EOS; continuation leading whitespace is preserved.
            a=tokenizer.encode(ctx,add_special_tokens=False)
            b=tokenizer.encode(end,add_special_tokens=False)
            assert a and b
            if len(a)+len(b)-1>4096:
                raise ValueError('Task exceeds context; choose and record an explicit truncation policy')
            doc['requests'].append(len(requests))
            requests.append(dict(doc=docid,choice=c,context=a,continuation=b))
    for name,values in [('documents',docs),('requests',requests)]:
        (OUT/f'{name}.jsonl').write_text(''.join(json.dumps(d,ensure_ascii=False)+'\n' for d in values))
    meta=dict(status='complete',counts=counts,total_questions=len(docs),total_requests=len(requests),
              max_input_tokens=max(len(r['context'])+len(r['continuation'])-1 for r in requests),
              primary_metrics=PRIMARY,num_fewshot=0,context_length=4096,truncated_requests=0,
              tokenization='LlamaTokenizer use_fast=False; separately encode context and continuation; add_special_tokens=False; no BOS/EOS',
              scoring='Causal loglikelihood of complete continuation only; right padding ignored; acc_norm divides by original choice character count; BoolQ ties choose no',
              splits={t:'test' if t.startswith('arc') else 'validation' for t in EXPECTED},
              source_sha256=sources,documents_sha256=digest(OUT/'documents.jsonl'),requests_sha256=digest(OUT/'requests.jsonl'),
              template_sha256={p:digest(p) for p in ['lm_eval/tasks/piqa.py','lm_eval/tasks/arc.py','lm_eval/tasks/superglue.py','lm_eval/tasks/hellaswag.py','lm_eval/tasks/winogrande.py','lm_eval/base.py']})
    (OUT/'tasks.json').write_text(json.dumps(meta,indent=2)+'\n')
    Path('benchmarks/qa_compare').mkdir(exist_ok=True)
    Path('benchmarks/qa_compare/tasks.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({k:meta[k] for k in ['counts','total_questions','total_requests','max_input_tokens']},indent=2))


if __name__=='__main__':
    main()
