"""Reuse frozen question templates, retokenize with Qwen3; train split only."""
import hashlib
import json
import random
import torch
from datasets import Dataset
from qwen3_models import MODEL,BASE,REPORT
from transformers import AutoTokenizer
from calibrate_model import TRAIN_ARROW


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    BASE.mkdir(parents=True,exist_ok=True);REPORT.mkdir(exist_ok=True)
    tok=AutoTokenizer.from_pretrained(MODEL,use_fast=True,local_files_only=True)
    from pathlib import Path
    src=Path('benchmarks/cache/qa_compare')
    docs=[json.loads(s) for s in (src/'documents.jsonl').read_text().splitlines()]
    reqs=[]
    for i,d in enumerate(docs):
        d['requests']=[]
        for j,(ctx,end) in enumerate(zip(d['contexts'],d['endings'])):
            a,b=tok.encode(ctx,add_special_tokens=False),tok.encode(end,add_special_tokens=False)
            assert a and b and len(a)+len(b)-1<=4096
            d['requests'].append(len(reqs));reqs.append(dict(doc=i,choice=j,context=a,continuation=b))
    for name,items in [('documents',docs),('requests',reqs)]:
        (BASE/f'{name}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in items))
    meta=json.loads((src/'tasks.json').read_text())
    meta.update(tokenization='Qwen3 fast tokenizer; separate context and continuation; no special tokens; no chat template',
                model=str(MODEL),max_input_tokens=max(len(x['context'])+len(x['continuation'])-1 for x in reqs),
                requests_sha256=sha(BASE/'requests.jsonl'),documents_sha256=sha(BASE/'documents.jsonl'))
    (BASE/'tasks.json').write_text(json.dumps(meta,indent=2)+'\n')
    (REPORT/'tasks.json').write_text(json.dumps(meta,indent=2)+'\n')
    data=Dataset.from_file(TRAIN_ARROW)
    ids=tok('\n\n'.join(data['text']),return_tensors='pt').input_ids
    random.seed(2);starts=[random.randint(0,ids.numel()-2049) for _ in range(128)]
    tokens=torch.cat([ids[:,s:s+2048] for s in starts]);torch.save(tokens,BASE/'train.pt')
    train=dict(source_arrow=TRAIN_ARROW,split='train',nsamples=128,seqlen=2048,seed=2,sample_starts=starts,
        tokens_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest())
    (BASE/'train.json').write_text(json.dumps(train,indent=2)+'\n')
    (REPORT/'train.json').write_text(json.dumps(train,indent=2)+'\n')
    print('Prepared',len(docs),'questions,',len(reqs),'continuations; training',tokens.shape,flush=True)


if __name__=='__main__':main()
