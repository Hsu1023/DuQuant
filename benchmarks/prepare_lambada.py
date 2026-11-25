"""Freeze English LAMBADA OpenAI and both families' WikiText test tokens."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'benchmarks/cache/qwen3_deps'))
import json,hashlib
import torch
from transformers import AutoTokenizer
from datasets import Dataset
from eval_ppl import DEFAULT_ARROW

BASE=ROOT/'benchmarks/cache/lambada_compare'
REPORT=ROOT/'benchmarks/lambada_compare'
MODELS={'llama2':Path('/mnt/public_models/llama2-7b-hf'),'qwen3':Path('/mnt/public_models/qwen3-1.7b-base')}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    source=ROOT/'benchmarks/cache/lambada_openai.jsonl'
    rows=[json.loads(x) for x in source.read_text().splitlines()]
    assert len(rows)==5153,len(rows)
    test=Dataset.from_file(DEFAULT_ARROW)
    for family,model in MODELS.items():
        out=BASE/family;out.mkdir(parents=True,exist_ok=True)
        tok=AutoTokenizer.from_pretrained(model,use_fast=family=='qwen3',local_files_only=True)
        requests=[]
        for i,row in enumerate(rows):
            text=row['text'];ctx,word=text.rsplit(' ',1)
            context=tok.encode(ctx,add_special_tokens=False)
            together=tok.encode(text,add_special_tokens=False)
            assert together[:len(context)]==context,(family,i,'Target boundary does not preserve context prefix')
            target=together[len(context):]
            assert context and target and len(together)-1<=4096
            requests.append(dict(index=i,context=context,continuation=target,target_word=word))
        p=out/'requests.jsonl';p.write_text(''.join(json.dumps(x)+'\n' for x in requests))
        tokens=tok('\n\n'.join(test['text']),return_tensors='pt').input_ids
        torch.save(tokens,out/'wikitext_tokens.pt')
        protocol=dict(family=family,model=str(model),lambada=dict(dataset='EleutherAI/lambada_openai',subset='English/default',split='test',count=5153,
            source=str(source),source_sha256=sha(source),requests_sha256=sha(p),
            tokenization='No special tokens; encode full text and context; take suffix after exact context-token prefix. Prevents artificial isolated SentencePiece leading-space token.',
            accuracy='All target subword tokens are top-1 under teacher forcing; equivalent to exact greedy target-word completion.',
            ppl='exp(-sum(word loglikelihood)/5153), word-level LAMBADA harness PPL; also retain target-token PPL.',
            max_input_tokens=max(len(r['context'])+len(r['continuation'])-1 for r in requests),target_tokens=sum(len(r['continuation']) for r in requests)),
            wikitext=dict(dataset='Salesforce/wikitext',config='wikitext-2-raw-v1',split='test',source_arrow=DEFAULT_ARROW,source_arrow_sha256=sha(DEFAULT_ARROW),
                join='two newlines',add_special_tokens=True,seqlen=2048,token_count=tokens.numel(),token_ids_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
                blocks=tokens.numel()//2048,scored_tokens=tokens.numel()//2048*2047,unused_tail_tokens=tokens.numel()%2048,
                scoring='Every full disjoint 2048 block; shifted FP32 CE; no overlap; logically reset caches each block; no fitting'))
        (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        (REPORT/f'{family}_protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        print(json.dumps(protocol),flush=True)

if __name__=='__main__':main()
