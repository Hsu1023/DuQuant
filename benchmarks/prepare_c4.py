"""FlatQuant's documented C4-new evaluation selection, frozen per tokenizer."""
from prepare_lambada import *
import gzip,itertools

def main():
    source=Path('/mnt/yichen/wyc/QDLM/data/dataset/dataset/allenai/c4/en/c4-validation.00000-of-00008.json.gz')
    with gzip.open(source,'rt') as f:rows=[json.loads(s)['text'] for s in itertools.islice(f,1100)]
    assert len(rows)==1100
    for family,model in MODELS.items():
        tok=AutoTokenizer.from_pretrained(model,use_fast=family=='qwen3',local_files_only=True)
        tokens=tok(' '.join(rows),return_tensors='pt').input_ids[:,:256*2048]
        out=BASE/family;torch.save(tokens,out/'c4_tokens.pt')
        meta=dict(family=family,model=str(model),dataset='allenai/c4',split='validation',source=str(source),source_sha256=sha(source),
            selection='First 1100 documents of English validation shard 00000-of-00008, joined with single spaces; first 256*2048 tokenizer tokens, matching FlatQuant get_c4_new eval_mode',
            tokenizer_use_fast=bool(tok.is_fast),add_special_tokens=True,seqlen=2048,token_count=tokens.numel(),token_ids_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
            blocks=tokens.numel()//2048,scored_tokens=tokens.numel()//2048*2047,unused_tail_tokens=tokens.numel()%2048,
            scoring='All complete selected blocks, shifted FP32 CE, no overlap, reset cache per block; no fitting on validation')
        (out/'c4_protocol.json').write_text(json.dumps(meta,indent=2)+'\n');(REPORT/f'{family}_c4_protocol.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)

if __name__=='__main__':main()
