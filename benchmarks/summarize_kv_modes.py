"""Validate and collect completed KV16/KV4 experiments, retaining caveats."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path('benchmarks/kv_modes')
    rows = []
    for family in ('llama1','llama2','llama3','vicuna','mistral'):
        record = json.loads((root/family/'summary.json').read_text())
        assert record['status']=='complete'
        for path,digest in record['source_sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
        for bits in ('16','4'):
            value=record['results'][bits]
            ref=value['integer_reference']
            assert ref['predictions_equal'] and ref['sampled_logits_equal'] and ref['max_block_nll_delta']==0
            assert value['decode']['status']=='complete'
            assert value['decode']['packed_attention_oracle']['passed']
            assert value['decode']['cached_decode']['scored_tokens']==256
            if bits=='16':
                assert value['decode']['max_relative_l2']==0
                assert value['decode']['cached_decode']['mean_nll_delta']==0
        baseline=json.loads((Path('benchmarks/families')/family/'ppl_fp16.json').read_text())
        record['fp16_ppl']=baseline['ppl']
        record['scored_tokens']=baseline['scored_tokens']
        rows.append(record)
    long_root=root/'mistral/long_context'
    long_runs={}
    for mode,bits in [('fp16',16),('cuda',16),('cuda',4)]:
        value=json.loads((long_root/f'model_{mode}_kv{bits}_p8192_g32.json').read_text())
        assert value['graph_matches_eager'] and value['finite_logits']
        assert len(value['samples'])==10 and value['prompt_tokens']==8192
        long_runs[f'{mode}_kv{bits}']=value
    result=dict(status='complete',families=rows,long_context=long_runs,
        protocol='Same existing no-LWC calibration exports; query FP16 in both explicit modes; full WikiText-2 prefill PPL; batch-1 128/1024 -> 32 generation, 10-run medians; Mistral additionally 8192 -> 32',
        kv4='Packed uint8 codes (2 per byte) and FP16 scale/zero per token/head; K clip .9, V clip 1; RoPE then Q/K Hadamard; no additional V/O rotation; tiled decode without full FP16 cache materialization',
        numerical_caveat='Full prefill PPL matches integer reference exactly. KV4 cached decode matches a shared-input FP32 attention oracle within recorded tolerances, but downstream W4A4 rounding amplifies tiny differences: full-model logits/predictions are NOT bitwise equivalent to dense-reference decode. The separate 256-token cached-decode diagnostic is limited, not a full evaluation.',
        reuse='The GQA optimization changes decode only. Its cached-decode diagnostics and speed/memory were rerun; unchanged prefill PPL, KV16, and MHA results were preserved with reuse/source records.')
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for row in rows:
        print(row['family'], 'FP16/legacy/KV16/KV4 PPL', row['fp16_ppl'],row['prior_q4kv4_fake_cache_ppl'],
              row['results']['16']['ppl'],row['results']['4']['ppl'])


if __name__=='__main__':
    main()
