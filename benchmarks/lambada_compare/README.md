# Added LAMBADA and PPL comparison: LLaMA-2 7B and Qwen3-1.7B-Base

FlatQuant's default six-task list uses `lambada_openai` with PIQA, ARC-E,
ARC-C, HellaSwag and WinoGrande. Our preceding six-task experiment used BoolQ
instead of LAMBADA. This extension evaluates the missing task and adds PPL,
without retraining or changing any FP16/INT4 checkpoint.

Both families evaluate FP16, QuaRot, SpinQuant, DuQuant and FlatQuant. Quantized
rows retain real W4A4KV4, Q16, symmetric per-channel W, asymmetric per-token A
and asymmetric per-head K/V (group 128), with the existing common searched
clipping/GPTQ exports. The prior QA run and calibration hashes are verified
before loading, including the source versions used for the original results.

## Completed results

All ten LAMBADA/WikiText runs and all ten C4 runs completed and passed validation.
Accuracy is higher-is-better; all PPL columns are lower-is-better.

### llama2

| Method | LAMBADA acc (%) | LAMBADA word PPL | WikiText-2 PPL | C4 PPL | LAMBADA six-task mean (%) |
|---|---:|---:|---:|---:|---:|
| fp16 | 73.92 | 3.395 | 5.472 | 7.263 | 67.05 |
| quarot | 70.83 | 4.084 | 6.430 | 8.875 | 63.99 |
| spinquant | 69.92 | 4.509 | 6.666 | 9.118 | 62.85 |
| duquant | 72.56 | 3.702 | 5.891 | 8.011 | 65.04 |
| flatquant | 72.91 | 3.576 | 5.774 | 7.799 | 65.72 |

### qwen3

| Method | LAMBADA acc (%) | LAMBADA word PPL | WikiText-2 PPL | C4 PPL | LAMBADA six-task mean (%) |
|---|---:|---:|---:|---:|---:|
| fp16 | 63.77 | 5.609 | 9.388 | 15.408 | 64.86 |
| quarot | 48.11 | 13.037 | 15.200 | 26.694 | 54.77 |
| spinquant | 48.55 | 13.349 | 14.180 | 24.198 | 55.92 |
| duquant | 51.52 | 10.820 | 12.361 | 20.995 | 58.56 |
| flatquant | 57.31 | 8.068 | 11.547 | 19.527 | 60.32 |

The six-task mean replaces BoolQ with LAMBADA and keeps the five prior QA scores.
FlatQuant minus DuQuant is approximately +0.68 points on LLaMA-2 and +1.76 on
Qwen3 under this task composition. This is a local controlled-format result;
prior task protocol and transformation-training-budget differences still apply.
FlatQuant has lower WikiText/C4 PPL and higher LAMBADA accuracy than DuQuant for
both tested checkpoints. No evaluation result was used to retune any model.

## Datasets and scores

- **LAMBADA OpenAI English/default test:** all 5153 passages, downloaded from
  OpenAI's original `gpt-2/data/lambada_test.jsonl` release. Source SHA256:
  `4aa8d02cd17c719165fc8a7887fddd641f43fcafa4b1c806ca8abc31fabdb226`.
  Each context excludes the last space-separated word. No BOS/EOS or chat
  template is added. Target tokens are the suffix of the full-text encoding
  after its exact context-token prefix. Every passage passes this boundary
  assertion. This prevents an artificial isolated SentencePiece leading-space
  token when separately encoding a target such as ` word`.
- **LAMBADA accuracy:** every target subword must be the top-1 prediction at its
  causal position. This teacher-forced all-token condition is equivalent to exact
  greedy completion of the target word; it is not multiple-choice ranking.
- **LAMBADA word PPL:** `exp(-sum(target-word log likelihoods)/5153)`, the harness
  word-level definition. Target-token PPL is also saved with its denominator:
  8808 target tokens for LLaMA-2 and 6496 for Qwen3. Word PPL here does not score
  all tokens in the narrative passages.
- **WikiText-2 raw test:** existing local test Arrow, all complete disjoint
  2048-token blocks, joined with two newlines, tokenizer-default special tokens.
  LLaMA-2: 166 blocks, 339802 scored tokens; Qwen3: 146 blocks, 298862 scored
  tokens. Tail tokens outside a complete block are excluded and recorded.
- **C4 validation:** FlatQuant `get_c4_new(eval_mode=True)` selection: English
  validation shard `00000-of-00008`, first 1100 texts joined with single spaces,
  first 256 x 2048 tokenizer tokens. Each family scores 256 blocks / 524032
  shifted tokens. This is the standard selected evaluation slice, not the entire
  C4 corpus. File and tokenizer-token hashes are retained in protocols.
- Both continuous-text PPL evaluations use logically reset caches per block,
  shifted FP32 cross entropy and FP64 loss summation. K/V quantization is active
  during prefill. No calibration or selection uses these evaluation data.

## Means and relation to paper numbers

The new six-task mean uses LAMBADA in place of BoolQ, retaining the five existing
QA task scores. It matches **task composition**, not every official evaluation
setting. The prior five scores preserve their frozen legacy prompts and separate
context/choice tokenization; new LAMBADA uses the full-text/context-prefix
boundary rule described above. Do not equate this mean with a paper-default
reproduction. The original six-task mean and the expanded seven-task mean
(including both BoolQ and LAMBADA) are also saved in `summary.json`.

The two families use different tokenizers, so PPL comparisons should primarily
be made between methods within a family. Existing differences in transformation
training data/budget and local Qwen3 adaptations remain applicable.

## Checks and reproduction

`tests/test_lambada.py` independently generates greedy one-token and multi-token
continuations, introduces a wrong target, and checks both the greedy flags and
conditional log likelihoods against repeated causal-prefix inference. The test
passed. All 5153 token boundaries were checked for each tokenizer. LLaMA-2 FP16
and DuQuant WikiText PPL reproduce the previous results (5.472090 and 5.890531).

Use the existing hb Python/CUDA/CUTLASS setup. Qwen3-compatible Transformers and
Tokenizers are isolated in `benchmarks/cache/qwen3_deps`; no previous inference
sources are edited by this extension.

```bash
python benchmarks/prepare_lambada.py
python benchmarks/prepare_c4.py
python -m unittest discover -s tests -p test_lambada.py -v
python benchmarks/eval_lambada_ppl.py --family llama2 --method duquant
python benchmarks/eval_c4_common.py --family llama2 --method duquant
# Repeat both scripts for qwen3 and fp16/quarot/spinquant/duquant/flatquant.
python benchmarks/summarize_lambada.py
```

Resume requires explicit `--resume` with matching configuration/source hashes.
Each method stores run metadata, full LAMBADA target predictions/log likelihoods,
per-block PPL losses, and hashes. `summary.json` and `table.md` are generated only
when all ten LAMBADA/WikiText runs and all ten C4 runs are complete and validate.
Wall times include compilation/batching/Python overhead and are not controlled
latency benchmarks. Full-prefill scoring does not resolve the separate packed
KV4 cached-decode numerical limitation.
