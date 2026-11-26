# Full-model quality: FP16, fake W4A4 and real INT4

This page keeps the historical results of the first, uncalibrated prototype. Training-set calibration and numeric fixes have since been completed, see [CALIBRATED.md](CALIBRATED.md): real INT4 PPL is 6.1638, and top-1/NLL on the full test set match the independent integer reference.

**Full WikiText-2 PPL is close, but the outputs are not equivalent, and this prototype has a clear quality loss relative to FP16.** This evaluation is not an end-to-end numeric-consistency acceptance, and it does not represent the accuracy of the original calibrated DuQuant.

## Scope and protocol

- Model: local `/mnt/public_models/llama2-7b-hf`, all 32 layers; RTX 4090, physical GPU 4.
- Data: the `wikitext-2-raw-v1` test split of [Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext), local Arrow cache, 4,358 lines. Source revision, files and token SHA256 are stored in the JSON.
- The full text is joined with two newlines and tokenized with the model's slow LlamaTokenizer with default special tokens: 341,469 tokens.
- Sequence length 2048, all 166 complete non-overlapping blocks, dropping the trailing 1,501 tokens. Each block starts from an empty valid KV and predicts positions 1–2047: **339,802 scored tokens** in total.
- FP32 next-token cross entropy; the mean NLL is taken over the actual number of scored tokens, then exponentiated. All three modes use the same Flash SDPA. The full model runs on real text input, including the FP16 embedding, final norm and LM head.
- fake is the PyTorch fake-quant reference of this performance prototype: dequantize to FP16 first, then run a floating-point matmul. CUDA applies the affine scale/zero-point correction after packed INT4 dot products. Their rounding order differs.
- Both quantized modes use the same uncalibrated seed=42 random dual rotations and permutations; W4A4, weight clip=0.8, activation clip=0.9, Q/K clip=0.9, V clip=1.0. The original calibration, SmoothQuant and LWC were not run, and no parameters were tuned on test data.
- **This is not a comparison with the calibrated model of the original `main.py`.** It answers how real INT4 differs in quality from the fake path with identical parameters, for the same prototype that was timed earlier.

## Full results

| mode | PPL ↓ | next-token top-1 accuracy ↑ |
|---|---:|---:|
| FP16 | 5.472100 | 61.0132% |
| fake W4A4, same parameters | 8.345393 | 54.5362% |
| real INT4 CUDA W4A4 | 8.359285 | 54.5506% |

Real INT4 raises PPL by **0.013893 (0.1665%)** over fake, and by **52.76%** over FP16; fake itself is **52.51%** above FP16. So most of the average PPL degradation is already present in the fake path, and the earlier 2.54–2.60× speedup cannot be described as lossless.

Accuracy here is the rate of correctly predicting the next token given the true prefix; it is **not QA accuracy**, nor the agreement rate between the two modes.

## Same parameters, still clearly different outputs

The following two SHA256 hashes are identical for fake and real INT4:

- Rotations and permutations of all layers: `bb9da39428706312d09237dfd58d4d8f30642c9b016f9f266bf099dff3d505d7`
- FP16 weights recovered from each quantized representation: `be7c0f09d01c9ac173f2bd709ae9036176f7b57404a5e6220efbdcd8b82885ab`

Over all 339,802 scored positions, real INT4 and fake **agree on top-1 only 83.3647% of the time**. In a logits diagnostic over 64 evenly spaced positions in each of the first four blocks (256 positions):

- Relative L2 error of the logits: 34.8772%; not bit-identical.
- Mean `KL(fake || CUDA)`: 0.125232 nats.

These compare the same true input prefixes under teacher forcing; free generation may diverge further. Close full-set PPL implies neither identical per-sample outputs nor identical quality on other tasks.

A 10,000-sample bootstrap over the 166 paired blocks gives a descriptive 95% interval of `[0.99705, 1.00655]` for the CUDA/fake PPL ratio. This is not a formal equivalence test.

## Additional integer-arithmetic diagnostic

To check whether the difference is explained only by the element-wise FP16 dequantization rounding of fake, the **first 4 blocks, 8,188 scored tokens** were also evaluated. The independent reference stores the integer codes as FP32, computes exact integer dot products and then applies the scales; norm, activations and Hadamard use PyTorch. This reference passed the same parameter-hash check.

| mode, first 4 blocks | PPL | top-1 agreement with the integer reference |
|---|---:|---:|
| fake W4A4 | 9.195651 | 82.4621% |
| CUDA W4A4 | 9.756077 | 83.5125% |
| independent integer-arithmetic reference | 9.793481 | 100% |

CUDA and the integer reference have close PPL on these blocks, but the outputs still differ. Existing single-operator tests support that quantization boundaries and rounding order amplify numeric differences; **this diagnostic does not prove that all of the deviation comes from fake's dequantization rounding, and no per-layer attribution of the errors of each fused operator was done**. Numeric consistency still had to be tightened, and real calibration parameters imported and re-evaluated. The conclusions are limited to the average metrics on this data.

## Reproduction and files

A new all-position logits mode `last_token_only=False` was added; the default generation interface still computes only the last-position logits. Tests cover the correspondence between causal prefixes and all-position outputs, the next-token shift, FP32 CE and the exact scored-token count; together with the existing CUDA tests, 12 tests pass. No CUDA kernel was changed.

```bash
export CUDA_VISIBLE_DEVICES=4
export CUDA_HOME=/usr/local/cuda-12.4
export CUTLASS_PATH=/mnt/yichen/wyc/QDLM/QuaRot/third-party/cutlass
export TRITON_CACHE_DIR=/tmp/duquant_triton
export DUQUANT_PYTHON=/home/yichen/miniconda3/envs/hb/bin/python

"$DUQUANT_PYTHON" benchmarks/eval_ppl.py --mode fp16 --output benchmarks/ppl_fp16.json
"$DUQUANT_PYTHON" benchmarks/eval_ppl.py --mode fake --output benchmarks/ppl_fake.json
"$DUQUANT_PYTHON" benchmarks/eval_ppl.py --mode cuda --output benchmarks/ppl_cuda.json
"$DUQUANT_PYTHON" benchmarks/eval_ppl.py --mode integer_reference --max-blocks 4 \
  --output benchmarks/ppl_integer_reference_4blocks.json
"$DUQUANT_PYTHON" benchmarks/summarize_ppl.py
```

Run from the repository root, each mode in its own process. Evaluation and load times are not a new official speed benchmark; see [MODEL.md](MODEL.md) for the speed and memory protocol.

- `benchmarks/ppl_{fp16,fake,cuda}.json`: per-block NLL over the full set, scored counts, PPL, data and parameter fingerprints.
- `benchmarks/ppl_integer_reference_4blocks.json`: the additional integer reference.
- `benchmarks/ppl_summary.json`: metrics and diagnostics after the consistency checks.
- `benchmarks/cache/ppl_*.npz`: local per-position predictions, targets and sampled logits for recomputing the diagnostics; in the git-ignored cache directory.
