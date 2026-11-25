# FlatQuant under the existing LLaMA-2 7B comparison protocol

This extends `benchmarks/qa_compare` with one FlatQuant W4A4KV4 row. The original
results, backend sources, task data and scoring functions remain unchanged.
`eval_flatquant.py` imports the original scorer and verifies every shared source
hash against the completed DuQuant run before evaluation.

## Complete results

Accuracy in percent; all six full splits completed:

| Method | PIQA | ARC-E | ARC-C | BoolQ | HellaSwag | WinoGrande | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 / KV16 | 78.40 | 69.32 | 40.61 | 71.10 | 72.95 | 67.09 | 66.58 |
| QuaRot | 74.97 | 67.42 | 38.74 | 68.44 | 68.35 | 63.61 | 63.59 |
| SpinQuant | 74.48 | 65.57 | 37.80 | 69.57 | 66.95 | 62.35 | 62.79 |
| DuQuant | 76.50 | 68.48 | 37.71 | 72.02 | 70.33 | 64.64 | 64.95 |
| FlatQuant | 77.37 | 69.02 | 38.31 | 68.90 | 70.78 | 65.90 | 65.05 |

FlatQuant's mean is 0.10 percentage points above DuQuant, 1.46 above QuaRot,
2.26 above SpinQuant and 1.53 below FP16. The small mean difference from DuQuant
does not establish a broadly superior method. FlatQuant scores higher on five
tasks here, while DuQuant scores 3.12 points higher on BoolQ.

FlatQuant scoring took 256.4 seconds on GPU 0, excluding model loading. DuQuant's
previous run recorded 263.7 seconds. This is an evaluation wall-time record,
not a controlled same-session speed benchmark. All 19965 question predictions
and 67106 candidate scores are retained, and summary validation passed.

## Matched settings

- Same checkpoint: `/mnt/public_models/llama2-7b-hf`.
- True INT4 linear arithmetic: symmetric per-output-channel W4, asymmetric
  per-token A4, Q16, asymmetric per-token/head K4/V4 with group size 128.
- Same W clipping grid (0.20–1.00 by 0.01), A/K/V grid (0.40–1.00 by 0.025),
  reconstruction MSE objective and 8192 sampled activation rows per layer.
- Same 128 x 2048 WikiText-2 train token sequences for search/GPTQ, verified by
  hash. GPTQ uses block size 128, 1% damping, no activation ordering, fixed
  searched scales and true-sequential A4 inputs from its own quantized prefix.
- Same six full zero-shot task splits, prompts, tokenization, batch schedule and
  continuation-likelihood scoring: 19965 questions, 67106 candidate answers.
- ARC-C and HellaSwag use character-normalized accuracy; other primary metrics
  use raw accuracy. Both metrics are retained; mean is unweighted over tasks.

## Preserved FlatQuant method

Official transformation checkpoint:
[ruikangliu/FlatQuant, llama-2-7b/w4a4/flat_matrices.pth](https://huggingface.co/ruikangliu/FlatQuant/blob/main/llama-2-7b/w4a4/flat_matrices.pth).
Its SHA256, checked against the published Hugging Face LFS object, is
`7133f8a7bab8179ffa9cc0404c66d22e656f5b67a5abd31066d16bdd25d336ef`.
Official source commit: `9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1`.

The implementation preserves the learned non-orthogonal Kronecker transforms,
per-channel diagonals, across-head O transform, V transform and post-RoPE
inverse-transpose Q / forward K transform. It does not substitute orthogonal
rotations or an extra Q/K Hadamard. Inverse-transpose weight folding uses FP64,
then casts to FP16; diagonals fold into RMSNorm/up-projection as in the official
reparameterization. Online small transforms use FP16 matmuls, followed by real
integer GEMM for each quantized linear projection.

The checkpoint's learned W/A/K/V clipping is replaced by the common search,
followed by the common GPTQ. Transformation learning itself retains the official
checkpoint's original optimization budget and settings. Consequently this is a
controlled quantizer/backend comparison, not a reproduction of FlatQuant's
paper-default accuracy or an equal-total-training-budget comparison.

K/V are quantized during **prefill as well as decode**. A dedicated Triton
packing step preserves FlatQuant's learned Q/K basis and writes the same packed
asymmetric KV4 format. Prefill Flash SDPA consumes its quantized/dequantized K/V;
it does not bypass KV quantization. Packed decode reuses the existing kernel.

## Validation and reproduction

`tests/test_flatquant.py` checks full floating-point layer equivalence, integer
linear outputs, full integer decoder output and packed KV values against
independent references. Calibration checks floating-point layer equivalence and
inverse-transpose matrices for every layer before quantization.

All four tests passed. Across 32 calibrated layers, maximum floating-point
relative L2 error was 0.00185065 and maximum inverse-transpose identity error
was 9.26428e-7. A separate full-model check of 32 continuations produced exactly
equal log-likelihoods with the real INT4 path and independent integer reference
(maximum difference 0); see `integer_reference_check.json` and the two diagnostic
run directories. Calibration took approximately 297.5 seconds across layers.

Use the existing `hb` environment, CUDA/CUTLASS and cached dependencies described
in `benchmarks/qa_compare/README.md`:

```bash
python -m unittest discover -s tests -p test_flatquant.py -v
python benchmarks/calibrate_flatquant.py
python benchmarks/eval_flatquant.py
python benchmarks/summarize_flatquant.py
```

Calibration refuses to overwrite an existing manifest. Evaluation resumes only
with explicit `--resume` and matching code/configuration hashes. The small
full-model integer-oracle check is reproducible with `--limit-requests 32`,
separate output directories and modes `cuda` / `integer_reference`; it does not
run full fake-quant evaluation.

`results/` retains all choice scores and question predictions; `summary.json`
and `table.md` append the validated result to the previous comparison. Large
model exports and downloaded matrices remain in ignored cache directories.
Evaluation wall time includes compilation/padding/Python scheduling, excludes
model loading and is not a controlled latency benchmark. Online transform
fusion is not optimized, and the existing KV4 cached-decode numerical caveat
remains applicable.
