# FlatQuant extension of the Qwen3-1.7B-Base comparison

This experiment appends FlatQuant to the completed FP16/QuaRot/SpinQuant/DuQuant
experiment in the parent directory. The previous scores and shared inference
and scoring sources are preserved. Evaluation verifies their hashes against the
completed DuQuant run before loading the new model.

## Complete six-task results

Accuracy in percent; all 19965 questions / 67106 continuations completed:

| Method | PIQA | ARC-E | ARC-C | BoolQ | HellaSwag | WinoGrande | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 75.90 | 73.78 | 45.05 | 79.08 | 66.52 | 64.17 | 67.42 |
| QuaRot | 68.61 | 64.39 | 37.80 | 65.96 | 55.86 | 53.83 | 57.74 |
| SpinQuant | 68.61 | 67.63 | 38.99 | 66.88 | 55.40 | 56.35 | 58.98 |
| DuQuant | 70.73 | 72.18 | 41.72 | 72.51 | 58.86 | 56.35 | 62.06 |
| FlatQuant | 71.55 | 69.32 | 40.61 | 70.34 | 63.05 | 60.06 | 62.49 |

FlatQuant's mean is 0.43 percentage points above DuQuant, 4.75 above QuaRot,
3.51 above SpinQuant and 4.93 below FP16. Each of FlatQuant and DuQuant scores
higher on three tasks. FlatQuant scores 4.18 points higher on HellaSwag and 3.71
higher on WinoGrande, while DuQuant scores 2.86 higher on ARC-E and 2.17 higher
on BoolQ. These are local adaptations under one shared deployment setting;
transformation-optimization budgets differ and the small mean difference does
not establish general superiority across models/configurations.

Full true-INT4 scoring took 171.0 seconds excluding model loading. This wall-time
record includes compilation and scheduling and is not a controlled latency
comparison. Summary checks passed for protocol/source hashes, shared train data,
clipping/GPTQ settings, completeness and score integrity. Per-choice and
per-question outputs are retained in `results/` and the five-row comparison is
in `summary.json` and `table.md`.

## Training and shared settings

No official Qwen3-1.7B transformation checkpoint was found in the documented
[FlatQuant model zoo](https://huggingface.co/ruikangliu/FlatQuant). This row uses
locally trained transforms with official SVD transformation and quantizer code
from repository commit `9d88ffcb7d2c6bda59fb5c44dad36adc101aadb1` and a local
Qwen3 forward adapter for head-wise Q/K RMSNorm and GQA.

Training follows layerwise reconstruction: original frozen FP16 prefixes supply
inputs and targets; each layer is optimized for 15 epochs on the same 128 x
2048 WikiText-2 train token sequences used by the other methods. Batch size is
4. AdamW uses lr .005 for transforms/diagonals and .05 for learned clipping,
with cosine decay to 5e-6. The loss is layer-output MSE, normalized by its
detached value before backward, matching the official training loop. Q remains
16-bit, W/A/K/V use 4-bit fake quantization **during transformation learning**.
The six-task main evaluation uses true integer kernels.

Learned components are SVD-parameterized non-orthogonal Kronecker transforms and
per-channel diagonals for QKV, gate/up and down; an across-head O transform; a
shared per-head V transform (compatible with GQA); and matched inverse-transpose
Q / forward K transforms after RoPE. Original Q/K norm stays before RoPE.
No extra Q/K Hadamard is substituted for these transforms.

All three disjoint layer-training shards compute their own identical original
FP16 prefixes; they do not depend on another layer's trained outputs. A
per-layer initialization seed equal to its zero-based index makes training
independent of the shard schedule. Files and hashes are merged only after all
28 layers finish. A short initial single-layer training diagnostic is retained
in `training.log`; completed results use the three `training_shard*.log` runs.

After learning, the trained clipping parameters are replaced by the common
search, as in the earlier LLaMA-2 FlatQuant comparison. W is symmetric
per-output-channel, A asymmetric per-token, and K/V asymmetric per-token/head
(group 128); Q16. W clipping grid .20–1.00 by .01, A/K/V .40–1.00 by .025,
reconstruction-MSE objective; GPTQ block128, damping1%, no act-order, fixed
searched scales and true-sequential A4 Hessians from each model's own quantized
prefix. The transformation-learning quantizer is the official FlatQuant
quantizer; the deployed search and integer quantizer are shared with the other
rows. Total transformation-optimization budgets differ between methods.

## Evaluation and validation

Same base model, FP16 conversion, full six zero-shot datasets, Qwen tokenizer,
no chat template/BOS/EOS, batch schedule and scorer as the parent experiment.
19965 questions and 67106 continuations; ARC-C/HellaSwag use normalized accuracy,
other tasks raw accuracy. Both metrics and all scores/predictions are retained.
K/V is quantized during prefill and packed into the common KV4 cache.

`tests/test_qwen3_flat.py` checks a learned layer's floating-point equivalence,
complete integer decoder output against the independent integer reference, and
GQA packed-cache content for prefill and contiguous single-token appends.
Calibration additionally checks all layers' inverse matrices and floating-point
transformation equivalence before introducing quantization. Full-model 32-choice
integer reference diagnostics are recorded separately.

All three added tests passed. All 28 calibration layers passed the floating-point
equivalence and inverse checks: maximum relative L2 0.00260306 and maximum inverse
identity error 1.93471e-6. The 32-continuation full-model real-INT4 / independent
integer-reference diagnostic is exactly equal (maximum log-likelihood difference
zero), recorded in `integer_reference_check.json`.

The Qwen3 non-orthogonal transform path uses unfused FP16 small matmuls followed
by the common INT4 GEMM. It is not a performance-optimized FlatQuant deployment;
evaluation elapsed times are not controlled inference speed benchmarks, and
prefill QA scoring does not establish cached-decode bitwise equivalence.

```bash
# Existing hb Python, CUDA, CUTLASS and isolated Qwen3 dependency setup.
python benchmarks/train_flat_qwen3.py --shard 0 --shards 3  # available GPU 0
python benchmarks/train_flat_qwen3.py --shard 1 --shards 3  # available GPU 4
python benchmarks/train_flat_qwen3.py --shard 2 --shards 3  # available GPU 7
python benchmarks/finish_flat_qwen3.py  # merge, shared clipping/GPTQ, full QA
python -m unittest discover -s tests -p test_qwen3_flat.py -v
python benchmarks/eval_flat_qwen3.py --mode integer_reference --limit-requests 32 --output <fresh-diagnostic-directory>
python benchmarks/summarize_flat_qwen3.py
```

Outputs are under this directory. Large training states and exported model
weights stay under ignored `benchmarks/cache/qwen3_compare/` directories.
