# DuQuant with FlatQuant-inspired layer reconstruction

Experiment completed on 2026-11-25. Existing baseline checkpoints and results are preserved.

## What changed

DuQuant rotations and permutations stay fixed. Learn positive per-channel folding scales and W/A/K/V clipping by layer-output reconstruction against frozen FP16 targets. Additional scales fold offline into existing norm/linear weights. This borrows the reconstruction and learned-clipping ideas; it does not replace DuQuant rotations with FlatQuant learnable dense transforms.

Both models use the same 128 x 2048 WikiText-2 training-sequence protocol, 15 epochs per layer, batch 4, AdamW (scale learning rate .005, clipping .05), cosine decay, gradient norm clipping 10, and the final epoch. No test-set checkpoint or hyperparameter selection. Both RTN and fixed-scale true-sequential GPTQ exports are retained.

W4 symmetric per output channel, asymmetric per-token A4, Q16 and asymmetric packed KV4 per head of 128. Evaluation uses the existing real INT4 kernels and unchanged shared scorers. New scales add no online transforms; the new LLaMA-2 RTN generation latency and peak memory are measured below. Qwen3 and the GPTQ exports have not been rebenchmarked.

## Results

Original six: PIQA, ARC-E, ARC-C, BoolQ, HellaSwag, WinoGrande (19,965 questions / 67,106 candidates). ARC-C and HellaSwag use normalized accuracy; others use accuracy. LAMBADA six replaces BoolQ with LAMBADA. LAMBADA evaluates all 5,153 examples. WikiText uses disjoint length-2048 full test blocks (166 LLaMA, 146 Qwen); C4 uses the same 256-block validation slice, not the entire corpus.

### Qwen3-1.7B-Base

| Method | Original six mean (%) | LAMBADA six mean (%) | LAMBADA acc (%) | WikiText PPL | C4 PPL |
|---|---:|---:|---:|---:|---:|
| fp16 | 67.42 | 64.86 | 63.77 | 9.388 | 15.408 |
| duquant | 62.06 | 58.56 | 51.52 | 12.361 | 20.995 |
| flatquant | 62.49 | 60.32 | 57.31 | 11.547 | 19.527 |
| reconstruct_rtn | 64.45 | 61.54 | 57.66 | 10.700 | 17.626 |
| reconstruct_gptq | 63.24 | 61.12 | 59.09 | 10.820 | 17.824 |

| Method | PIQA | ARC-E | ARC-C | BoolQ | HellaSwag | WinoGrande |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 75.90 | 73.78 | 45.05 | 79.08 | 66.52 | 64.17 |
| duquant | 70.73 | 72.18 | 41.72 | 72.51 | 58.86 | 56.35 |
| flatquant | 71.55 | 69.32 | 40.61 | 70.34 | 63.05 | 60.06 |
| reconstruct_rtn | 72.36 | 71.89 | 43.00 | 75.11 | 62.09 | 62.27 |
| reconstruct_gptq | 72.14 | 70.71 | 42.15 | 71.80 | 61.71 | 60.93 |

Paired question-bootstrap comparisons to this FlatQuant checkpoint (4,000 replicates, seed 0; does not measure training-seed uncertainty):

- RTN original-six delta: +1.97 percentage points, 95% interval [+1.16, +2.76].
- GPTQ original-six delta: +0.75 percentage points, 95% interval [-0.02, +1.55].

Recorded layer-training time summed across three disjoint GPU shards: 1705.1 GPU-seconds. Excludes prefix construction, exports, evaluation, and failed numerical diagnostics. Training manifests and raw logs are retained.

Artifacts: `benchmarks/qwen3_compare/duquant_reconstruct/summary.json`, `benchmarks/qwen3_compare/duquant_reconstruct/integer_check.json`, `benchmarks/qwen3_compare/duquant_reconstruct/paired_bootstrap.json` and both variants' complete QA/quality outputs.

### LLaMA-2-7B

| Method | Original six mean (%) | LAMBADA six mean (%) | LAMBADA acc (%) | WikiText PPL | C4 PPL |
|---|---:|---:|---:|---:|---:|
| fp16 | 66.58 | 67.05 | 73.92 | 5.472 | 7.263 |
| duquant | 64.95 | 65.04 | 72.56 | 5.891 | 8.011 |
| flatquant | 65.05 | 65.72 | 72.91 | 5.774 | 7.799 |
| reconstruct_rtn | 65.57 | 65.59 | 71.61 | 5.818 | 7.831 |
| reconstruct_gptq | 65.25 | 65.40 | 71.96 | 5.791 | 7.866 |

| Method | PIQA | ARC-E | ARC-C | BoolQ | HellaSwag | WinoGrande |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 78.40 | 69.32 | 40.61 | 71.10 | 72.95 | 67.09 |
| duquant | 76.50 | 68.48 | 37.71 | 72.02 | 70.33 | 64.64 |
| flatquant | 77.37 | 69.02 | 38.31 | 68.90 | 70.78 | 65.90 |
| reconstruct_rtn | 77.26 | 68.64 | 38.99 | 71.53 | 70.80 | 66.22 |
| reconstruct_gptq | 76.71 | 67.97 | 38.74 | 71.10 | 70.70 | 66.30 |

Paired question-bootstrap comparisons to this FlatQuant checkpoint (4,000 replicates, seed 0; does not measure training-seed uncertainty):

- RTN original-six delta: +0.53 percentage points, 95% interval [-0.07, +1.18].
- GPTQ original-six delta: +0.21 percentage points, 95% interval [-0.43, +0.84].

Recorded layer-training time summed across three disjoint GPU shards: 5532.4 GPU-seconds. Excludes prefix construction, exports, evaluation, and failed numerical diagnostics. Training manifests and raw logs are retained.

Artifacts: `benchmarks/llama_reconstruct/summary.json`, `benchmarks/llama_reconstruct/integer_check.json`, `benchmarks/llama_reconstruct/paired_bootstrap.json` and both variants' complete QA/quality outputs.

## Interpretation and limits

Qwen3: RTN improves the original-six mean by 2.39 points over the parent DuQuant and 1.97 over this FlatQuant comparison, with lower WikiText/C4 PPL. Both new exports also improve LAMBADA accuracy over this FlatQuant checkpoint.

LLaMA-2: the original-six mean improves modestly, but the paired interval against FlatQuant includes zero. LAMBADA and PPL still trail FlatQuant. This experiment therefore supports a useful DuQuant extension, especially on Qwen3, rather than a general superiority claim.

The existing FlatQuant comparison uses learned transforms plus the common clipping-search/GPTQ pipeline. LLaMA uses author-released transforms; Qwen uses the previously trained adaptation. Neither is an untouched author-default quantizer. A claim against native FlatQuant would require preserving its native learned clipping and matching calibration budget. The DuQuant extension adds optimization budget and changes calibration relative to original DuQuant.

Both variants on both models match the independent integer reference exactly on 32 full-model likelihood requests. This checks the prefill path; the documented packed-KV cached-decode rounding distinction remains. No speed or memory result is inferred from unchanged operator count.

## New LLaMA-2 RTN performance

| LLaMA-2-7B | End-to-end time ↓ | Speedup ↑ | Model storage ↓ | Peak allocated GPU memory ↓ | Peak memory reduction ↑ |
|---|---:|---:|---:|---:|---:|
| FP16 | 552.13 ms | 1.00× | 12.55 GiB | 13.23 GiB | — |
| DuQuant RTN W4A4KV4 | 207.62 ms | **2.66×** | 3.54 GiB | 3.90 GiB | **70.5%** |

Measured on one NVIDIA GeForce RTX 4090 (Ada), batch size 1, 1024-token prompt and 32 generated tokens, including all 32 decoder layers and the LM head. Values are medians of 10 repetitions with CUDA Graphs for both FP16 and INT4. Model loading, compilation and graph capture are excluded; output transfer is included. Peak memory includes capture/warmup allocation, including the graph pool. This measures the new RTN checkpoint used below. Model tensor storage is reduced by **71.8%**.

[Measurements and memory accounting](llama_reconstruct/performance/summary.json) · [Generation benchmark](bench_llama_reconstruct.py). The standalone backend is separate from the original `main.py` workflow.
