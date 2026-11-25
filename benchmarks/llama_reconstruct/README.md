# LLaMA-2 7B: frozen DuQuant transforms with learned folding scales/clipping

This is the second checkpoint for the reconstruction experiment in
`benchmarks/qwen3_compare/duquant_reconstruct`. Qwen3 showed gains under the
predeclared metrics, and the same change was verified on LLaMA-2 7B. The previous checkpoints and results are preserved.

## Fixed experiment

Parent: `benchmarks/cache/ptq_llama2_final/sym_gptq`. DuQuant block rotations and
permutations are fixed. The new trainable values are additional positive
per-channel smoothing/folding scales, per-output-channel symmetric W clips,
per-fused-projection scalar A clips and scalar K/V clips. These scales fold into
existing norm/linear weights and introduce no additional online transforms.

Same 128 x 2048 WikiText-2 train token sequences as the parent's manifest,
verified by hash; original frozen FP16 prefixes/outputs as layerwise MSE targets;
15 epochs per layer, batch4, AdamW scale lr .005, clip lr .05, cosine decay to
5e-6, normalized loss and grad norm clip10. Per-layer initialization and sample
schedule are fixed. Three disjoint training shards each compute the original
FP16 prefix independently. The last epoch is exported; evaluation data are not
used for checkpoint or hyperparameter selection.

This shares the numerically stable training quantizer with the Qwen3 experiment:
FP16-rounded forward values, FP32 STE intermediate gradients, and detached
integer zero-point computation. The learned channel scales are FP16 in both
training and export, with the same folding order. A dedicated test validates
finite gradients and bitwise equality of training/exported folded tensors.

## Export and evaluation

Both RTN and fixed-scale true-sequential GPTQ (block128, damping1%, no act-order)
are predeclared and retained. Both use the learned clipping and actual quantized
prefix inputs. Deploy using existing real INT4 kernels, symmetric W4 encoded at
zero8, asymmetric per-token A4, Q16, and packed asymmetric KV4 per head128.

Evaluate the complete original six zero-shot tasks, LAMBADA OpenAI test, full
WikiText-2 test blocks and the same 256-block C4 validation slice. Original six,
LAMBADA six and seven-task means are all retained. Compare to the previous
unchanged parent DuQuant and FlatQuant exports. The latter uses official learned
FlatQuant transforms with common searched clipping/GPTQ; it is not an untouched
native-default FlatQuant reproduction. Do not hide per-task regressions or
infer superiority on other models/settings.

```bash
python -m unittest discover -s tests -p test_llama_reconstruct.py -v
python benchmarks/train_llama_reconstruct.py --shard 0  # available GPU 0
python benchmarks/train_llama_reconstruct.py --shard 1  # available GPU 4
python benchmarks/train_llama_reconstruct.py --shard 2  # available GPU 7
python benchmarks/finish_llama_reconstruct.py
python benchmarks/eval_llama_reconstruct.py --variant rtn --output benchmarks/llama_reconstruct/rtn/results
python benchmarks/eval_llama_reconstruct_quality.py --variant rtn
python benchmarks/eval_llama_reconstruct_quality.py --variant gptq
python benchmarks/check_llama_reconstruct_integer.py
python benchmarks/summarize_llama_reconstruct.py
python benchmarks/paired_reconstruct.py --family llama2
```

Model exports remain in ignored `benchmarks/cache/llama_reconstruct`. The common
backend's prefill/cached-decode numerical distinction remains applicable.
The new RTN export has a controlled generation benchmark in
`performance/summary.json`; Qwen3 and this model's GPTQ export have not been rebenchmarked.

## Completed results

| Method | Original six mean (%) | LAMBADA six mean (%) | LAMBADA acc (%) | WikiText PPL | C4 PPL |
|---|---:|---:|---:|---:|---:|
| fp16 | 66.58 | 67.05 | 73.92 | 5.472 | 7.263 |
| duquant | 64.95 | 65.04 | 72.56 | 5.891 | 8.011 |
| flatquant | 65.05 | 65.72 | 72.91 | 5.774 | 7.799 |
| reconstruct_rtn | 65.57 | 65.59 | 71.61 | 5.818 | 7.831 |
| reconstruct_gptq | 65.25 | 65.40 | 71.96 | 5.791 | 7.866 |

Original six tasks are PIQA, ARC-E, ARC-C, BoolQ, HellaSwag and WinoGrande.
The LAMBADA six replaces BoolQ with LAMBADA. Each new export scored all 67,106
candidate continuations for 19,965 original-six questions, all 5,153 LAMBADA
examples, 166 WikiText blocks and 256 C4 blocks at sequence length 2048.

RTN improves the original-six mean by 0.63 percentage points over the parent and
0.53 over the existing FlatQuant comparison. The latter paired question-bootstrap
95% interval is [-0.07, +1.18] points, which includes zero. The LAMBADA-six mean,
LAMBADA accuracy, WikiText PPL and C4 PPL do not beat this FlatQuant checkpoint.
GPTQ does not improve the original-six mean over RTN. These results do not support
a general claim of beating FlatQuant across both models.

Both exports pass the independent integer-dot/dense-KV reference check on 32
full-model requests with exactly zero score difference. This validates the tested
prefill likelihood path; it does not extend that claim to cached generation.
See `integer_check.json`, `summary.json`, `paired_bootstrap.json`, and the complete
per-task/per-question outputs in `rtn/results` and `gptq/results`.

## New RTN performance

| LLaMA-2-7B | End-to-end time ↓ | Speedup ↑ | Model storage ↓ | Peak allocated GPU memory ↓ | Peak memory reduction ↑ |
|---|---:|---:|---:|---:|---:|
| FP16 | 552.13 ms | 1.00× | 12.55 GiB | 13.23 GiB | — |
| DuQuant RTN W4A4KV4 | 207.62 ms | **2.66×** | 3.54 GiB | 3.90 GiB | **70.5%** |

Measured on one NVIDIA GeForce RTX 4090 (Ada), batch size 1, 1024-token prompt and 32 generated tokens, including all 32 decoder layers and the LM head. Values are medians of 10 repetitions with CUDA Graphs for both FP16 and INT4. Model loading, compilation and graph capture are excluded; output transfer is included. Peak memory includes capture/warmup allocation, including the graph pool. This measures the new RTN checkpoint used below. Model tensor storage is reduced by **71.8%**.

[Measurements and memory accounting](performance/summary.json) · [Generation benchmark](../bench_llama_reconstruct.py). The standalone backend is separate from the original `main.py` workflow.

Raw per-configuration results and logs are retained in `performance/`.
