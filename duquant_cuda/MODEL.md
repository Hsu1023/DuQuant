# Full LLaMA-2-7B: continuous generation speed and memory

This page keeps the speed and memory results of the first, uncalibrated prototype. The re-measurement of the current calibrated model with the final code is in [CALIBRATED.md](CALIBRATED.md): 2.52–2.57× end-to-end speedup, together with full PPL and numeric-consistency checks.

`model.py` wires up the full 32-layer model together with the embedding, final RMSNorm, FP16 LM head and greedy argmax. `benchmarks/bench_model.py` measures real autoregressive generation: every step consumes the previously generated token and writes and extends the valid KV cache. It is neither a repeated single step at a fixed position nor a sum of single-layer times.

These results belong to an **uncalibrated W4A4 performance prototype**. All decoder projections use real packed INT4 weights and the W4A4 GEMM/GEMV; embedding, LM head, attention computation and KV storage are still FP16. The rotations and permutations are fixed random parameters; no DuQuant calibration, SmoothQuant or LWC has been run. The performance results must not be read as acceptance of full-model quality. The later [full WikiText-2 quality evaluation](QUALITY.md) gives FP16 / fake / INT4 PPL of 5.4721 / 8.3454 / 8.3593; INT4 and fake agree on top-1 only 83.36% of the time, so output equivalence was not established.

## Protocol

- GPU: RTX 4090 (this machine reports about 48 GiB of memory), physical GPU 4; batch=1, single GPU, no CPU offload.
- Model: local `/mnt/public_models/llama2-7b-hf`, all real weights.
- Prompt: the same fixed text repeated to 128 or 2048 tokens; identical inputs.
- Output: always 32 generated tokens, no early stop at EOS, so both modes do comparable work.
- The two modes run in separate processes; no second copy of the model is resident during timing. FP16 uses the same merged projections, fused norm/RoPE/SwiGLU and Flash SDPA.
- A prefill graph and a complete decode graph are captured for the given prompt and output lengths; they run in sequence and share a graph memory pool.
- The first generated token comes from prefill, the remaining 31 from continuous decode; the reserved KV capacity is `prompt_length + 31`.
- 5 measurements per configuration, median reported; GPU clocks not locked.
- Total time is wall time from the tokenized input on the GPU until the generated token IDs are copied back to the CPU. It includes the full-model computation, sampling and real cache updates; it excludes tokenization, model loading/quantized export, first compilation, graph capture and the prompt H2D copy.
- The prefill column is GPU event time and includes the prediction of the first token; it is not the user-perceived TTFT of a network service. Decode throughput is computed over the remaining 31 tokens.

The graphs fix the request shape and generation length. This validates real continuous generation for a fixed workload; it is not a serving framework with dynamic batching, dynamic stopping and arbitrary request lengths.

## Memory accounting

All memory figures are in **GiB = 2^30 bytes**, and the following are distinguished:

- **Model tensors**: deduplicated by unique backing storage; includes weights, quantization metadata, rotations/permutations, norms and the shared RoPE tables; excludes KV.
- **KV cache**: the actually preallocated K/V tensors, FP16 on both paths.
- **Peak allocated**: peak of live PyTorch tensors. Counted over warmup, the inference run during capture, and replay; model loading, quantized export and the HF comparison load are excluded. The capture-phase peak is kept because graph replay no longer requests temporaries from the allocator, so resetting before replay would miss the live peak of the temporary workspace.
- **Reserved**: memory actually held by PyTorch during steady inference, including the private graph pool and allocator fragmentation. Capacity planning cannot look only at model tensors or allocated memory.
- **Device usage**: read through CUDA `mem_get_info`; additionally includes the CUDA context, graphs and allocations outside the PyTorch allocator. Measured on the otherwise idle GPU 4.

W4A4 reduces the model tensors from 12.55 GiB to 3.54 GiB, about 71.8%; this is not a theoretical 4× reduction of whole-device memory. KV is not compressed, and the allocator fragmentation from offline construction and the graph/context overheads remain.

## Validation and limitations

- The full FP16 model was compared with Hugging Face on a separate short prompt: last-position logits differ by about 0.084% relative L2, with the same top-1. The reference model is freed before timing and the memory peak is reset.
- The tests check that growing-cache decode matches recomputing the full prefix, real token feedback across two layers, graph replay after changing the prompt, and results after clearing the cache.
- After every full-model capture, the 32-token result is checked to equal eager generation token by token, and the logits are checked to be finite.
- On the two fixed repeated-text prompts used here, FP16 and the prototype also produce the same token IDs. This is a very limited sample and **does not mean PPL or broad task quality are lossless**. The [multi-layer PPL evaluation](QUALITY.md) confirmed a clear quality loss for the prototype; real calibration and further numeric-consistency work were still pending.
- All regression tests and the CUDA memcheck log are in `benchmarks/model_memcheck.log`.

## Files and reproduction

The raw data are `benchmarks/model_{fp16,cuda}_p{128,2048}_g32.json`, which keep every timing sample, all memory figures, the generated token IDs and the text. The summary is `benchmarks/model_summary.json`.

```bash
export CUDA_VISIBLE_DEVICES=4
export CUDA_HOME=/usr/local/cuda-12.4
export CUTLASS_PATH=/mnt/yichen/wyc/QDLM/QuaRot/third-party/cutlass
export TRITON_CACHE_DIR=/tmp/duquant_triton
export DUQUANT_PYTHON=/home/yichen/miniconda3/envs/hb/bin/python

# Each command runs in its own process; start the next only after the previous one exits.
"$DUQUANT_PYTHON" benchmarks/bench_model.py --mode fp16 --prompt 128 \
  --new-tokens 32 --verify-hf --output benchmarks/model_fp16_p128_g32.json
"$DUQUANT_PYTHON" benchmarks/bench_model.py --mode cuda --prompt 128 \
  --new-tokens 32 --output benchmarks/model_cuda_p128_g32.json
"$DUQUANT_PYTHON" benchmarks/bench_model.py --mode fp16 --prompt 2048 \
  --new-tokens 32 --output benchmarks/model_fp16_p2048_g32.json
"$DUQUANT_PYTHON" benchmarks/bench_model.py --mode cuda --prompt 2048 \
  --new-tokens 32 --output benchmarks/model_cuda_p2048_g32.json
"$DUQUANT_PYTHON" benchmarks/summarize_model.py
```

`--verify-hf` temporarily loads an extra FP16 model for validation and needs enough memory for it; this is not counted in the inference memory below.

## Measured summary

| input → output tokens | FP16 total ms | W4A4 total ms | speedup | FP16 prefill ms | W4A4 prefill ms | FP16 decode tok/s | W4A4 decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128 → 32 | 459.63 | 176.75 | 2.60× | 17.85 | 7.32 | 70.2 | 183.0 |
| 2048 → 32 | 654.34 | 257.82 | 2.54× | 178.89 | 54.41 | 65.2 | 152.5 |

| input → output tokens | mode | model tensors GiB | KV GiB | peak allocated GiB | reserved GiB | device usage GiB |
|---|---|---:|---:|---:|---:|---:|
| 128 → 32 | fp16 | 12.55 | 0.08 | 12.67 | 12.68 | 13.26 |
| 128 → 32 | w4a4 | 3.54 | 0.08 | 3.65 | 4.73 | 5.44 |
| 2048 → 32 | fp16 | 12.55 | 1.02 | 13.88 | 14.02 | 14.60 |
| 2048 → 32 | w4a4 | 3.54 | 1.02 | 4.93 | 5.12 | 5.84 |
