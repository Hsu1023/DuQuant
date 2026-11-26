# Full LLaMA decoder layer experiment

`decoder.py` wires the W4A4 Linear into a complete LLaMA-2 decoder block. The main entry point `main.py` has not been switched to this backend; this is a full-layer prototype that can be run, validated and measured on its own. Continuous 32-layer generation and memory measurements are in [MODEL.md](MODEL.md).

## What is timed

One call includes two RMSNorms, all seven projections, RoPE, attention, SwiGLU, two residual additions, and the KV-cache write and read. QKV and gate/up are each merged into one projection; both paths use the same merging, fused RMSNorm/SwiGLU/RoPE and forced FP16 Flash SDPA.

The W4A4 path additionally includes the dual rotation, permutation, dynamic quantization, packing and INT4 GEMM/GEMV. After RoPE, Q/K go through a normalized Hadamard transform and per-head-token 4-bit asymmetric quantization; V goes through per-head-token 4-bit asymmetric quantization. Q/K clipping = 0.9, V clipping = 1.0. The historical timings on this page used a fused butterfly Hadamard; the current code uses a fused Tensor Core dot and matches the norm and SwiGLU rounding, see [CALIBRATED.md](CALIBRATED.md) for the final accuracy and speed. FMA contraction is disabled in RoPE to keep the FP16 intermediate rounding of the reference implementation.

Attention computation and KV **storage** are still FP16: W4A4 stores the quantize-dequantized K/V values. V quantization does not depend on the history length, so it is done once when writing the cache. No packed INT4 KV cache or INT4 attention GEMM is claimed here.

Prefill uses causal attention; decode has query length 1 and attends to the whole written history plus the current K/V. The history cache is initialized by a real prefill of the corresponding path, and the initialization is not counted in the single decode step. Each CUDA Graph fixes one position and repeatedly measures one step at that position; this is not the latency of a full generation loop.

## Measurement conditions

- RTX 4090, GPU 4; PyTorch 2.7.1+cu126, Triton 3.3.1, CUDA Toolkit 12.4, CUTLASS 3.4.1.
- **Real layer-0 weights** of the local LLaMA-2-7B, hidden=4096, intermediate=11008, 32 attention heads, head_dim=128.
- The input is the token embedding of real tokenizer output; the text is fixed in the script and repeated to the required length.
- Fixed random rotations and permutations; QKV share one set of parameters and gate/up share another. No DuQuant calibration, SmoothQuant or LWC is used yet.
- Both sides use CUDA Graphs; the median of three measurements is reported, alternating the order. All online GPU work is counted; construction, weight export, compilation and history-cache initialization are not.
- Cold-cache results write a 256 MiB L2 pollution buffer before each call; the flush is excluded from timing with external events.
- The FP16 baseline was checked numerically against the Hugging Face LLaMA decoder layer; the slower eager attention of the original repository is not used as the speed baseline.

## Correctness and current limitations

Two independent references are kept so that different arithmetic models are not conflated:

1. `integer_reference`: represents the 0–15 integer codes in FP32, computes the dense dot products independently, then applies zero-point correction and scaling. For the measured K the dot products stay below `225*K < 2^24`, so FP32 represents these integer accumulations exactly. It does not call the INT4 kernel. This path uses plain PyTorch norm, RoPE, Hadamard and SwiGLU.
2. `reference`: conventional fake quantization: every quantized value is dequantized and rounded to FP16 first, then a floating-point GEMM is run. This differs slightly from integer accumulation followed by one scaling, and subsequent 4-bit quantization amplifies changes that cross a rounding boundary.

In this batch=1/4 real-weight experiment the relative L2 difference between the CUDA output and the integer-arithmetic reference is **0–1.11%**; the three batch=1 decode points and the batch=4 decode point match element for element. The difference to the conventional fake-quant reference is **2.86–5.02%**. All these differences are stored in the JSON files and are not used as a quality metric for the FP16 model.

For the uncalibrated experiment on this page, the integer reference differs from the FP16 output by roughly **11–17%** relative L2. These historical results show that the full-layer prototype has a real performance gain; **they do not show that the full DuQuant model reaches deployable accuracy**. Calibration, full-model numeric comparisons, PPL and end-to-end speed re-measurements were done later, see [CALIBRATED.md](CALIBRATED.md).

The tests cover: comparison with the Hugging Face FP16 layer, an independent attention-preprocessing comparison, full-layer integer/fake-quant comparisons, token-by-token decode against causal prefill, multiple batches, unchanged cache prefixes, CUDA Graph replay with updated inputs, and memcheck. The log is `benchmarks/decoder_memcheck.log`.

## Reproduction

From the repository root, with the single-layer CUDA environment:

```bash
export CUDA_VISIBLE_DEVICES=4
export CUDA_HOME=/usr/local/cuda-12.4
export CUTLASS_PATH=/mnt/yichen/wyc/QDLM/QuaRot/third-party/cutlass
export TRITON_CACHE_DIR=/tmp/duquant_triton
export DUQUANT_PYTHON=/home/yichen/miniconda3/envs/hb/bin/python

"$DUQUANT_PYTHON" -m unittest discover -s tests -v
"$DUQUANT_PYTHON" benchmarks/bench_decoder.py --cold \
  --output benchmarks/decoder_reproduced_b1.json
"$DUQUANT_PYTHON" benchmarks/bench_decoder.py --batch 4 \
  --prefill 128 --decode-context 2048 --cold \
  --output benchmarks/decoder_reproduced_b4.json
```

The interface is `LlamaDecoderBlock(weights, rotations, mode='cuda')`, `cache = block.cache(batch, capacity)`, `y = block(x, cache, position)`. Currently supported: MHA with head_dim=128, prefill from an empty cache and single-token decode. Not supported: GQA, chunked prefill with history, training backward, or whole-sequence generation graphs with dynamic positions.

## Measured results

From `benchmarks/decoder_final_b1.json` and `benchmarks/decoder_final_b4.json`, in μs. For decode, the length is the number of history tokens before the call.

| batch | scenario | input/history length | FP16 | W4A4 | speedup | cold-cache speedup |
|---:|---|---:|---:|---:|---:|---:|
| 1 | prefill | 128 | 550.5 | 220.8 | 2.49× | 2.45× |
| 1 | prefill | 512 | 1440.5 | 416.9 | 3.46× | 3.47× |
| 1 | prefill | 2048 | 5733.2 | 1729.2 | 3.32× | 3.25× |
| 1 | decode | 128 | 435.2 | 162.5 | 2.68× | 2.20× |
| 1 | decode | 2048 | 454.6 | 196.2 | 2.32× | 2.05× |
| 1 | decode | 4095 | 470.2 | 230.5 | 2.04× | 1.96× |
| 4 | prefill | 128 | 1410.2 | 395.8 | 3.56× | 3.60× |
| 4 | decode | 2048 | 585.3 | 310.5 | 1.88× | 1.76× |
