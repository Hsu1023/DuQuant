# DuQuant single-layer W4A4 CUDA prototype

Implements the complete online path of one layer: two 128×128 block rotations, a cross-block permutation, per-token asymmetric activation quantization and packing, real INT4 GEMM/GEMV, zero-point correction and FP16 output. Weight rotation, quantization and packing run once at construction; inference never reconstructs the full floating-point weight and caches neither activations nor outputs.

## Implementation

Hopper (sm_90 / GH200) uses a separate INT8 wgmma GEMM and a single-launch decode kernel, with the same interface and automatic dispatch, see [HOPPER.md](HOPPER.md).
Ampere (sm_80 A100 / sm_86 A40) reuses the INT4 GEMM (compiled and dispatched per device) and the Hopper single-launch decode kernel, see [AMPERE.md](AMPERE.md).

- `backend.py`: PyTorch interface, Triton dual rotation, quantization/packing, single-token packed INT4 GEMV.
- `gemm.cu`: CUTLASS unsigned INT4 Tensor Core GEMM; an INT32-output version is kept for comparison.
- `fused_gemm.cuh`: fuses the asymmetric zero-point correction, scaling, bias and FP16 write-back into the GEMM epilogue, removing the INT32 intermediate matrix.
- `kv4.py`: real packed KV4 cache, fused RoPE/quantization/write, split-K decode that reads the compressed cache directly; GQA shares the unpacked values and uses Tensor Cores. The full model also supports explicit KV16, see the [KV mode evaluation](CALIBRATED.md#kv-modes).
- `ptq.py`: clipping search and GPTQ export for symmetric W4; A/K/V stay asymmetric, with per-layer searched clipping. Pre-quantized weights go straight into the existing integer kernels, see the [LLaMA-2 ablation](CALIBRATED.md#ptq-search).
- The rotation intermediate uses a `[K,M]` layout; the second rotation fuses the permutation into its loads, improving coalescing of the gather.
- Quantization uses correctly rounded division, matching the original repository's FP16 scale, zero point, clipping and round-to-even rules.
- Small-batch decode uses a separate GEMV; other shapes use GEMM tiles chosen by M/N. All choices depend only on the shape.

`IMMA.16864.U4.U4.SAT` was confirmed in the compiled SASS. The final forward is usually four GPU kernel launches (dual rotation, quantization, GEMM/GEMV).

## Scope

This file covers the standalone Linear. The full decoder block is in [DECODER.md](DECODER.md); accuracy, continuous generation and memory of the current calibrated 32-layer model are in [CALIBRATED.md](CALIBRATED.md). The `QuantLinear` of the original main entry point has not been replaced.

- Validation platform: RTX 4090, `sm_89`, PyTorch 2.7.1+cu126, Triton 3.3.1, CUDA Toolkit 12.4.
- Build dependency: CUTLASS 3.4.1 (local checkout `ffa34e70`), set through `CUTLASS_PATH`; external dependencies are unmodified.
- Contiguous FP16 input `[...,K]`; K a multiple of 128, N a multiple of 32. Multi-dimensional batches, bias, non-default CUDA streams and CUDA Graphs are supported.
- Two 128×128 rotation matrices shared by all blocks, and one channel permutation of length K; this matches the structure of the original repository's default `permutation_times=1`.
- Asymmetric per-output-channel weight quantization, per-token activation quantization, no weight groups.
- Not yet supported: other bit widths, BF16, extra online `smooth_scales`/LET, or an arbitrary number of rotations. Offline-folded smoothing and per-channel LWC weight clipping are supported, see [CALIBRATED.md](CALIBRATED.md); any additional online scaling would also have to be timed.
- The single-Linear timings in this file exclude RMSNorm, SwiGLU, attention and the KV cache; the full-layer experiment includes them, see [DECODER.md](DECODER.md). Full-model calibration and PPL validation are in [CALIBRATED.md](CALIBRATED.md).

## Usage and reproduction

Run from the repository root. The dependencies exist on this machine; the paths below are the environment used for these measurements.

```bash
export CUDA_VISIBLE_DEVICES=4
export CUDA_HOME=/usr/local/cuda-12.4
export CUTLASS_PATH=/mnt/yichen/wyc/QDLM/QuaRot/third-party/cutlass
export TRITON_CACHE_DIR=/tmp/duquant_triton
export DUQUANT_PYTHON=/home/yichen/miniconda3/envs/hb/bin/python

# The first construction compiles the CUDA library; outputs and compile logs go to duquant_cuda/build/.
# Construct and warm up before CUDA Graph capture.
"$DUQUANT_PYTHON" -m unittest discover -s tests -v
"$DUQUANT_PYTHON" benchmarks/bench_single_layer.py --cold

# Use the real Q/up/down weights of LLaMA-2 layer 0 (activations and rotations are still synthetic).
"$DUQUANT_PYTHON" benchmarks/bench_single_layer.py \
  --model /mnt/public_models/llama2-7b-hf --tokens 1,128,2048 --cold \
  --output benchmarks/llama2_weights_reproduced.json
```

```python
from duquant_cuda import DuQuantLinearCUDA

# weight: original [N,K] FP16 CUDA weight; r1/r2: [128,128] FP16 CUDA;
# perm: CUDA integer permutation vector. The constructor does the offline weight transform and packing.
layer = DuQuantLinearCUDA(weight, r1, perm, r2,
                          bias=bias, act_clip=0.9, weight_clip=0.8)
y = layer(x)  # x: contiguous FP16 CUDA [...,K]
```

`--tune` searches launch configurations before timing; the final result files use the default shape-based dispatch, so no online autotuning is needed. Delete `duquant_cuda/build/` and recompile after switching CUDA/CUTLASS versions. The prototype is inference-only and provides no backward.

## Measurement method and accuracy

The baseline is `torch.nn.functional.linear(x, original_fp16_weight)`. Both sides use CUDA Graphs; DuQuant timing includes all online rotation, permutation, quantization, packing and matrix multiplication. Offline weight export and first compilation are not counted in inference latency.

Warm-cache measurements repeat the single layer; cold-cache measurements write a 256 MiB buffer before every call and exclude the flush from timing with CUDA Graph external events, reporting the median of 30 runs. The cold-cache numbers are still single-layer experiments and cannot be taken as end-to-end model results. GPU clocks are not locked; microsecond-level marginal gains should be re-measured.

The final GPU 4 measurements are in `benchmarks/final_synthetic.json` and `benchmarks/final_llama2_weights.json`. The latter uses the real layer-0 weights of LLaMA-2-7B, but random activations and fixed random rotations and permutations; it is **not a calibrated full DuQuant model**. At all 24 final measurement points the packed activation bytes match the PyTorch reference; the L2 error relative to the fake-quant output is about 0.03%–0.05%. The fused computation does not reproduce every FP16 dequantization rounding step of fake quantization, so the output is not promised to be bit-identical. This error is not the accuracy loss of W4A4 relative to the original FP16 model.

The numeric tests cover the original repository's quantizer, zero/constant/one-sided inputs, non-power-of-two K, M/N tail tiles, both intermediate layouts, fused and unfused GEMM, GEMV, bias, multi-dimensional batches, non-default streams and changing inputs after graph replay. `benchmarks/memcheck.log` holds the memcheck results. `benchmarks/history/` keeps results of older versions from the optimization process; they do not represent current performance.

### Results with LLaMA-2 layer-0 weights

| K → N | tokens | FP16 μs | DuQuant μs | warm-cache speedup | cold-cache speedup |
|---|---:|---:|---:|---:|---:|
| 4096 → 4096 | 1 | 16.13 | 11.28 | 1.43× | 2.39× |
| 4096 → 4096 | 128 | 31.68 | 17.96 | 1.76× | 1.47× |
| 4096 → 4096 | 2048 | 419.15 | 126.19 | 3.32× | 2.94× |
| 4096 → 11008 | 1 | 93.76 | 15.81 | 5.93× | 2.59× |
| 4096 → 11008 | 128 | 119.50 | 25.75 | 4.64× | 3.13× |
| 4096 → 11008 | 2048 | 1170.47 | 248.97 | 4.70× | 4.39× |
| 11008 → 4096 | 1 | 94.86 | 20.31 | 4.67× | 2.40× |
| 11008 → 4096 | 128 | 118.54 | 37.27 | 3.18× | 2.34× |
| 11008 → 4096 | 2048 | 1160.23 | 349.58 | 3.32× | 2.94× |

## Next steps

The warm-cache gain for 16 tokens at 4096×4096 is only about 1.05–1.08×, so small-M GEMM parallelism is still worth optimizing. For long inputs to the down projection, rotation and quantization still take a large share. The full model now reuses the input transforms of QKV and gate/up, and calibration, PPL and prefill/decode evaluation are done; these overheads and the calibration accuracy can still be improved.
