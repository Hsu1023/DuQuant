# Hopper (sm_90 / GH200) backend

On Hopper, the same `DuQuantLinearCUDA` interface switches automatically to the kernels described here. The quantization format and checkpoints are identical to the Ada path. Dispatch depends only on the device compute capability and the shape.

## Why Hopper needs a separate path

Hopper has no INT4 Tensor Cores (the `IMMA.16864.U4.U4` used by the Ada path does not exist on sm_90), but it has INT8 `wgmma`. Therefore:

- **Weights** stay stored as packed u4 (unchanged memory footprint). Inside the kernel they are read from shared memory into registers and split into s8 (0..15) with `__byte_perm`, as the register operand of `wgmma` (CUTLASS sm90 mixed-input, RS mode).
- **Activations** are emitted by the quantization kernel directly as s8 codes (0..15, one byte each), as the shared-memory operand of `wgmma`.
- The integer accumulation is bit-identical to the Ada u4×u4 path. Zero-point correction, both scales and the bias are computed in the epilogue with the same expression and operation order as Ada (CUTLASS sm90 EVT).

## Implementation

| File | Contents |
|---|---|
| `gemm_sm90.cu` | INT8 wgmma GEMM (TMA + warp-specialized cooperative). The register operand of `wgmma` must be A, so the problem is transposed: `D'[N,M] = W[N,K]·X[M,K]^T`, and column-major `D'` is row-major `Y[M,N]`. Token-dimension tiles 16/32/64/128/256 (data-parallel) and 16/32/64/128 (Stream-K): 9 configs. |
| `linear_small.cuh` | Decode (M ≤ 4): the whole linear in one cooperative launch, see the next section. |
| `small_m.cuh` | Small-M dp4a u4 GEMV and shared helpers. |
| `backend.py` | Device detection, compilation (`-arch=sm_90a`), shape dispatch. |
| `kv4.py` | On Hopper, MHA KV4 decode uses 16-token tiles / 2 warps (128×128 tiles spill registers badly on sm_90). |

### Decode linear (M ≤ 4): a single launch

The original path used 4 launches per linear (two rotations, quantization, GEMV); at H100-class bandwidth the fixed overhead dominates. The new kernel does it all at once:

1. Worker CTA b (one per 128-wide block) computes the `t[perm[128b+i]]` values its block needs directly from the whole x row in shared memory. This is the same amount of work as the block rotation, and saves the global round trip of t and one barrier. It then multiplies by R2 to get z, and publishes the per-block row min/max.
2. After barrier 1, it computes the row scale/zero with the FP16 rounding sequence of `backend._quant`, quantizes and packs its block, and publishes the codes and the in-block code sums. The division `div_rn` uses a reciprocal-multiply fast path and falls back to IEEE division only when the quotient is near .5; the `nearbyint` result is bit-identical to `div_rn`.
3. After barrier 2, all CTAs run a persistent dp4a GEMV (`dp4a` directly on the two nibble planes, no unpacking) and the fused epilogue.
4. Weights do not depend on the activations:
   - Before the barrier, each CTA loads its first weight group into registers and prefetches more weights into L2.
   - This layer's streamed weights use an `evict_first` policy.
   - The rotation matrices, permutation and leading weights of the **next** linear (the model chains them qkv→o→gate_up→down) are prefetched into L2 (`evict_last`).

The barriers are 64-bit monotonic counters (never reset; the launch epoch is derived from the arrival count). There is one counter slot per (M bucket, config, with/without GEMV), because the epoch depends on the grid size. The cooperative launch is used only to guarantee that all CTAs are co-resident.

Both rotations run on Tensor Cores (`mma.m16n8k16`, FP32 accumulation, ascending k), and round bit-identically to the `tl.dot`/cuBLAS reference path. The first rotation computes only the columns this block needs: each mma takes 8 consecutive outputs along n, puts "source block of each output × token" in its 16 rows, and keeps only the diagonal elements. An earlier version accumulated sequentially with scalar FMA; it flipped one activation code in 2–4% of rows (relative error ~0.5%) and was dropped.

## Correctness

- New in `tests/test_cuda_layer.py`:
  - s8 activation codes are bit-identical to the packed codes;
  - the sm90 GEMM (9 configs, including tail tiles) equals a float64 reference bit for bit;
  - small-M activation codes, scales, zero points and code sums are bit-identical to the Triton path, and the GEMV matches the integer reference.
- On 2000 rows of random input (K=4096/11008), the small-M path has 0 activation-code differences from the torch reference.
- Test results: everything passes except tests that need calibrated checkpoints. `test_cuda_layer.py` used to disable gradients globally and leak that into other test modules; it now restores the mode in `tearDown`. `test_flatquant.py` has the same problem and was not changed.

## Calibration and quality (recalibrated on GH200)

The full LLaMA-2-7B calibration pipeline was rerun from scratch on GH200: calibrate (55 s) → RTN/GPTQ clipping search (3 + 3 min) → reconstruction training (3 shards concurrently on one GPU, 48 min) → export.
- The training tokens are bit-identical to the 4090 run (same hashes and sampling starts).
- The per-layer calibration files differ because floating point is not reproducible across GPUs.
- The training MSE is close to the 4090 record.
- The artifacts are in `benchmarks/cache/` (git-ignored); the reports are in `benchmarks/hopper/calibration/`.

WikiText-2 test, seqlen 2048, all 166 blocks (`benchmarks/eval_hopper_reconstruct.py`; the token hash matches the protocol recorded in the repository):

| model | 4090 (Ada) | GH200 (Hopper) |
|---|---:|---:|
| FP16 | 5.4721 | 5.4721 |
| reconstruction RTN | 5.8179 | 5.8170 |
| reconstruction GPTQ | 5.7909 | 5.8002 |

On the first 8 blocks of both exports, the per-block NLL of the CUDA backend and the independent integer reference are **identical** (difference 0).

LAMBADA and the six QA tasks were not re-measured: their data preparation scripts depend on the raw data under `/mnt/yichen` on the 4090 machine.

## Measurements (GH200 120GB, PyTorch 2.9.1+cu128, Triton 3.5.1, CUDA 12.9, CUTLASS 3.9.2)

### End to end (same protocol as the README)

The protocol is identical to the Ada table: same script and arguments, with the `scope`/`excluded`/memory fields and sample counts checked field by field.
- LLaMA-2-7B, batch 1, 1024-token prompt + 32 generated tokens, including all 32 layers and the LM head.
- CUDA Graphs, median of 10; the total is wall time (as in the README table).
- Summary in `benchmarks/hopper/performance/summary.json`.

| | prefill | decode / token | total (wall) | speedup | model storage | peak allocated |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | 23.45 ms | 5.06 ms | 180.25 ms | — | 12.55 GiB | 13.28 GiB |
| W4A4KV4 reconstruction RTN | 19.02 ms | 3.93 ms | 140.73 ms | **1.28×** | 3.55 GiB | 3.98 GiB (−70.1%) |

Model storage is 11.16 MiB larger than Ada's 3.54 GiB:
- 11.09 MiB are the Hopper-specific padded R1ᵀ/R2ᵀ copies and int32 permutations;
- 0.07 MiB are the barrier counters of the decode kernel.

The generated tokens are identical to FP16.

The speedup is much smaller than on the 4090 (2.66×) for two reasons:
- The GH200 FP16 baseline is strong: 5.06 ms per decode token, versus 14.9 ms on the 4090.
- The W4A4 activation transform of every linear (two rotations, permutation, per-token quantization) is a latency-bound fixed cost, which is a large share at H100-class bandwidth.

During decode, the 4 linears of a layer take about 88 µs in total (cold L2), while streaming their weights at full bandwidth needs only about 30 µs.

### Single layer (real LLaMA-2 layer-0 weights, `benchmarks/hopper/single_layer_final.json`)

Warm-cache / cold-cache (256 MiB flushed before each call) speedup over FP16:

| K → N | M=1 | M=16 | M=128 | M=2048 |
|---|---|---|---|---|
| 4096 → 4096 | 0.79 / 1.15 | 0.48 / 0.80 | 0.55 / 0.69 | 1.24 / 1.00 |
| 4096 → 11008 | 1.63 / 1.57 | 1.44 / 1.34 | 1.24 / 1.17 | 1.61 / 1.23 |
| 11008 → 4096 | 1.60 / 1.49 | 0.94 / 1.00 | 0.89 / 0.90 | 1.22 / 1.04 |

The GEMM itself reaches about 1.2–1.4 POPS at large M (2.1–2.4× FP16 cuBLAS).

The medium-M config sweep is in `benchmarks/hopper/gemm_config_sweep.json`. Stream-K helps the down projection (few output tiles, long K): 37.6 → 30.6 µs at M≤16 and 53.9 → 44.0 µs at M=128.

## Known limitations / next steps

- At medium M (2–512) W4A4 is still slower than FP16: the activation transform takes 3 Triton kernels, and the CUTLASS mixed-input GEMM has a fixed latency of about 12 µs with small tiles.
- The activation phase of the decode linear takes about 9–11 µs, mostly latency: 2.7–5 µs to load the cold R matrices, plus two global barriers. Possible directions:
  - replace the global barriers with thread block clusters + DSMEM;
  - overlap the prologue with the previous kernel using PDL;
  - keep all rotation matrices (8 MB) resident with L2 persistence.
- The prefill activation transform (40 µs at M=2048, 117 µs at K=11008) is still 3 separate kernels.
- KV4 decode attention takes about 14 µs/layer at 1k context with very low bandwidth utilization; it is worth rewriting in CUDA.

## Reproduction

```bash
source /work/nvme/bfne/duquant/env_hopper.sh  # conda env, CUDA 12.9, CUTLASS, model/data paths
python -m unittest discover -s tests
python benchmarks/bench_single_layer.py --model $DUQUANT_LLAMA2 --tokens 1,2,4,16,128,512,2048 --cold --output benchmarks/hopper/single_layer_final.json
python benchmarks/bench_model.py --mode fp16 --kv-bits 16 --prompt 1024 --new-tokens 32 --repeats 10 --output benchmarks/hopper/performance/fp16.json
PYTHONPATH=benchmarks python benchmarks/bench_llama_reconstruct.py --mode cuda --kv-bits 4 --prompt 1024 --new-tokens 32 --repeats 10 --output benchmarks/hopper/performance/rtn_kv4.json
python benchmarks/eval_hopper_reconstruct.py --output benchmarks/hopper/reconstruct_wikitext.json
```

Calibration pipeline (one GPU is enough): `slurm_scripts/calib_stage12.sh` and `slurm_scripts/calib_stage3.sh` (in `/work/nvme/bfne/duquant/`, outside the repository). They correspond to `calibrate_model.py` → `search_ptq.py --method rtn/gptq` → `prepare_llama_reconstruct.py` → `train_llama_reconstruct.py --shard 0/1/2` → `finish_llama_reconstruct.py` in the repository.
