# Ampere (sm_80 A100 / sm_86 A40) backend

On Ampere, the same `DuQuantLinearCUDA` interface switches automatically to the kernels described here. The quantization format and checkpoints are identical to the Ada/Hopper paths. Dispatch depends only on the device compute capability and the shape.

## Relation to Ada / Hopper

Ampere has INT4 Tensor Cores (`mma.sync m16n8k64` u4), so unlike Hopper there is no need to widen the weights to s8:

- **Prefill / medium and large M**: reuses the Ada CUTLASS Sm80 u4×u4 GEMM (packed activations, fused affine epilogue, `gemm.cu` / `fused_gemm.cuh`). It is compiled per device architecture (`-arch=sm_80` / `sm_86`) with an extended tile list, and A100 and A40 each have their own shape-based dispatch table.
- **Decode (M ≤ 2)**: reuses the Hopper single cooperative-launch decode linear (`linear_small.cuh`: Tensor Core dual rotation + permutation, A4 quantization, dp4a GEMV, cross-layer L2 prefetch). It only needs sm_80 features (`mma.m16n8k16`, `cp.async`, L2 cache hints, `createpolicy`). However, the original implementation assumed one resident worker CTA per 128-block, with R1ᵀ and R2ᵀ both in shared memory, which does not hold on the A40 (see below).

Activation codes, scales, zero points and code sums are bit-identical to the Triton path; integer accumulation is identical to the Ada path.

## Implementation

| File | Contents |
|---|---|
| `gemm_sm80.cu` | Ampere translation unit: defines 12 tile configs, then includes `gemm.cu` and `linear_small.cuh`. Builds `build/gemm_sm80.x86_64.so` / `build/gemm_sm86.x86_64.so` (library names carry the host CPU architecture because `build/` is shared between x86 and aarch64 nodes). |
| `gemm.cu`, `fused_gemm.cuh` | Tile configs become the macro table `DUQUANT_GEMM_CONFIGS(X)` (with a new stage-count parameter). Ada still compiles only the original 3 configs; its behavior is unchanged. |
| `linear_small.cuh` | Device-dependent layouts: one worker can own several 128-blocks (`own`), and R2ᵀ can reuse the shared-memory slot of R1ᵀ (`alias_r2`). New dry-run entry point `duquant_linear_small_plan`. |
| `backend.py` | `load_gemm_ampere(arch)`; `sm80_config` / `sm86_config` GEMM dispatch; decode GEMV config, `small_m` and prefetch sizes set per architecture; `small_fits` falls back to the GEMM path when no layout fits. |
| `kv4.py` | MHA KV4 decode also uses 16-token tiles / 2 warps on Ampere. |

### Decode-kernel layout selection

The original kernel needs one resident worker CTA per 128-wide block. Each worker's shared memory must hold R1ᵀ and R2ᵀ (33 KB each, padded to 132 columns) and the whole x row.

| | SMs | max shared memory per block |
|---|---:|---:|
| GH200 / H200 | 132 | 227 KB |
| A100 | 108 | 163 KB |
| A40 | 84 | 99 KB |

For example, on the A40, K=11008 (86 blocks) with M=1 needs 92 KB. Only 1 CTA fits per SM, and 84 < 86, so the original kernel cannot launch. The new host-side `launch_linear_small` tries `own = 1..4` × `alias_r2 ∈ {0,1}` in order. It picks the first layout that fits in shared memory and satisfies `own × min(blocks, resident CTAs) ≥ blocks`:

- `own > 1`: worker w owns blocks w, w+W, w+2W, … (W workers). Rotation A, rotation B, min/max and quantization loop over every owned block; the barrier arrival count becomes W.
- `alias_r2`: R2ᵀ is loaded into the R1ᵀ slot only after rotation A has finished for all owned blocks. This saves 33 KB, at the cost of not overlapping the R2 load with rotation A.

On Hopper and A100 (LLaMA's K=4096/11008), the choice is always `own=1, alias_r2=0`: the original path, with unchanged shared memory size and grid. On the A40, K=11008 uses `own=2` (M=1) or `own=2, alias_r2=1` (M=2). On the A40, 3–4 rows at K≥11008 have no feasible layout; `small_fits` returns False and forward uses the GEMM path. LLaMA-3's K=14336 also uses `own=2` / `alias_r2` on A100. The tests cover all of these layouts.

### Not adopted: staging weights in shared memory during decode

The shared memory of non-worker CTAs is idle during the activation phase. One experiment used `cp.async` during the activation phase to load each CTA's next weight groups into shared memory, overlapping the weight stream with rotation/quantization. It was slower:

- A100 qkv (M=1, cold L2): 42.9 → 46.0 µs;
- A40: 59.4 → 64.5 µs.

With HBM saturated by the bulk reads, the latency of the workers' R1/x loads grows substantially; on A100 the activation phase stretched from 14.6 µs to 21.5 µs. The change was removed.

## Correctness

- `tests/test_cuda_layer.py`: the small-M test now also runs on Ampere, with M=1–4 and an extra K=14336 (covering `own>1` and `alias_r2`). In every feasible layout, activation codes, scales, zero points and code sums are bit-identical to the Triton path, and the GEMV matches the integer reference. The GEMM test iterates over every tile config of the current device.
- Hopper regression: on H200 (Delta `gpuH200x8`, sm_90a), all 20 tests in `test_cuda_layer`, `test_decoder`, `test_model`, `test_gqa` and `test_kv4` pass. This includes the bit-exact sm90 GEMM comparison and the decode kernel going through the new layout planner. On Hopper the planner chooses `own=1, alias_r2=0`, the same as the original implementation.
- `python -m unittest discover -s tests` on A100 and A40 (42 tests): all pass except the following 4.
  - 3 need Qwen3 calibration caches (`benchmarks/cache/qwen3_compare/…`) that are not on this machine.
  - `test_llama_reconstruct` passes on its own, but fails with "does not require grad" when run after other test modules that run CUDA inference first. This reproduces on the unmodified HEAD as well, is unrelated to this backend, and was not addressed.
- WikiText-2 with the same checkpoint (seqlen 2048, all 166 blocks, `benchmarks/eval_hopper_reconstruct.py`; the script is hardware-independent):

| model | A100 | A40 | GH200 (HOPPER.md) |
|---|---:|---:|---:|
| FP16 | 5.4721 | 5.4721 | 5.4721 |
| reconstruction RTN | 5.8235 | 5.8236 | 5.8170 |
| reconstruction GPTQ | 5.7967 | 5.7967 | 5.8002 |

On the first 8 blocks of both exports, the per-block NLL of the CUDA backend and the independent integer reference are **identical** (on both A100 and A40). This uses the checkpoint calibrated and exported on GH200 (`benchmarks/cache/llama_reconstruct/`); it was not recalibrated on Ampere. The PPL differences from GH200 come from the rounding of the FP16 parts (attention, rotations, cuBLAS) on different GPUs.

## Measurements (PyTorch 2.9.1+cu128, Triton 3.5.1, CUDA 12.9, CUTLASS 3.9.2)

Hardware: A100-SXM4-40GB (Delta `gpuA100x4`), A40 48GB (Delta `gpuA40x4`).

### End to end (same protocol as the README)

- Same script and arguments: LLaMA-2-7B, batch 1, 1024-token prompt + 32 generated tokens, including all 32 layers and the LM head.
- CUDA Graphs, median of 10; the total is wall time; peak memory includes capture/warmup and the graph pool.
- Summaries in `benchmarks/ampere/{a100,a40}/performance/summary.json` (`benchmarks/summarize_ampere.py`). On both GPUs the generated tokens are identical to FP16.

| GPU | | prefill | decode / token | total (wall) | speedup | model storage | peak allocated |
|---|---|---:|---:|---:|---:|---:|---:|
| A100 | FP16 | 63.46 ms | 11.65 ms | 424.68 ms | — | 12.55 GiB | 13.24 GiB |
| A100 | W4A4KV4 reconstruction RTN | 31.60 ms | 6.41 ms | 230.50 ms | **1.84×** | 3.55 GiB | 3.95 GiB (−70.2%) |
| A40 | FP16 | 140.85 ms | 25.55 ms | 933.05 ms | — | 12.55 GiB | 13.24 GiB |
| A40 | W4A4KV4 reconstruction RTN | 62.75 ms | 9.19 ms | 347.72 ms | **2.68×** | 3.55 GiB | 3.95 GiB (−70.2%) |

### Tuning highlights

- **GEMM tiles** (`benchmarks/ampere/*/gemm_config_sweep.json`, full-layer time, `benchmarks/sweep_gemm_configs.py`):
  - Smaller tiles pay off clearly at small M. On A100 with 32×64 / 64×64 / 64×128×256 tiles, the down projection drops from 50.7 to 39.5 µs at M=16 and from 83.5 to 67.7 µs at M=128. At M=256, a 128×128×256 tile gives 97.4 → 82.7 µs.
  - At large M, Ada's 128×128×128 tile is still best. On the A40 at M=4096, qkv / gate_up switch to 256×128: 1365 → 1325 µs and 2319 → 2114 µs.
- **Decode linear** (`benchmarks/ampere/*/small_m_sweep.json`, cold L2, `benchmarks/sweep_small_m.py`):
  - Hopper's default 16 MiB L2 weight warm-up is slower on Ampere: 0 on A100, 4 MiB on A40.
  - For 3–4 rows the cooperative kernel is slower than the GEMM path, so Ampere uses `small_m=2`.
  - Single linear at M=1, cold L2:

| K → N | A100 FP16 | A100 W4A4 | A40 FP16 | A40 W4A4 |
|---|---:|---:|---:|---:|
| 4096 → 12288 (qkv) | 105.5 µs | 41.3 µs | 191.5 µs | 59.4 µs |
| 4096 → 4096 (o) | 45.5 µs | 27.1 µs | 72.7 µs | 30.7 µs |
| 4096 → 22016 (gate_up) | 167.5 µs | 57.8 µs | 357.4 µs | 94.2 µs |
| 11008 → 4096 (down) | 95.9 µs | 51.8 µs | 171.0 µs | 62.5 µs |

- **Cross-layer prefetch**: A100 keeps 8 MiB (0 MiB: 238.5 ms, 16 MiB: 240.5 ms). The A40 has only a 6 MB L2 and uses 2 MiB (354.6 → 347.6 ms).
- **KV4 decode** (`benchmarks/ampere/*/kv4_decode_sweep.json`, `benchmarks/sweep_kv4_decode.py`): at 1k context, A100 33.8 → 26.8 µs/layer, A40 41.0 → 28.4 µs/layer.

## Known limitations / next steps

- A100 decode takes 6.4 ms per token, while streaming the weights at full bandwidth needs only about 2.3 ms. The phase trace (`layer.small_trace`) for qkv on A100 (M=1, cold L2) shows:
  - an activation phase of about 14.6 µs, of which about 6.7 µs is the first read of R1/x;
  - a following GEMV phase of about 19 µs;
  - about 36 µs for the whole kernel, which reads 25 MB of weights (about 0.7 TB/s on average).

  Cross-layer prefetch partly hides the R1/x read; fixed latency (two global barriers, the launch) is the main remaining cost.
- As on Hopper, the activation transform at medium M is still 3 Triton kernels. On the A40 at M=4096 and K=11008 it takes about half the time (858 / 1691 µs).
- On the A40, 3–4 rows at K≥11008 do not use the cooperative kernel (not enough shared memory).

## Reproduction

```bash
source /work/nvme/bfne/duquant/env_ampere.sh  # conda env, CUDA 12.9 (HPC SDK), CUTLASS, model/data paths
python -m unittest discover -s tests
G=a100  # or a40
python benchmarks/sweep_gemm_configs.py --output benchmarks/ampere/$G/gemm_config_sweep.json
python benchmarks/sweep_small_m.py --output benchmarks/ampere/$G/small_m_sweep.json
python benchmarks/sweep_kv4_decode.py --output benchmarks/ampere/$G/kv4_decode_sweep.json
python benchmarks/bench_model.py --mode fp16 --kv-bits 16 --prompt 1024 --new-tokens 32 --repeats 10 --output benchmarks/ampere/$G/performance/fp16.json
PYTHONPATH=benchmarks python benchmarks/bench_llama_reconstruct.py --mode cuda --kv-bits 4 --prompt 1024 --new-tokens 32 --repeats 10 --output benchmarks/ampere/$G/performance/rtn_kv4.json
python benchmarks/summarize_ampere.py $G
python benchmarks/eval_hopper_reconstruct.py --output benchmarks/ampere/$G/reconstruct_wikitext.json
```
