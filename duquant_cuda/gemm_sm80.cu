// Ampere (sm_80 A100 / sm_86 A40) DuQuant W4A4 backend.
//
// Ampere has INT4 Tensor Cores (mma.sync m16n8k64 u4), so prefill reuses the
// Ada CUTLASS Sm80 u4 x u4 GEMM with packed activations and the fused affine
// epilogue (gemm.cu / fused_gemm.cuh), with a longer tile list for tuning.
// Decode (M <= 4) reuses the single-launch cooperative linear from the Hopper
// backend (linear_small.cuh), which only needs sm_80 features (mma.m16n8k16,
// cp.async, L2 cache hints).
// Built once per device architecture (build/gemm_sm80.so, build/gemm_sm86.so).

// X(id, BM, BN, BK, WM, WN, STAGES); ids 0-2 are the Ada tiles.
#define DUQUANT_GEMM_CONFIGS(X)                                                      \
  X(0, 32, 128, 128, 32, 64, 3) X(1, 64, 128, 128, 32, 64, 3)                          \
  X(2, 128, 128, 128, 64, 64, 3) X(3, 128, 256, 128, 64, 64, 3)                        \
  X(4, 256, 128, 128, 64, 64, 3) X(5, 128, 128, 256, 64, 64, 3)                        \
  X(6, 64, 128, 256, 32, 64, 3) X(7, 128, 128, 128, 64, 64, 4)                         \
  X(8, 64, 64, 128, 32, 32, 4) X(9, 32, 64, 128, 32, 32, 4)                            \
  X(10, 32, 128, 128, 32, 64, 4) X(11, 64, 128, 128, 32, 64, 4)
#include "gemm.cu"
#include "linear_small.cuh"
