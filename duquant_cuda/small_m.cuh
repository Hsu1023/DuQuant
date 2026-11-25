#pragma once
// Small-M (decode) kernels for the Hopper backend. Plain CUDA, sm_80+.
//
// duquant_gemv_u4: packed u4 activations x packed u4 weights for M <= 8 rows.
// duquant_linear_small (linear_small.cuh): the whole decode linear in one
//   cooperative launch.
// Both nibble streams use the same interleave, so dp4a on the low and high
// nibble planes multiplies matching elements without unpacking.
#include <cuda_fp16.h>
#include <stdint.h>

namespace duq {

__device__ __forceinline__ float div_full(float a, float b) {
  // Triton lowers fp32 '/' to div.full.f32; match it bit for bit.
  float r;
  asm("div.full.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
  return r;
}
__device__ __forceinline__ float f16r(float v) { return __half2float(__float2half_rn(v)); }

template <int THREADS>
__device__ __forceinline__ float block_reduce(float v, float* red, bool is_max) {
  for (int o = 16; o > 0; o >>= 1) {
    float u = __shfl_xor_sync(0xffffffffu, v, o);
    v = is_max ? fmaxf(v, u) : fminf(v, u);
  }
  int w = threadIdx.x / 32, l = threadIdx.x % 32;
  __syncthreads();
  if (l == 0) red[w] = v;
  __syncthreads();
  v = red[l < THREADS / 32 ? l : 0];
  for (int o = 16; o > 0; o >>= 1) {
    float u = __shfl_xor_sync(0xffffffffu, v, o);
    v = is_max ? fmaxf(v, u) : fminf(v, u);
  }
  return v;
}

template <int THREADS>
__device__ __forceinline__ int block_sum(int v, int* red) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  int w = threadIdx.x / 32, l = threadIdx.x % 32;
  __syncthreads();
  if (l == 0) red[w] = v;
  __syncthreads();
  v = l < THREADS / 32 ? red[l] : 0;
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}

__device__ __forceinline__ int dot_u4x32(uint4 w, uint4 a, int acc) {
  const uint32_t* wv = reinterpret_cast<const uint32_t*>(&w);
  const uint32_t* av = reinterpret_cast<const uint32_t*>(&a);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    acc = __dp4a(int(wv[i] & 0x0F0F0F0Fu), int(av[i] & 0x0F0F0F0Fu), acc);
    acc = __dp4a(int((wv[i] >> 4) & 0x0F0F0F0Fu), int((av[i] >> 4) & 0x0F0F0F0Fu), acc);
  }
  return acc;
}

__device__ __forceinline__ uint4 ldg_stream(const uint4* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

// Each warp owns RPW consecutive output channels; activations sit in shared memory.
template <int MR, int RPW, int WARPS>
__global__ void __launch_bounds__(WARPS * 32) gemv_u4(
    const uint8_t* __restrict__ a, const uint8_t* __restrict__ w, __half* __restrict__ y,
    int m, int n, int k, const __half* __restrict__ as, const __half* __restrict__ az,
    const int* __restrict__ ax, const __half* __restrict__ ws, const __half* __restrict__ wz,
    const int* __restrict__ wc, const __half* __restrict__ bias) {
  extern __shared__ uint4 sa[];
  const int kb = k / 32;  // 16-byte vectors per row
  for (int i = threadIdx.x; i < m * kb; i += WARPS * 32) sa[i] = reinterpret_cast<const uint4*>(a)[i];
  __syncthreads();
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int row0 = (blockIdx.x * WARPS + warp) * RPW;
  if (row0 >= n) return;
  const uint4* wr = reinterpret_cast<const uint4*>(w) + size_t(row0) * kb;
  int acc[RPW][MR];
#pragma unroll
  for (int r = 0; r < RPW; ++r)
#pragma unroll
    for (int i = 0; i < MR; ++i) acc[r][i] = 0;
#pragma unroll 2
  for (int j = lane; j < kb; j += 32) {
    uint4 wv[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r)
      wv[r] = row0 + r < n ? ldg_stream(wr + size_t(r) * kb + j) : make_uint4(0, 0, 0, 0);
#pragma unroll
    for (int i = 0; i < MR; ++i) {
      if (i < m) {
        uint4 av = sa[i * kb + j];
#pragma unroll
        for (int r = 0; r < RPW; ++r) acc[r][i] = dot_u4x32(wv[r], av, acc[r][i]);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < RPW; ++r)
#pragma unroll
    for (int i = 0; i < MR; ++i)
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) acc[r][i] += __shfl_xor_sync(0xffffffffu, acc[r][i], o);
  // Spread the epilogue over lanes: lane = r * MR + i.
#pragma unroll
  for (int r = 0; r < RPW; ++r) {
#pragma unroll
    for (int i = 0; i < MR; ++i) {
      if (lane == (r * MR + i) % 32 && i < m && row0 + r < n) {
        const int col = row0 + r;
        float out = (float(acc[r][i]) + __half2float(az[i]) * float(wc[col]) -
                     __half2float(wz[col]) * float(ax[i])) *
                        __half2float(as[i]) * __half2float(ws[col]) +
                    __half2float(bias[col]);
        y[size_t(i) * n + col] = __float2half_rn(out);
      }
    }
  }
}

}  // namespace duq

template <int MR, int RPW, int WARPS>
static int launch_gemv(const void* a, const void* w, void* y, int m, int n, int k, cudaStream_t st,
                       const void* as, const void* az, const void* ax, const void* ws,
                       const void* wz, const void* wc, const void* bias) {
  size_t smem = size_t(MR) * k / 2;
  if (smem > 48 * 1024) {
    auto err = cudaFuncSetAttribute(duq::gemv_u4<MR, RPW, WARPS>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, int(smem));
    if (err != cudaSuccess) return -int(err);
  }
  int rows_per_block = RPW * WARPS;
  duq::gemv_u4<MR, RPW, WARPS><<<(n + rows_per_block - 1) / rows_per_block, WARPS * 32, smem, st>>>(
      (const uint8_t*)a, (const uint8_t*)w, (__half*)y, m, n, k, (const __half*)as,
      (const __half*)az, (const int*)ax, (const __half*)ws, (const __half*)wz, (const int*)wc,
      (const __half*)bias);
  return -int(cudaGetLastError());
}

// config selects rows-per-warp: 0 -> 1, 1 -> 2, 2 -> 4.
template <int MR>
static int gemv_rows(int config, const void* a, const void* w, void* y, int m, int n, int k,
                     cudaStream_t st, const void* as, const void* az, const void* ax,
                     const void* ws, const void* wz, const void* wc, const void* bias) {
  if (config == 0) return launch_gemv<MR, 1, 8>(a, w, y, m, n, k, st, as, az, ax, ws, wz, wc, bias);
  if (config == 1) return launch_gemv<MR, 2, 8>(a, w, y, m, n, k, st, as, az, ax, ws, wz, wc, bias);
  return launch_gemv<MR, 4, 8>(a, w, y, m, n, k, st, as, az, ax, ws, wz, wc, bias);
}

extern "C" int duquant_gemv_u4(const void* a, const void* w, void* y, int m, int n, int k,
                               void* stream, int config, const void* as, const void* az,
                               const void* ax, const void* ws, const void* wz, const void* wc,
                               const void* bias) {
  auto st = (cudaStream_t)stream;
  if (m <= 1) return gemv_rows<1>(config, a, w, y, m, n, k, st, as, az, ax, ws, wz, wc, bias);
  if (m <= 2) return gemv_rows<2>(config, a, w, y, m, n, k, st, as, az, ax, ws, wz, wc, bias);
  if (m <= 4) return gemv_rows<4>(config, a, w, y, m, n, k, st, as, az, ax, ws, wz, wc, bias);
  if (m <= 8) return gemv_rows<8>(config, a, w, y, m, n, k, st, as, az, ax, ws, wz, wc, bias);
  return 9999;
}
