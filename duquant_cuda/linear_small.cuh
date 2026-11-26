#pragma once
// Whole decode-size DuQuant linear (M <= 4) in one cooperative launch.
//
// Worker CTA b (one per 128-wide block, b < K/128):
//   A  g[i] = (x @ blockdiag(R1))[perm[128b+i]]  computed directly from x in
//           shared memory: no global t and no barrier between the rotations.
//   B  z = g @ R2.
//      Both rotations run on Tensor Cores (mma m16n8k16, FP32 accumulate,
//      k ascending, FP16 result) so t and z round exactly like the tl.dot /
//      cuBLAS reference path; scalar FMA accumulation flipped an activation
//      code in ~2-4% of rows. Per-block row min/max is published; barrier 1.
//   C  row scale/zero with the backend._quant FP16 sequence; quantize/pack the
//      own block; publish codes and block code sums; barrier 2.
// Every CTA: persistent dp4a u4 GEMV + fused affine epilogue.
// Weights do not depend on the activation: each CTA loads the first part of
// its rows into registers and prefetches more into L2 before the barrier.
// Cross-layer prefetch: the next linear's rotations, permutation and leading
// weight bytes are prefetched into L2 (evict_last) while this one runs; this
// layer's streamed weights use an evict_first policy so they do not push the
// prefetched data out.
//
// Quantizing per block (not the full row in every CTA) keeps the redundant
// work independent of K; the second barrier is cheaper than that work.
//
// The cooperative launch only guarantees co-residency. Barriers use 64-bit
// counters that are never reset; the launch epoch comes from an arrival counter
// (one counter slot per distinct grid size, see backend.small_counters).
//
// Smaller GPUs (Ampere; A40: 84 SMs, 99 KB shared memory per block) cannot
// always hold one resident worker per 128-block with both rotations staged:
//   own > 1   worker CTA w handles blocks w, w + W, w + 2W, ... (W workers);
//   alias_r2  R2^T is loaded into the R1^T slot after every owned block has
//             finished rotation A (no overlap of the R2 load, ~33 KB less smem).
// The host picks the first fitting layout in launch_plan; Hopper and A100 use
// own = 1 without aliasing (the original single-block worker path).
#include "small_m.cuh"

namespace duq {

struct SmallParams {
  const __half* x; const __half* r1t; const __half* r2t; const int* perm;
  const uint8_t* w; __half* y;
  int m, n, k; float clip;
  const __half* ws; const __half* wz; const int* wc; const __half* bias;
  // Written and re-read inside the kernel by other CTAs. Reads go through
  // __ldcg (L2 only): L1 is not coherent across SMs, and the caching allocator
  // hands the same scratch addresses to consecutive launches, so a plain load
  // can hit a stale L1 line left by the previous launch.
  float* pmin; float* pmax; int* psum;
  uint8_t* codes; __half* scale; __half* zero; int* sums;
  int do_gemv; long long prefetch_bytes;
  unsigned long long* counters;  // [4]
  long long* trace;              // optional [gridDim.x, 8] phase timestamps (debug)
  // Next linear in execution order (nullable): prefetched into L2.
  const void* next_r1t; const void* next_r2; const void* next_perm; int next_k;
  const void* next_w; long long next_w_bytes;
  int own;       // 128-blocks per worker CTA (set by the host)
  int alias_r2;  // R2^T shares the R1^T shared-memory slot
};

__device__ __forceinline__ long long gtimer() {
  long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
#define DUQ_MARK(i) \
  if (p.trace && threadIdx.x == 0) p.trace[blockIdx.x * 8 + (i)] = gtimer();

constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kPre = 4;  // 16-byte weight vectors per lane and row held in registers
constexpr int kPad = 132;  // padded row stride (halves) of R1^T and R2^T
constexpr int kGs = 136;   // padded row stride (halves) of g
constexpr int kXb = 136;   // padded 128-block stride (halves) of x (spreads banks)

__device__ __forceinline__ unsigned long long ld_acquire(const unsigned long long* p) {
  unsigned long long v;
  asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void red_release(unsigned long long* c) {
  asm volatile("red.release.gpu.global.add.u64 [%0], 1;" ::"l"(c) : "memory");
}
__device__ __forceinline__ unsigned long long evict_first_policy() {
  unsigned long long pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ uint4 ldg_stream_ef(const uint4* ptr, unsigned long long pol) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(ptr), "l"(pol));
  return r;
}
// Grid-strided L2 prefetch (evict_last) of a byte range, one request per 128 B line.
__device__ __forceinline__ void prefetch_range_last(const void* base, size_t bytes) {
  if (!base) return;
  const char* b = reinterpret_cast<const char*>(base);
  for (size_t off = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 128; off < bytes;
       off += size_t(gridDim.x) * blockDim.x * 128)
    asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(b + off));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(
                   static_cast<unsigned>(__cvta_generic_to_shared(dst))), "l"(src));
}
__device__ __forceinline__ void cp_async8(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(
                   static_cast<unsigned>(__cvta_generic_to_shared(dst))), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

// L2 prefetch of this CTA's weight row groups (budget split over the grid).
__device__ __forceinline__ void prefetch_weights(const uint8_t* w, int n, size_t row_bytes,
                                                 int group_rows, int groups, long long budget) {
  const long long per_cta = budget / gridDim.x;
  long long done = 0;
  for (int g = blockIdx.x; g < groups && done < per_cta; g += gridDim.x) {
    const int rows = min(group_rows, n - g * group_rows);
    const char* base = reinterpret_cast<const char*>(w) + size_t(g) * group_rows * row_bytes;
    const size_t bytes = size_t(rows) * row_bytes;
    for (size_t off = size_t(threadIdx.x) * 128; off < bytes; off += size_t(blockDim.x) * 128)
      asm volatile("prefetch.global.L2 [%0];" ::"l"(base + off));
    done += bytes;
  }
}

// nearbyint(v / s) with the quotient correctly rounded, as tl.div_rn + nearbyint.
// v * (1/s) is within a few ulp of the exact quotient; it can only round to a
// different integer when the quotient is within that distance of a half-way
// point, so only those (rare) values take the exact IEEE division.
__device__ __forceinline__ float round_div(float v, float s, float inv) {
  float q = v * inv;
  const float frac = q - floorf(q);
  if (fabsf(frac - 0.5f) < 1.0e-3f) q = __fdiv_rn(v, s);
  return nearbyintf(q);
}

constexpr int kMaxOwn = 4;  // max 128-blocks per worker CTA

template <int MR>
__host__ __device__ constexpr size_t act_smem(int k, int own = 1, bool alias = false) {
  return size_t((alias ? 1 : 2) * 128 * kPad) * 2  // padded R1^T, R2^T
         + size_t(MR) * (k / 128) * kXb * 2      // x rows (padded blocks)
         + size_t(own) * MR * kGs * 2            // g per owned block
         + size_t(own) * MR * 128 * 2            // z per owned block
         + size_t(own) * 128 * 4;                // perm of each owned block
}

// D = A(16x16, row) * B(16x8, col) + D, FP16 inputs, FP32 accumulate.
__device__ __forceinline__ void mma16816(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%0,%1,%2,%3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int MR>
__host__ __device__ constexpr size_t gemv_smem(int k) {
  return size_t(MR) * k / 2;  // packed codes
}

template <int MR, int RPW>
__global__ void __launch_bounds__(kThreads) linear_small(SmallParams p) {
  extern __shared__ __align__(16) unsigned char smem[];
  __shared__ float s_scale[MR], s_zp[MR], s_inv[MR];
  __shared__ int s_ax[MR];
  __shared__ float s_lo[kWarps * MR], s_hi[kWarps * MR];
  __shared__ int s_sum[kWarps * MR];
  __shared__ unsigned long long s_epoch;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int m = p.m, k = p.k, n = p.n, nb = k / 128;
  const int group_rows = RPW * kWarps, groups = (n + group_rows - 1) / group_rows;
  const size_t row_bytes = size_t(k) / 2;
  const int kb = k / 32;  // 16-byte vectors per packed row
  const int b = blockIdx.x;
  // W worker CTAs; worker b owns blocks b, b + W, ... (p.own of them at most).
  const int nbw = min(nb, int(gridDim.x));
  const bool worker = b < nbw;  // host guarantees own * W >= nb
  const int own = worker ? (nb - 1 - b) / nbw + 1 : 0;
  const bool alias = p.alias_r2 != 0;

  DUQ_MARK(0)
  if (tid == 0) s_epoch = atomicAdd(p.counters, 1ull) / gridDim.x;

  // Activation-transform scratch (workers only).
  __half* sR1t = reinterpret_cast<__half*>(smem);
  __half* sR2t = alias ? sR1t : sR1t + 128 * kPad;
  __half* sx = sR1t + (alias ? 1 : 2) * 128 * kPad;
  __half* sg = sx + size_t(MR) * nb * kXb;   // [own][MR][kGs]
  __half* sz = sg + p.own * MR * kGs;        // [own][MR][128]
  int* sperm = reinterpret_cast<int*>(sz + p.own * MR * 128);  // [own][128]
  if (worker) {
    // r1t / r2t arrive transposed and padded to [128][kPad] from the host.
    for (int i = tid; i < 128 * kPad / 8; i += kThreads) cp_async16(sR1t + i * 8, p.r1t + i * 8);
    for (int v = tid; v < m * k / 8; v += kThreads) {
      const int r = v / (k / 8), e = (v % (k / 8)) * 8;
      cp_async16(sx + (size_t(r) * nb + (e >> 7)) * kXb + (e & 127), p.x + size_t(v) * 8);
    }
    for (int i = tid; i < own * 32; i += kThreads)
      cp_async16(sperm + i * 4, p.perm + (b + (i >> 5) * nbw) * 128 + (i & 31) * 4);
    cp_async_commit();
    if (!alias)
      for (int i = tid; i < 128 * kPad / 8; i += kThreads) cp_async16(sR2t + i * 8, p.r2t + i * 8);
    cp_async_commit();
  }

  // Weights: first kPre vectors per lane and row into registers, more into L2.
  const unsigned long long ef = evict_first_policy();
  const int g0 = b, row0 = g0 * group_rows + warp * RPW;
  uint4 pre[kPre][RPW];
#pragma unroll
  for (int t = 0; t < kPre; ++t)
#pragma unroll
    for (int r = 0; r < RPW; ++r) {
      const int jj = lane + t * 32;
      pre[t][r] = (p.do_gemv && g0 < groups && row0 + r < n && jj < kb)
                      ? ldg_stream_ef(reinterpret_cast<const uint4*>(p.w) + size_t(row0 + r) * kb + jj, ef)
                      : make_uint4(0, 0, 0, 0);
    }
  if (p.do_gemv && !worker && p.prefetch_bytes > 0)
    prefetch_weights(p.w, n, row_bytes, group_rows, groups, p.prefetch_bytes);
  if (p.do_gemv) {
    prefetch_range_last(p.next_r1t, size_t(128) * kPad * 2);
    prefetch_range_last(p.next_r2, size_t(128) * kPad * 2);
    prefetch_range_last(p.next_perm, size_t(p.next_k) * 4);
    prefetch_range_last(p.next_w, size_t(p.next_w_bytes));
  }

  if (worker) {
    cp_async_wait<1>();
    __syncthreads();
    DUQ_MARK(1)
    // A: gathered first-rotation outputs g[r][i] = (x @ blockdiag(R1))[r][perm[128b+i]]
    // on Tensor Cores (m16n8k16, FP32 accumulate, k ascending): the same
    // rounding as tl.dot / cuBLAS in the reference path. One mma covers 8
    // consecutive outputs i (n = lane>>2); its 16 rows are (source block of
    // output g) x (token 0/1) so every lane loads its own source block, and only
    // the diagonal (row source == column) entries are kept. Tokens 2-3 use a
    // second mma with the same B operand.
    for (int o = 0; o < own; ++o) {
      const int g = lane >> 2, t = lane & 3;
      const int* operm = sperm + o * 128;
      __half* og = sg + o * MR * kGs;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int i = (warp + half * kWarps) * 8 + g;
        const int c = operm[i], src = c >> 7, col = c & 127;
        const __half* x0 = sx + (size_t(0) * nb + src) * kXb;
        const __half* x1 = sx + (size_t(1) * nb + src) * kXb;
        const __half* x2 = sx + (size_t(2) * nb + src) * kXb;
        const __half* x3 = sx + (size_t(3) * nb + src) * kXb;
        const __half* rr = sR1t + col * kPad;
        float acc[4] = {0.f, 0.f, 0.f, 0.f}, acc2[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
          const int kx = kk * 16 + 2 * t;
          const uint32_t b0 = *reinterpret_cast<const uint32_t*>(rr + kx);
          const uint32_t b1 = *reinterpret_cast<const uint32_t*>(rr + kx + 8);
          const uint32_t a0 = m > 0 ? *reinterpret_cast<const uint32_t*>(x0 + kx) : 0u;
          const uint32_t a1 = m > 1 ? *reinterpret_cast<const uint32_t*>(x1 + kx) : 0u;
          const uint32_t a2 = m > 0 ? *reinterpret_cast<const uint32_t*>(x0 + kx + 8) : 0u;
          const uint32_t a3 = m > 1 ? *reinterpret_cast<const uint32_t*>(x1 + kx + 8) : 0u;
          mma16816(acc, a0, a1, a2, a3, b0, b1);
          if (MR > 2) {
            const uint32_t e0 = m > 2 ? *reinterpret_cast<const uint32_t*>(x2 + kx) : 0u;
            const uint32_t e1 = m > 3 ? *reinterpret_cast<const uint32_t*>(x3 + kx) : 0u;
            const uint32_t e2 = m > 2 ? *reinterpret_cast<const uint32_t*>(x2 + kx + 8) : 0u;
            const uint32_t e3 = m > 3 ? *reinterpret_cast<const uint32_t*>(x3 + kx + 8) : 0u;
            mma16816(acc2, e0, e1, e2, e3, b0, b1);
          }
        }
        // Lane holds rows g (token 0/2) and g+8 (token 1/3) at columns 2t, 2t+1;
        // the diagonal entry for output i (column g) sits in lanes with g>>1 == t.
        if ((g >> 1) == t) {
          const int e = g & 1;
          if (m > 0) og[0 * kGs + i] = __float2half_rn(acc[e]);
          if (m > 1) og[1 * kGs + i] = __float2half_rn(acc[2 + e]);
          if (MR > 2 && m > 2) og[2 * kGs + i] = __float2half_rn(acc2[e]);
          if (MR > 3 && m > 3) og[3 * kGs + i] = __float2half_rn(acc2[2 + e]);
        }
      }
    }
    __syncthreads();
    if (alias) {
      // Every owned block is past rotation A: R2^T may now replace R1^T.
      for (int i = tid; i < 128 * kPad / 8; i += kThreads) cp_async16(sR2t + i * 8, p.r2t + i * 8);
      cp_async_commit();
    }
    cp_async_wait<0>();
    __syncthreads();
    DUQ_MARK(2)
    // B: z = g @ R2 on Tensor Cores; warp w computes columns 8w..8w+7 and
    // 64+8w..64+8w+7. B operand from R2^T (rows = output columns).
    for (int o = 0; o < own; ++o) {
      const int g = lane >> 2, t = lane & 3;
      const __half* og = sg + o * MR * kGs;
      __half* oz = sz + o * MR * 128;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int nt = warp + half * kWarps;
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
          const int kx = kk * 16 + 2 * t;
          const uint32_t a0 = g < m ? *reinterpret_cast<const uint32_t*>(og + g * kGs + kx) : 0u;
          const uint32_t a2 = g < m ? *reinterpret_cast<const uint32_t*>(og + g * kGs + kx + 8) : 0u;
          const uint32_t b0 = *reinterpret_cast<const uint32_t*>(sR2t + (nt * 8 + g) * kPad + kx);
          const uint32_t b1 = *reinterpret_cast<const uint32_t*>(sR2t + (nt * 8 + g) * kPad + kx + 8);
          mma16816(acc, a0, 0u, a2, 0u, b0, b1);
        }
        if (g < m) {
          oz[g * 128 + nt * 8 + 2 * t] = __float2half_rn(acc[0]);
          oz[g * 128 + nt * 8 + 2 * t + 1] = __float2half_rn(acc[1]);
        }
      }
    }
    __syncthreads();
    // Per-row min/max of z for every owned block; a warp covers 32 columns of one row.
    for (int o = 0; o < own; ++o) {
      const __half* oz = sz + o * MR * 128;
#pragma unroll
      for (int base = 0; base < MR * 128; base += kThreads) {
        const int idx = base + tid, r = idx >> 7, j = idx & 127;
        float lo = INFINITY, hi = -INFINITY;
        if (r < m) lo = hi = __half2float(oz[r * 128 + j]);
#pragma unroll
        for (int q = 16; q > 0; q >>= 1) {
          lo = fminf(lo, __shfl_xor_sync(0xffffffffu, lo, q));
          hi = fmaxf(hi, __shfl_xor_sync(0xffffffffu, hi, q));
        }
        if (lane == 0 && r < m) {
          s_lo[idx >> 5] = lo;
          s_hi[idx >> 5] = hi;
        }
      }
      __syncthreads();
      if (tid < m) {
        float lo = s_lo[tid * 4], hi = s_hi[tid * 4];
        for (int w = 1; w < 4; ++w) {
          lo = fminf(lo, s_lo[tid * 4 + w]);
          hi = fmaxf(hi, s_hi[tid * 4 + w]);
        }
        const int blk = b + o * nbw;
        p.pmin[tid * nb + blk] = lo;
        p.pmax[tid * nb + blk] = hi;
      }
      __syncthreads();
    }
    DUQ_MARK(3)
    if (tid == 0) red_release(p.counters + 1);
    if (p.do_gemv && p.prefetch_bytes > 0)
      prefetch_weights(p.w, n, row_bytes, group_rows, groups, p.prefetch_bytes);
    if (tid == 0) {
      const unsigned long long target = (s_epoch + 1) * nbw;
      while (ld_acquire(p.counters + 1) < target) {
      }
    }
    __syncthreads();
    DUQ_MARK(4)

    // Row scale/zero (identical in every worker; CTA 0 publishes).
    if (warp < m) {
      float lo = INFINITY, hi = -INFINITY;
      for (int q = lane; q < nb; q += 32) {
        lo = fminf(lo, __ldcg(p.pmin + warp * nb + q));
        hi = fmaxf(hi, __ldcg(p.pmax + warp * nb + q));
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        lo = fminf(lo, __shfl_xor_sync(0xffffffffu, lo, o));
        hi = fmaxf(hi, __shfl_xor_sync(0xffffffffu, hi, o));
      }
      lo = f16r(lo * p.clip);
      hi = f16r(hi * p.clip);
      const float span = f16r(hi - lo);
      const float scale = f16r(fminf(fmaxf(f16r(div_full(span, 15.f)), 1.0e-5f), 1.0e4f));
      const float zp = nearbyintf(fminf(fmaxf(f16r(__fdiv_rn(-lo, scale)), -1.0e4f), 1.0e4f));
      if (lane == 0) {
        s_scale[warp] = scale;
        s_zp[warp] = zp;
        s_inv[warp] = __frcp_rn(scale);
        if (b == 0) {
          p.scale[warp] = __float2half_rn(scale);
          p.zero[warp] = __float2half_rn(zp);
        }
      }
    }
    __syncthreads();
    // Quantize/pack every owned block: thread -> (row, packed byte); 64 bytes per row.
    for (int o = 0; o < own; ++o) {
      const __half* oz = sz + o * MR * 128;
      const int blk = b + o * nbw;
      for (int idx = tid; idx < MR * 64; idx += kThreads) {
        const int r = idx >> 6, pb = idx & 63;
        int sum = 0;
        if (r < m) {
          int c[2];
#pragma unroll
          for (int e = 0; e < 2; ++e) {
            float q = f16r(round_div(__half2float(oz[r * 128 + 2 * pb + e]), s_scale[r], s_inv[r]));
            q = fminf(fmaxf(f16r(q + s_zp[r]), 0.f), 15.f);
            c[e] = int(q);
          }
          p.codes[size_t(r) * (k / 2) + blk * 64 + pb] = uint8_t(c[0] | (c[1] << 4));
          sum = c[0] + c[1];
        }
#pragma unroll
        for (int q = 16; q > 0; q >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, q);
        if (lane == 0) s_sum[idx >> 5] = sum;
      }
      __syncthreads();
      if (tid < m) p.psum[tid * nb + blk] = s_sum[2 * tid] + s_sum[2 * tid + 1];
      __syncthreads();
    }
    DUQ_MARK(5)
    if (tid == 0) red_release(p.counters + 2);
  }
  if (!p.do_gemv && b != 0) return;

  // Barrier 2: packed codes and per-block code sums of every block published.
  if (tid == 0) {
    const unsigned long long target = (s_epoch + 1) * nbw;
    while (ld_acquire(p.counters + 2) < target) {
    }
  }
  __syncthreads();
  DUQ_MARK(6)
  uint4* sa = reinterpret_cast<uint4*>(smem);
  if (p.do_gemv)
    for (int i = tid; i < m * kb; i += kThreads) sa[i] = __ldcg(reinterpret_cast<const uint4*>(p.codes) + i);
  if (warp < m) {
    int sum = 0;
    for (int q = lane; q < nb; q += 32) sum += __ldcg(p.psum + warp * nb + q);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) {
      s_ax[warp] = sum;
      if (b == 0) p.sums[warp] = sum;
      if (!worker) {
        s_scale[warp] = __half2float(__ldcg(p.scale + warp));
        s_zp[warp] = __half2float(__ldcg(p.zero + warp));
      }
    }
  }
  __syncthreads();
  if (!p.do_gemv) return;

  // Persistent u4 GEMV; the first group starts from the register-held vectors.
  for (int g = g0; g < groups; g += gridDim.x) {
    const int rw = g * group_rows + warp * RPW;
    if (rw >= n) continue;
    const uint4* wr = reinterpret_cast<const uint4*>(p.w) + size_t(rw) * kb;
    int acc[RPW][MR];
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
      for (int i = 0; i < MR; ++i) acc[r][i] = 0;
    int start = lane;
    if (g == g0) {
#pragma unroll
      for (int t = 0; t < kPre; ++t) {
        const int jj = lane + t * 32;
        if (jj < kb)
#pragma unroll
          for (int i = 0; i < MR; ++i)
            if (i < m) {
              const uint4 av = sa[i * kb + jj];
#pragma unroll
              for (int r = 0; r < RPW; ++r) acc[r][i] = dot_u4x32(pre[t][r], av, acc[r][i]);
            }
      }
      start = lane + kPre * 32;
    }
#pragma unroll 2
    for (int jj = start; jj < kb; jj += 32) {
      uint4 wv[RPW];
#pragma unroll
      for (int r = 0; r < RPW; ++r)
        wv[r] = rw + r < n ? ldg_stream_ef(wr + size_t(r) * kb + jj, ef) : make_uint4(0, 0, 0, 0);
#pragma unroll
      for (int i = 0; i < MR; ++i)
        if (i < m) {
          const uint4 av = sa[i * kb + jj];
#pragma unroll
          for (int r = 0; r < RPW; ++r) acc[r][i] = dot_u4x32(wv[r], av, acc[r][i]);
        }
    }
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
      for (int i = 0; i < MR; ++i)
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) acc[r][i] += __shfl_xor_sync(0xffffffffu, acc[r][i], o);
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
      for (int i = 0; i < MR; ++i)
        if (lane == (r * MR + i) % 32 && i < m && rw + r < n) {
          const int col = rw + r;
          const float out = (float(acc[r][i]) + s_zp[i] * float(p.wc[col]) -
                             __half2float(p.wz[col]) * float(s_ax[i])) *
                                s_scale[i] * __half2float(p.ws[col]) +
                            __half2float(p.bias[col]);
          p.y[size_t(i) * n + col] = __float2half_rn(out);
        }
  }
  __syncthreads();
  DUQ_MARK(7)
}

template <int MR, int RPW>
int launch_linear_small(SmallParams& p, cudaStream_t stream, int* info = nullptr) {
  auto fn = linear_small<MR, RPW>;
  // Host-side caches: device limits once, then the chosen layout per K.
  static int sms = 0, max_dynamic = 0;
  static int cached_k[8] = {}, cached_own[8] = {}, cached_alias[8] = {}, cached_bps[8] = {};
  static size_t cached_smem[8] = {};
  if (sms == 0) {
    int dev, optin = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaFuncAttributes attr;
    auto err = cudaFuncGetAttributes(&attr, fn);
    if (err != cudaSuccess) return -int(err);
    max_dynamic = optin - int(attr.sharedSizeBytes);
    err = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic);
    if (err != cudaSuccess) return -int(err);
  }
  const int nb = p.k / 128;
  int slot = -1;
  for (int i = 0; i < 8; ++i)
    if (cached_k[i] == p.k) slot = i;
  if (slot < 0) {
    // First layout (fewest owned blocks, then no aliasing) in which every
    // 128-block has a resident worker.
    int own = 0, alias = 0, bps = 0;
    size_t smem = 0;
    for (int o = 1; o <= kMaxOwn && !own; ++o)
      for (int a = 0; a < 2 && !own; ++a) {
        const size_t s = max(act_smem<MR>(p.k, o, a), gemv_smem<MR>(p.k));
        if (s > size_t(max_dynamic)) continue;
        int b = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b, fn, kThreads, s) != cudaSuccess || b < 1)
          continue;
        if (size_t(o) * min(nb, sms * b) >= size_t(nb)) own = o, alias = a, bps = b, smem = s;
      }
    if (!own) return 6666;  // no layout gives every 128-block a resident worker
    for (int i = 0; i < 8 && slot < 0; ++i)
      if (cached_k[i] == 0) slot = i;
    if (slot < 0) slot = 7;
    cached_k[slot] = p.k, cached_own[slot] = own, cached_alias[slot] = alias;
    cached_bps[slot] = bps, cached_smem[slot] = smem;
  }
  p.own = cached_own[slot];
  p.alias_r2 = cached_alias[slot];
  const size_t smem = cached_smem[slot];
  const int cap = sms * cached_bps[slot];
  const int groups = (p.n + RPW * kWarps - 1) / (RPW * kWarps);
  const int grid = p.do_gemv ? min(cap, max(groups, nb)) : min(cap, nb);
  if (info) {  // dry run: report the plan without launching
    info[0] = p.own, info[1] = p.alias_r2, info[2] = grid, info[3] = int(smem);
    return 0;
  }
  void* args[] = {&p};
  return -int(cudaLaunchCooperativeKernel((void*)fn, dim3(grid), dim3(kThreads), args, smem, stream));
}

template <int MR>
int linear_small_rows(int config, SmallParams& p, cudaStream_t st, int* info = nullptr) {
  if (config == 0) return launch_linear_small<MR, 1>(p, st, info);
  if (config == 1) return launch_linear_small<MR, 2>(p, st, info);
  return launch_linear_small<MR, 4>(p, st, info);
}

}  // namespace duq

extern "C" int duquant_linear_small(
    const void* x, const void* r1t, const void* r2t, const void* perm, const void* w, void* y,
    int m, int n, int k, float clip, void* stream, int config, const void* ws, const void* wz,
    const void* wc, const void* bias, void* pmin, void* pmax, void* psum, void* codes, void* scale,
    void* zero, void* sums, int do_gemv, long long prefetch_bytes, void* counters, void* trace,
    const void* next_r1t, const void* next_r2, const void* next_perm, int next_k,
    const void* next_w, long long next_w_bytes) {
  duq::SmallParams p{(const __half*)x, (const __half*)r1t, (const __half*)r2t, (const int*)perm,
                     (const uint8_t*)w, (__half*)y, m, n, k, clip, (const __half*)ws,
                     (const __half*)wz, (const int*)wc, (const __half*)bias,
                     (float*)pmin, (float*)pmax, (int*)psum, (uint8_t*)codes, (__half*)scale, (__half*)zero,
                     (int*)sums, do_gemv, prefetch_bytes, (unsigned long long*)counters,
                     (long long*)trace, next_r1t, next_r2, next_perm, next_k, next_w,
                     next_w_bytes, 1, 0};
  auto st = (cudaStream_t)stream;
  if (m <= 1) return duq::linear_small_rows<1>(config, p, st);
  if (m <= 2) return duq::linear_small_rows<2>(config, p, st);
  if (m <= 4) return duq::linear_small_rows<4>(config, p, st);
  return 9999;
}

// Dry run of duquant_linear_small: 0 if the layer can use the cooperative
// kernel on the current device; info = {own, alias_r2, grid, smem bytes}.
extern "C" int duquant_linear_small_plan(int m, int n, int k, int config, int do_gemv, int* info) {
  duq::SmallParams p{};
  p.m = m, p.n = n, p.k = k, p.do_gemv = do_gemv;
  if (m <= 1) return duq::linear_small_rows<1>(config, p, nullptr, info);
  if (m <= 2) return duq::linear_small_rows<2>(config, p, nullptr, info);
  if (m <= 4) return duq::linear_small_rows<4>(config, p, nullptr, info);
  return 9999;
}
