// Hopper (sm_90a) DuQuant W4A4 GEMM.
//
// Hopper has no INT4 tensor-core MMA, so weights stay packed as unsigned
// nibbles in global/shared memory and are widened to s8 in registers right
// before an INT8 wgmma (CUTLASS mixed-input, register-sourced A operand).
// Activations are produced directly as s8 codes in [0,15] by the quantizer.
// The integer accumulator is therefore bit-identical to the Ada u4 x u4 path.
//
// wgmma takes its register operand as A, so the problem is swapped:
//   D'[N,M] = W[N,K] * X[M,K]^T,  D' column-major == Y[M,N] row-major.
// In the swapped frame, per-weight-channel vectors are column broadcasts
// (along M') and per-token vectors are row broadcasts (along N').
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>

namespace cutlass {
// u4 -> s8 widening (values 0..15). Element i sits in nibble i (low first),
// matching duquant_cuda.backend.pack and cutlass::uint4b_t arrays.
template <int N, FloatRoundStyle Round>
struct NumericArrayConverter<int8_t, uint4b_t, N, Round> {
  using result_type = Array<int8_t, N>;
  using source_type = Array<uint4b_t, N>;
  static_assert(N % 4 == 0, "u4->s8 conversion expects multiples of 4");

  CUTLASS_HOST_DEVICE static result_type convert(source_type const& s) {
    result_type out;
#if defined(__CUDA_ARCH__)
    if constexpr (N % 8 == 0) {
      uint32_t const* in = reinterpret_cast<uint32_t const*>(&s);
      uint32_t* o = reinterpret_cast<uint32_t*>(&out);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < N / 8; ++i) {
        uint32_t lo = in[i] & 0x0F0F0F0Fu;         // e0 e2 e4 e6
        uint32_t hi = (in[i] >> 4) & 0x0F0F0F0Fu;  // e1 e3 e5 e7
        o[2 * i] = __byte_perm(lo, hi, 0x5140);
        o[2 * i + 1] = __byte_perm(lo, hi, 0x7362);
      }
    } else {
      uint16_t const* in = reinterpret_cast<uint16_t const*>(&s);
      uint32_t* o = reinterpret_cast<uint32_t*>(&out);
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < N / 4; ++i) {
        uint32_t r = in[i];
        o[i] = __byte_perm(r & 0x0F0Fu, (r >> 4) & 0x0F0Fu, 0x5140);
      }
    }
#else
    for (int i = 0; i < N; ++i) out[i] = int8_t(int(s[i]));
#endif
    return out;
  }
  CUTLASS_HOST_DEVICE result_type operator()(source_type const& s) const {
    return convert(s);
  }
};
}  // namespace cutlass

#include <cute/tensor.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/packed_stride.hpp>

using namespace cute;

namespace {

template <class T>
struct AffineEpilogue {
  // Same expression and operation order as the Ada fused epilogue.
  CUTLASS_HOST_DEVICE T operator()(T const& acc, T const& as, T const& az, T const& ax,
                                   T const& ws, T const& wz, T const& wc,
                                   T const& bias) const {
    T out;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < T::kElements; i++)
      out[i] = (acc[i] + az[i] * wc[i] - wz[i] * ax[i]) * as[i] * ws[i] + bias[i];
    return out;
  }
};

using H = cutlass::half_t;
using ElementW = cutlass::uint4b_t;  // swapped A: weights [N,K], K-major
using ElementX = int8_t;             // swapped B: activations [M,K], K-major
using LayoutW = cutlass::layout::RowMajor;
using LayoutX = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::ColumnMajor;  // D'[N,M] col-major == Y[M,N]
constexpr int AlignW = 32;
constexpr int AlignX = 16;

int sm_count() {
  static int count = [] {
    int dev = 0;
    cudaGetDevice(&dev);
    return cutlass::KernelHardwareInfo::query_device_multiprocessor_count(dev);
  }();
  return count;
}

// Sched: void = data-parallel persistent tiles; cutlass::gemm::StreamKScheduler
// splits K across CTAs when there are too few output tiles to fill the GPU
// (N/128 x M/tile tiles is only 32-96 for decode/short-prefill sizes).
template <class Tile, bool Fused, class Sched = void>
struct Config {
  using Cluster = Shape<_1, _1, _1>;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
  using EpiSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

  using RowH = cutlass::epilogue::fusion::Sm90RowBroadcast<0, Tile, H, float>;
  using RowI = cutlass::epilogue::fusion::Sm90RowBroadcast<0, Tile, int32_t, float>;
  using ColH = cutlass::epilogue::fusion::Sm90ColBroadcast<0, Tile, H, float>;
  using ColI = cutlass::epilogue::fusion::Sm90ColBroadcast<0, Tile, int32_t, float>;
  using Compute = cutlass::epilogue::fusion::Sm90Compute<
      AffineEpilogue, H, float, cutlass::FloatRoundStyle::round_to_nearest>;
  using AffineEVT = cutlass::epilogue::fusion::Sm90EVT<
      Compute, cutlass::epilogue::fusion::Sm90AccFetch,
      RowH, RowH, RowI, ColH, ColH, ColI, ColH>;
  using Plain = cutlass::epilogue::fusion::LinearCombination<int32_t, int32_t, void, int32_t>;

  using ElementD = cute::conditional_t<Fused, H, int32_t>;
  using ElementEpi = cute::conditional_t<Fused, float, int32_t>;
  using Fusion = cute::conditional_t<Fused, AffineEVT, Plain>;

  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, int32_t, ElementEpi,
      void, LayoutD, 128 / cutlass::sizeof_bits<ElementD>::value,
      ElementD, LayoutD, 128 / cutlass::sizeof_bits<ElementD>::value,
      EpiSchedule, Fusion>::CollectiveOp;

  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
      ElementW, LayoutW, AlignW, ElementX, LayoutX, AlignX, int32_t, Tile, Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using Kernel =
      cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue, Sched>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

// One persistent workspace per instantiation. Constructed during warmup,
// before any CUDA graph capture, so no allocation happens while capturing.
template <class Gemm>
void* workspace(size_t bytes) {
  static void* ptr = nullptr;
  static size_t size = 0;
  if (bytes > size) {
    if (ptr) cudaFree(ptr);
    if (cudaMalloc(&ptr, bytes) != cudaSuccess) return nullptr;
    size = bytes;
  }
  return ptr;
}

template <class Gemm>
int launch(typename Gemm::Arguments& args, cudaStream_t stream) {
  Gemm op;
  auto status = op.can_implement(args);
  if (status != cutlass::Status::kSuccess) return 1000 + int(status);
  size_t bytes = Gemm::get_workspace_size(args);
  void* ws = bytes ? workspace<Gemm>(bytes) : nullptr;
  if (bytes && !ws) return 2000;
  status = op.initialize(args, ws, stream);
  if (status != cutlass::Status::kSuccess) return 3000 + int(status);
  status = op.run(stream);
  if (status != cutlass::Status::kSuccess) return 4000 + int(status);
  return -int(cudaGetLastError());
}

template <class C>
auto mainloop_args(const void* x, const void* w, int m, int n, int k) {
  using StrideW = typename C::Kernel::StrideA;
  using StrideX = typename C::Kernel::StrideB;
  return typename C::Mainloop::Arguments{
      static_cast<const ElementW*>(w), cutlass::make_cute_packed_stride(StrideW{}, {n, k, 1}),
      static_cast<const ElementX*>(x), cutlass::make_cute_packed_stride(StrideX{}, {m, k, 1})};
}

template <class Tile, class Sched = void>
int run_fused(const void* x, const void* w, void* y, int m, int n, int k, cudaStream_t stream,
              const void* as, const void* az, const void* ax, const void* ws,
              const void* wz, const void* wc, const void* bias) {
  using C = Config<Tile, true, Sched>;
  using StrideD = typename C::Kernel::StrideD;
  auto d = cutlass::make_cute_packed_stride(StrideD{}, {n, m, 1});
  typename C::AffineEVT::Arguments fusion{
      {},                                                       // acc
      {static_cast<const H*>(as), H(0), {}},       // per token
      {static_cast<const H*>(az), H(0), {}},
      {static_cast<const int32_t*>(ax), 0, {}},
      {static_cast<const H*>(ws), H(0), {}},       // per channel
      {static_cast<const H*>(wz), H(0), {}},
      {static_cast<const int32_t*>(wc), 0, {}},
      {static_cast<const H*>(bias), H(0), {}},
      {}};
  cutlass::KernelHardwareInfo hw{0, sm_count()};
  typename C::Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {n, m, k, 1},
      mainloop_args<C>(x, w, m, n, k),
      {fusion, nullptr, d, static_cast<H*>(y), d}, hw};
  return launch<typename C::Gemm>(args, stream);
}

template <class Tile, class Sched = void>
int run_plain(const void* x, const void* w, void* c, int m, int n, int k, cudaStream_t stream) {
  using C = Config<Tile, false, Sched>;
  using StrideD = typename C::Kernel::StrideD;
  auto d = cutlass::make_cute_packed_stride(StrideD{}, {n, m, 1});
  cutlass::KernelHardwareInfo hw{0, sm_count()};
  typename C::Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm, {n, m, k, 1},
      mainloop_args<C>(x, w, m, n, k),
      {{1, 0}, nullptr, d, static_cast<int32_t*>(c), d}, hw};
  return launch<typename C::Gemm>(args, stream);
}

// Token-dimension tile (swapped N') chosen by the caller from the shape only.
// Configs 0-4: data-parallel 16/32/64/128/256; 5-8: Stream-K 16/32/64/128.
using T16 = Shape<_128, _16, _128>;
using T32 = Shape<_128, _32, _128>;
using T64 = Shape<_128, _64, _128>;
using T128 = Shape<_128, _128, _128>;
using T256 = Shape<_128, _256, _128>;
using SK = cutlass::gemm::StreamKScheduler;

}  // namespace

// x: s8 activation codes [M,K]; w: packed u4 weights [N,K/2]; c: int32 [M,N].
extern "C" int duquant_sm90_gemm(const void* x, const void* w, void* c, int m, int n, int k,
                                 void* stream, int config) {
  auto s = (cudaStream_t)stream;
  switch (config) {
    case 0: return run_plain<T16>(x, w, c, m, n, k, s);
    case 1: return run_plain<T32>(x, w, c, m, n, k, s);
    case 2: return run_plain<T64>(x, w, c, m, n, k, s);
    case 3: return run_plain<T128>(x, w, c, m, n, k, s);
    case 4: return run_plain<T256>(x, w, c, m, n, k, s);
    case 5: return run_plain<T16, SK>(x, w, c, m, n, k, s);
    case 6: return run_plain<T32, SK>(x, w, c, m, n, k, s);
    case 7: return run_plain<T64, SK>(x, w, c, m, n, k, s);
    default: return run_plain<T128, SK>(x, w, c, m, n, k, s);
  }
}

extern "C" int duquant_sm90_gemm_fused(const void* x, const void* w, void* y, int m, int n,
                                       int k, void* stream, int config, const void* as,
                                       const void* az, const void* ax, const void* ws,
                                       const void* wz, const void* wc, const void* bias) {
  auto s = (cudaStream_t)stream;
  switch (config) {
    case 0: return run_fused<T16>(x, w, y, m, n, k, s, as, az, ax, ws, wz, wc, bias);
    case 1: return run_fused<T32>(x, w, y, m, n, k, s, as, az, ax, ws, wz, wc, bias);
    case 2: return run_fused<T64>(x, w, y, m, n, k, s, as, az, ax, ws, wz, wc, bias);
    case 3: return run_fused<T128>(x, w, y, m, n, k, s, as, az, ax, ws, wz, wc, bias);
    case 4: return run_fused<T256>(x, w, y, m, n, k, s, as, az, ax, ws, wz, wc, bias);
    case 5: return run_fused<T16, SK>(x, w, y, m, n, k, s, as, az, ax, ws, wz, wc, bias);
    case 6: return run_fused<T32, SK>(x, w, y, m, n, k, s, as, az, ax, ws, wz, wc, bias);
    case 7: return run_fused<T64, SK>(x, w, y, m, n, k, s, as, az, ax, ws, wz, wc, bias);
    default: return run_fused<T128, SK>(x, w, y, m, n, k, s, as, az, ax, ws, wz, wc, bias);
  }
}

// Decode-size kernels (M <= 8): fused activation transform/quant and u4 GEMV.
#include "linear_small.cuh"
