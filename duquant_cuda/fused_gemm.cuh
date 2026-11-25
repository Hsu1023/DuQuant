#pragma once
#include <cutlass/device_kernel.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>

// This functor receives coordinate-aware row/column broadcasts from CUTLASS.
// wc = K * weight_zero - sum(weight_codes), precomputed at export.
template<class T> struct AffineEpilogue {
  CUTLASS_HOST_DEVICE T operator()(T const& acc,T const& as,T const& az,T const& ax,
                                   T const& ws,T const& wz,T const& wc,T const& bias) const {
    T out;
    CUTLASS_PRAGMA_UNROLL
    for(int i=0;i<T::kElements;i++)
      out[i]=(acc[i]+az[i]*wc[i]-wz[i]*ax[i])*as[i]*ws[i]+bias[i];
    return out;
  }
};

template<int BM,int BN,int BK,int WM,int WN>
int run_fused(const void* a,const void* b,void* y,int m,int n,int k,cudaStream_t stream,
              const void* as,const void* az,const void* ax,const void* ws,const void* wz,
              const void* wc,const void* bias){
  using namespace cute;
  using namespace cutlass::epilogue::threadblock;
  using H=cutlass::half_t;
  using TB=cutlass::gemm::GemmShape<BM,BN,BK>;
  using Warp=cutlass::gemm::GemmShape<WM,WN,BK>;
  using Map=OutputTileThreadLayout<TB,Warp,H,8,1>;
  using RowH=VisitorColBroadcast<Map,H,Stride<_1,_0,int32_t>>;
  using RowI=VisitorColBroadcast<Map,int32_t,Stride<_1,_0,int32_t>>;
  using ColH=VisitorRowBroadcast<Map,H,Stride<_0,_1,int32_t>>;
  using ColI=VisitorRowBroadcast<Map,int32_t,Stride<_0,_1,int32_t>>;
  using Compute=VisitorCompute<AffineEpilogue,H,float,cutlass::FloatRoundStyle::round_to_nearest>;
  using Expr=Sm80EVT<Compute,VisitorAccFetch,RowH,RowH,RowI,ColH,ColH,ColI,ColH>;
  using Store=VisitorAuxStore<Map,H,cutlass::FloatRoundStyle::round_to_nearest,Stride<int64_t,_1,int64_t>>;
  using EVT=Sm80EVT<Store,Expr>;
  using Generator=cutlass::gemm::kernel::DefaultGemmWithVisitor<
    cutlass::uint4b_t,cutlass::layout::RowMajor,cutlass::ComplexTransform::kNone,32,
    cutlass::uint4b_t,cutlass::layout::ColumnMajor,cutlass::ComplexTransform::kNone,32,
    H,cutlass::layout::RowMajor,8,int32_t,float,
    cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,TB,Warp,cutlass::gemm::GemmShape<16,8,64>,
    EVT,cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,
    cutlass::arch::OpMultiplyAddSaturate,1>;
  using Kernel=typename Generator::GemmKernel;

  typename Expr::Arguments expr=cute::make_tuple(
    cute::tuple<>{},
    typename RowH::Arguments{static_cast<const H*>(as),H(0),{_1{},_0{},m}},
    typename RowH::Arguments{static_cast<const H*>(az),H(0),{_1{},_0{},m}},
    typename RowI::Arguments{static_cast<const int32_t*>(ax),0,{_1{},_0{},m}},
    typename ColH::Arguments{static_cast<const H*>(ws),H(0),{_0{},_1{},n}},
    typename ColH::Arguments{static_cast<const H*>(wz),H(0),{_0{},_1{},n}},
    typename ColI::Arguments{static_cast<const int32_t*>(wc),0,{_0{},_1{},n}},
    typename ColH::Arguments{static_cast<const H*>(bias),H(0),{_0{},_1{},n}},
    cute::tuple<>{});
  typename EVT::Arguments callbacks{expr,
    typename Store::Arguments{static_cast<H*>(y),{n,_1{},int64_t(m)*n}}};
  typename Kernel::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm,{m,n,k},1,
    callbacks,a,b,nullptr,nullptr,int64_t(m)*k,int64_t(n)*k,0,0,k,k,0,0);
  // CUTLASS 3.4's non-StreamK visitor hides adapter metadata. Launch its
  // public kernel directly; identity swizzle does not use occupancy scheduling.
  typename Kernel::Params params(args,1,1);
  constexpr int smem=sizeof(typename Kernel::SharedStorage);
  if(smem>=48*1024){
    auto err=cudaFuncSetAttribute(cutlass::Kernel<Kernel>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
    if(err!=cudaSuccess)return -int(err);
  }
  cutlass::Kernel<Kernel><<<dim3((m+BM-1)/BM,(n+BN-1)/BN,1),Generator::GemmBase::kThreadCount,smem,stream>>>(params);
  return -int(cudaGetLastError());
}

extern "C" int duquant_gemm_fused(const void* a,const void* b,void* y,int m,int n,int k,
 void* stream,int config,const void* as,const void* az,const void* ax,const void* ws,
 const void* wz,const void* wc,const void* bias){
  if(config==1)return run_fused<64,128,128,32,64>(a,b,y,m,n,k,(cudaStream_t)stream,as,az,ax,ws,wz,wc,bias);
  if(config==2)return run_fused<128,128,128,64,64>(a,b,y,m,n,k,(cudaStream_t)stream,as,az,ax,ws,wz,wc,bias);
  return run_fused<32,128,128,32,64>(a,b,y,m,n,k,(cudaStream_t)stream,as,az,ax,ws,wz,wc,bias);
}
