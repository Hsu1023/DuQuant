#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>

// Both operands store unsigned nibbles; B is a row-major [N,K]
// buffer interpreted as column-major [K,N]. Zero points are applied separately.
template<int BM, int BN, int BK, int WM, int WN>
int run(const void* a, const void* b, void* c, int m, int n, int k, cudaStream_t stream) {
  using Gemm = cutlass::gemm::device::Gemm<
    cutlass::uint4b_t, cutlass::layout::RowMajor,
    cutlass::uint4b_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::RowMajor, int32_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<BM,BN,BK>,
    cutlass::gemm::GemmShape<WM,WN,BK>,
    cutlass::gemm::GemmShape<16,8,64>,
    cutlass::epilogue::thread::LinearCombination<int32_t,4,int32_t,int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3>;
  typename Gemm::Arguments args({m,n,k},
    {reinterpret_cast<const cutlass::uint4b_t*>(a),k},
    {reinterpret_cast<const cutlass::uint4b_t*>(b),k},
    {reinterpret_cast<int32_t*>(c),n}, {reinterpret_cast<int32_t*>(c),n}, {1,0});
  Gemm op;
  auto status=op.can_implement(args);
  if(status != cutlass::Status::kSuccess) return int(status);
  return int(op(args,nullptr,stream));
}
extern "C" int duquant_gemm(const void* a,const void* b,void* c,int m,int n,int k,void* stream,int config){
  if(config==1) return run<64,128,128,32,64>(a,b,c,m,n,k,(cudaStream_t)stream);
  if(config==2) return run<128,128,128,64,64>(a,b,c,m,n,k,(cudaStream_t)stream);
  return run<32,128,128,32,64>(a,b,c,m,n,k,(cudaStream_t)stream);
}

#include "fused_gemm.cuh"
