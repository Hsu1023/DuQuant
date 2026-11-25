# DuQuant 单层 W4A4 CUDA 原型

实现了完整的单层在线路径：两次 128×128 块旋转、跨块置换、per-token 非对称激活量化与打包、真实 INT4 GEMM/GEMV、零点修正及 FP16 输出。权重旋转、量化、打包在构造阶段执行一次，推理时不还原整份浮点权重，也不缓存激活或输出。

## 实现

- `backend.py`：PyTorch 接口、Triton 双旋转、量化/packing、单 token packed INT4 GEMV。
- `gemm.cu`：CUTLASS 无符号 INT4 Tensor Core GEMM，保留 INT32 输出版本用于对照。
- `fused_gemm.cuh`：把非对称零点修正、缩放、bias、FP16 写回融合到 GEMM epilogue，省掉 INT32 中间矩阵。
- `kv4.py`：真实 packed KV4 cache、融合 RoPE/量化/写入、直接读取压缩缓存的 split-K decode；GQA 共享解包结果并使用 Tensor Core。完整模型也支持显式 KV16，见 [KV 模式评测](CALIBRATED.md#kv-modes)。
- `ptq.py`：对称 W4 的 clipping 搜索和 GPTQ 导出；A/K/V 保持非对称，支持每层搜索后的 clipping。预量化权重直接进入现有整数 kernel，见 [LLaMA-2 消融](CALIBRATED.md#ptq-search)。
- 旋转中间张量采用 `[K,M]` 布局；第二次旋转把 permutation 融入读取，改善 gather 的合并访存。
- 量化使用精确舍入除法，匹配原仓库的 FP16 scale、zero point、clipping 和 round-to-even 规则。
- 小 batch decode 使用单独的 GEMV；其他形状使用按 M/N 选择的 GEMM tile。所有选择只依赖形状。

已经在编译后的 SASS 中确认 `IMMA.16864.U4.U4.SAT`。最终 forward 通常是四次 GPU kernel launch（双旋转、量化、GEMM/GEMV）。

## 支持范围

本文件介绍独立 Linear。完整 decoder block 见 [DECODER.md](DECODER.md)，当前校准后 32 层模型的精度、连续生成和显存测量见 [CALIBRATED.md](CALIBRATED.md)；尚未替换原始主入口中的 `QuantLinear`。

- 验证平台：RTX 4090，`sm_89`，PyTorch 2.7.1+cu126、Triton 3.3.1、CUDA Toolkit 12.4。
- 编译依赖：CUTLASS 3.4.1（本机 checkout `ffa34e70`），通过 `CUTLASS_PATH` 指定；未修改外部依赖。
- 连续 FP16 输入 `[...,K]`；K 是 128 的倍数，N 是 32 的倍数。支持多维 batch、bias、非默认 CUDA stream 和 CUDA Graph。
- 两个共享于各块的 128×128 旋转矩阵，以及一次长度 K 的通道置换；与原仓库默认 `permutation_times=1` 的结构对应。
- 非对称 per-output-channel 权重量化、per-token 激活量化，无 weight group。
- 暂不支持其他位宽、BF16、额外在线 `smooth_scales`/LET 或任意次数的旋转。已支持离线折叠 smoothing 和逐通道 LWC weight clipping，见 [CALIBRATED.md](CALIBRATED.md)；若引入额外在线缩放，其耗时也需要计入。
- 本文件的单个 Linear 计时不包含 RMSNorm、SwiGLU、attention 或 KV cache；完整层实验已计入这些工作，见 [DECODER.md](DECODER.md)。完整模型校准与 PPL 验证见 [CALIBRATED.md](CALIBRATED.md)。

## 使用与复现

在仓库根目录执行。本机有现成依赖，下面的路径是本次测量所用环境。

```bash
export CUDA_VISIBLE_DEVICES=4
export CUDA_HOME=/usr/local/cuda-12.4
export CUTLASS_PATH=/mnt/yichen/wyc/QDLM/QuaRot/third-party/cutlass
export TRITON_CACHE_DIR=/tmp/duquant_triton
export DUQUANT_PYTHON=/home/yichen/miniconda3/envs/hb/bin/python

# 首次构造会编译 CUDA 库，产物和编译日志在 duquant_cuda/build/。
# 应在 CUDA Graph capture 前完成构造和预热。
"$DUQUANT_PYTHON" -m unittest discover -s tests -v
"$DUQUANT_PYTHON" benchmarks/bench_single_layer.py --cold

# 使用 LLaMA-2 第一层真实 Q/up/down 权重（激活和旋转仍为合成数据）。
"$DUQUANT_PYTHON" benchmarks/bench_single_layer.py \
  --model /mnt/public_models/llama2-7b-hf --tokens 1,128,2048 --cold \
  --output benchmarks/llama2_weights_reproduced.json
```

```python
from duquant_cuda import DuQuantLinearCUDA

# weight: 原始 [N,K] FP16 CUDA 权重；r1/r2: [128,128] FP16 CUDA；
# perm: CUDA 整数置换向量。构造器执行离线权重变换和打包。
layer = DuQuantLinearCUDA(weight, r1, perm, r2,
                          bias=bias, act_clip=0.9, weight_clip=0.8)
y = layer(x)  # x: 连续 FP16 CUDA [...,K]
```

`--tune` 会在计时前搜索 launch 配置；最终结果文件使用的是默认形状分派，无需在线 autotune。切换 CUDA/CUTLASS 版本后应删除 `duquant_cuda/build/` 重新编译。该原型仅用于推理，不提供 backward。

## 测量方法与精度

基线是 `torch.nn.functional.linear(x, original_fp16_weight)`。两边都使用 CUDA Graph，DuQuant 计时包括全部在线旋转、置换、量化、packing 和矩阵乘法。离线权重导出及首次编译不计入推理延迟。

热缓存测量为重复单层执行；冷缓存测量在每次调用前写入 256 MiB 缓冲区，并通过 CUDA Graph external event 把清缓存操作排除在计时外，报告 30 次的中位数。冷缓存也只是单层实验，不能直接当作模型端到端结果。未锁定 GPU 时钟，微秒级边缘收益应重复验证。

最终 GPU 4 测量见 `benchmarks/final_synthetic.json` 和 `benchmarks/final_llama2_weights.json`。后者使用 LLaMA-2-7B 第一层真实权重，但激活为随机输入，旋转与置换为固定随机参数，**不是经过校准的完整 DuQuant 模型**。所有 24 个最终测量点的激活 packed 字节与 PyTorch 参考一致；相对 fake-quant 输出的 L2 误差约 0.03%–0.05%。融合计算不复现 fake quant 的每一步 FP16 反量化舍入，因此输出不承诺逐 bit 相等。该误差不是 W4A4 相对原始 FP16 模型的精度损失。

数值测试覆盖原仓库量化器、零/常量/单边输入、非 2 次幂 K、M/N 尾块、两种中间布局、融合与非融合 GEMM、GEMV、bias、多维 batch、非默认 stream 和 graph replay 后改变输入。`benchmarks/memcheck.log` 保存内存检查结果。`benchmarks/history/` 保存优化过程中的旧版本结果，不代表当前性能。

### LLaMA-2 第一层权重的测量结果

| K → N | tokens | FP16 μs | DuQuant μs | 热缓存加速 | 冷缓存加速 |
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

## 后续重点

16-token、4096×4096 的热缓存收益仅约 1.05–1.08 倍，仍值得优化小 M 的 GEMM 并行度。长输入的 down projection 中，旋转和量化仍占较大比例。完整模型现已复用 QKV 和 gate/up 的输入变换，并完成校准、PPL 和 prefill/decode 评测；后续仍可优化这些开销和校准精度。
