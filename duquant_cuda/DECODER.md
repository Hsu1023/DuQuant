# 完整 LLaMA decoder layer 实验

`decoder.py` 将已实现的 W4A4 Linear 接成了一个完整的 LLaMA-2 decoder block。模型主入口 `main.py` 尚未切换到此后端；这里是可独立运行、验证和测量的完整层原型。进一步的 32 层连续生成和显存测试见 [MODEL.md](MODEL.md)。

## 计时包含什么

一次调用包含两个 RMSNorm、全部七个投影、RoPE、attention、SwiGLU、两次残差加法，以及 KV cache 写入与读取。QKV 和 gate/up 合并为各一次投影；两条路径都使用相同的合并方式、融合 RMSNorm/SwiGLU/RoPE 和强制 FP16 Flash SDPA。

W4A4 路径额外包含双旋转、置换、动态量化、packing、INT4 GEMM/GEMV；Q/K 在 RoPE 后执行归一化 Hadamard 和 per-head-token 4-bit 非对称量化，V 执行 per-head-token 4-bit 非对称量化。Q/K clipping=0.9，V clipping=1.0。本页历史测速使用融合的蝶形 Hadamard；当前已改用融合 Tensor Core dot，并对齐 norm 和 SwiGLU 舍入，详见 [CALIBRATED.md](CALIBRATED.md) 的最终精度和速度。RoPE 禁止 FMA 合并，以保留参考实现中的 FP16 中间舍入。

Attention 的计算和 KV **存储**仍为 FP16：W4A4 的 K/V 存储量化后反量化的值。V 的量化独立于历史长度，因此在写 cache 时完成一次即可。没有声称实现 packed INT4 KV cache 或 INT4 attention GEMM。

prefill 使用 causal attention；decode 的 query 长度为 1，对已写入的全部历史及当前 K/V 做 attention。历史 cache 由对应路径的真实 prefill 初始化，初始化不计入 decode 单步耗时。每个 CUDA Graph 固定一个位置，重复测量该位置的一步；这不是完整生成循环的延迟。

## 测量条件

- RTX 4090，GPU 4；PyTorch 2.7.1+cu126、Triton 3.3.1、CUDA Toolkit 12.4、CUTLASS 3.4.1。
- 本地 LLaMA-2-7B **第 0 层真实权重**，hidden=4096、intermediate=11008、32 个 attention heads、head_dim=128。
- 输入为真实 tokenizer 生成的 token embedding，文本在脚本中固定并重复到所需长度。
- 使用固定随机旋转和置换，QKV 共享一组参数，gate/up 共享另一组参数；尚未使用 DuQuant 校准、SmoothQuant 或 LWC。
- 两边均使用 CUDA Graph；报告三次测量的中位数，并交替测量先后顺序。全部在线 GPU 工作计入，构造、权重导出、编译、历史 cache 初始化不计入。
- 冷缓存结果在调用前清理 256 MiB L2 污染缓冲区，清缓存开销通过 external events 排除在计时外。
- FP16 基线与 Hugging Face LLaMA decoder 层做了数值对照；没有用原仓库较慢的 eager attention 作为速度基线。

## 正确性与当前限制

保留两种独立参考，避免把不同算术模型混为一谈：

1. `integer_reference`：用 FP32 表示 0–15 整数码，独立计算稠密点积，再做零点修正和缩放。所测 K 下，点积不超过 `225*K < 2^24`，FP32 可精确表示这些整数累加结果。它没有调用 INT4 kernel。该路径采用纯 PyTorch 的 norm、RoPE、Hadamard 和 SwiGLU。
2. `reference`：传统 fake quant，先把每个量化值反量化、舍入到 FP16，再做浮点 GEMM。这与真实整数累加后统一缩放存在细小差异，后续 4-bit 量化会放大跨边界的变化。

在此次 batch=1/4 的真实权重实验中，CUDA 与整数数学参考的输出相对 L2 差异为 **0–1.11%**；三个 batch=1 decode 点和 batch=4 decode 点逐元素一致。与传统 fake-quant 参考的差异为 **2.86–5.02%**。这些差异都保存在 JSON 中，没有把它们作为 FP16 模型的质量指标。

本页未校准实验的整数参考相对 FP16 输出的 L2 差异约 **11–17%**。这些历史结果证明完整层原型有正的性能收益，**不证明完整 DuQuant 模型已达到可部署精度**。后续已完成校准接入、完整模型数值对照、PPL 和端到端速度复测，见 [CALIBRATED.md](CALIBRATED.md)。

测试包括：Hugging Face FP16 层对照、独立 attention 预处理对照、整数/fake-quant 整层对照、逐 token decode 对照 causal prefill、多 batch、cache 前缀保持不变、CUDA Graph replay 更新输入，以及内存检查。日志在 `benchmarks/decoder_memcheck.log`。

## 复现

在仓库根目录，沿用单层 CUDA 环境：

```bash
export CUDA_VISIBLE_DEVICES=4
export CUDA_HOME=/usr/local/cuda-12.4
export CUTLASS_PATH=/mnt/yichen/wyc/QDLM/QuaRot/third-party/cutlass
export TRITON_CACHE_DIR=/tmp/duquant_triton
export DUQUANT_PYTHON=/home/yichen/miniconda3/envs/hb/bin/python

"$DUQUANT_PYTHON" -m unittest discover -s tests -v
"$DUQUANT_PYTHON" benchmarks/bench_decoder.py --cold \
  --output benchmarks/decoder_reproduced_b1.json
"$DUQUANT_PYTHON" benchmarks/bench_decoder.py --batch 4 \
  --prefill 128 --decode-context 2048 --cold \
  --output benchmarks/decoder_reproduced_b4.json
```

接口为 `LlamaDecoderBlock(weights, rotations, mode='cuda')`、`cache = block.cache(batch, capacity)`、`y = block(x, cache, position)`。当前支持 head_dim=128 的 MHA、从零开始的 prefill 和单 token decode，不支持 GQA、带历史的 chunked prefill、训练 backward 或动态位置的整段生成 graph。

## 实测结果

以下来自 `benchmarks/decoder_final_b1.json` 与 `benchmarks/decoder_final_b4.json`，单位 μs。decode 的长度表示调用前已有的历史 token 数。

| batch | 场景 | 输入/历史长度 | FP16 | W4A4 | 加速比 | 冷缓存加速比 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | prefill | 128 | 550.5 | 220.8 | 2.49× | 2.45× |
| 1 | prefill | 512 | 1440.5 | 416.9 | 3.46× | 3.47× |
| 1 | prefill | 2048 | 5733.2 | 1729.2 | 3.32× | 3.25× |
| 1 | decode | 128 | 435.2 | 162.5 | 2.68× | 2.20× |
| 1 | decode | 2048 | 454.6 | 196.2 | 2.32× | 2.05× |
| 1 | decode | 4095 | 470.2 | 230.5 | 2.04× | 1.96× |
| 4 | prefill | 128 | 1410.2 | 395.8 | 3.56× | 3.60× |
| 4 | decode | 2048 | 585.3 | 310.5 | 1.88× | 1.76× |
