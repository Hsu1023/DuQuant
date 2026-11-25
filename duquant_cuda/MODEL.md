# 完整 LLaMA-2-7B：连续生成速度与显存

本页保留首次未校准原型的测速和显存结果。当前校准后模型及最终代码的复测见 [CALIBRATED.md](CALIBRATED.md)，端到端加速 2.52–2.57×，同时给出完整 PPL 与数值一致性验证。

`model.py` 已接入完整 32 层模型，以及 embedding、最终 RMSNorm、FP16 LM head 和 greedy argmax。`benchmarks/bench_model.py` 测量真正的自回归生成：每一步使用上一生成 token，逐步写入并扩展有效 KV cache。不是固定位置的重复单步，也不是单层时间的加总。

当前结果属于**未校准 W4A4 性能原型**。所有 decoder projection 使用真实 packed INT4 权重及 W4A4 GEMM/GEMV；embedding、LM head、attention 计算和 KV 存储仍为 FP16。旋转和置换参数是固定随机参数，尚未执行 DuQuant 校准、SmoothQuant 或 LWC。不能把性能结果当作完整模型质量已经验收。后续 [WikiText-2 全量质量评测](QUALITY.md) 得到 FP16 / fake / INT4 PPL 为 5.4721 / 8.3454 / 8.3593；INT4 与 fake 的 top-1 一致率只有 83.36%，尚未通过输出等价性验收。

## 实验口径

- GPU：RTX 4090（此机器报告约 48 GiB 显存），物理 GPU 4；batch=1，单 GPU，无 CPU offload。
- 模型：本地 `/mnt/public_models/llama2-7b-hf`，全部真实权重。
- Prompt：同一段固定文本重复到 128 或 2048 tokens，输入完全相同。
- 输出：固定生成 32 tokens，不因 EOS 提前停止，确保两种模式计算量可比。
- 两种模式在独立进程运行；测速时不存在第二份模型驻留。FP16 使用相同的合并投影、融合 norm/RoPE/SwiGLU 和 Flash SDPA。
- 为指定的 prompt 长度和输出长度分别捕获 prefill graph 与完整 decode graph，两者按顺序执行并共享 graph 内存池。
- 第一个生成 token 来自 prefill，剩余 31 个来自连续 decode；KV 预留容量为 `prompt_length + 31`。
- 每配置测量 5 次，报告中位数；未锁定 GPU 时钟。
- 总耗时是从已分词的 GPU token 输入开始，到生成的 token IDs 拷回 CPU 完成的 wall time。包含全模型计算、sampling 和实际 cache 更新，不包含 tokenization、模型加载/量化导出、首次编译、graph capture 或 prompt H2D。
- prefill 列为 GPU event 时间，包含第一个 token 的预测；它不是网络服务的用户感知 TTFT。decode 吞吐按剩余 31 个 token 计算。

当前 graph 固定请求形状及生成长度。这验证了固定工作负载下的真实连续生成，不代表已经实现支持动态 batch、动态停止和任意长度请求的服务框架。

## 显存口径

所有显存单位为 **GiB = 2^30 bytes**，区分以下几项：

- **模型张量**：按唯一 backing storage 去重，包括权重、量化 metadata、旋转/置换、norm 和共享 RoPE；不含 KV。
- **KV cache**：真实预分配的 K/V 张量，两条路径均为 FP16。
- **峰值 allocated**：PyTorch 活跃张量峰值。统计包含预热、capture 时实际运行的推理和 replay，排除模型加载、量化导出及 HF 对照加载。保留 capture 阶段的峰值，是因为 graph replay 不再向 allocator 逐次申请临时张量，replay 前直接 reset 会漏掉临时工作区的活跃峰值。
- **reserved**：稳定推理时 PyTorch 实际保留的显存，包括 graph 私有池和分配器碎片。容量规划不能只看模型张量或 allocated。
- **设备占用**：通过 CUDA `mem_get_info` 读取，进一步包含 CUDA context、graph 和 PyTorch allocator 之外的分配。这里在原本空闲的 GPU 4 上测量。

W4A4 的模型张量从 12.55 GiB 降为 3.54 GiB，减少约 71.8%；不是理论上的整机显存缩小四倍。KV 未压缩，而且离线构造造成的 allocator 碎片及 graph/context 开销仍然存在。

## 验证和限制

- 完整 FP16 模型与 Hugging Face 在独立短 prompt 上对照：最后位置 logits 相对 L2 误差约 0.084%，top-1 相同。参考模型在测速前释放，显存峰值也重新计数。
- 测试验证 growing-cache decode 与重新计算完整前缀的结果、两层真实 token 反馈、修改 prompt 后的 graph replay、以及清空 cache 后的结果一致。
- 每次完整模型 capture 后，都检查其 32-token 结果与 eager 生成逐 token 相等；检查 logits 有限。
- 两个本次固定重复文本 prompt 上，FP16 与原型的生成 token IDs 也相同。这是很有限的样例，**不等于 PPL 或广泛任务质量无损**。[多层 PPL 评测](QUALITY.md) 已确认原型有明显质量损失，真实校准及进一步的数值一致性工作仍待完成。
- 全部回归测试与 CUDA memcheck 日志见 `benchmarks/model_memcheck.log`。

## 文件与复现

原始数据是 `benchmarks/model_{fp16,cuda}_p{128,2048}_g32.json`，保留逐次时间、所有显存口径、生成 token IDs 和文本。汇总在 `benchmarks/model_summary.json`。

```bash
export CUDA_VISIBLE_DEVICES=4
export CUDA_HOME=/usr/local/cuda-12.4
export CUTLASS_PATH=/mnt/yichen/wyc/QDLM/QuaRot/third-party/cutlass
export TRITON_CACHE_DIR=/tmp/duquant_triton
export DUQUANT_PYTHON=/home/yichen/miniconda3/envs/hb/bin/python

# 每次命令单独启动进程，前一个退出后再启动下一个。
"$DUQUANT_PYTHON" benchmarks/bench_model.py --mode fp16 --prompt 128 \
  --new-tokens 32 --verify-hf --output benchmarks/model_fp16_p128_g32.json
"$DUQUANT_PYTHON" benchmarks/bench_model.py --mode cuda --prompt 128 \
  --new-tokens 32 --output benchmarks/model_cuda_p128_g32.json
"$DUQUANT_PYTHON" benchmarks/bench_model.py --mode fp16 --prompt 2048 \
  --new-tokens 32 --output benchmarks/model_fp16_p2048_g32.json
"$DUQUANT_PYTHON" benchmarks/bench_model.py --mode cuda --prompt 2048 \
  --new-tokens 32 --output benchmarks/model_cuda_p2048_g32.json
"$DUQUANT_PYTHON" benchmarks/summarize_model.py
```

`--verify-hf` 会在验证时暂时额外加载一份 FP16 模型，需要足够的验证显存；这不计入下表的推理显存。

## 实测汇总

| 输入 → 输出 tokens | FP16 总耗时 ms | W4A4 总耗时 ms | 加速比 | FP16 prefill ms | W4A4 prefill ms | FP16 decode tok/s | W4A4 decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| 128 → 32 | 459.63 | 176.75 | 2.60× | 17.85 | 7.32 | 70.2 | 183.0 |
| 2048 → 32 | 654.34 | 257.82 | 2.54× | 178.89 | 54.41 | 65.2 | 152.5 |

| 输入 → 输出 tokens | 模式 | 模型张量 GiB | KV GiB | 峰值 allocated GiB | reserved GiB | 设备占用 GiB |
|---|---|---:|---:|---:|---:|---:|
| 128 → 32 | fp16 | 12.55 | 0.08 | 12.67 | 12.68 | 13.26 |
| 128 → 32 | w4a4 | 3.54 | 0.08 | 3.65 | 4.73 | 5.44 |
| 2048 → 32 | fp16 | 12.55 | 1.02 | 13.88 | 14.02 | 14.60 |
| 2048 → 32 | w4a4 | 3.54 | 1.02 | 4.93 | 5.12 | 5.84 |
