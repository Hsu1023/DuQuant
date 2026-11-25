# 完整模型质量：FP16、fake W4A4 与真实 INT4

本页保留首次未校准原型的历史结果。当前已完成训练集校准和数值修正，见 [CALIBRATED.md](CALIBRATED.md)：真实 INT4 PPL 为 6.1638，完整 test 的 top-1/NLL 与独立整数参考一致。

**全量 WikiText-2 PPL 接近，但输出不等价，且当前原型相对 FP16 有明显质量损失。** 本次评测不构成端到端数值一致性验收，也不代表原版 DuQuant 校准后的精度。

## 对照范围和口径

- 模型：本地 `/mnt/public_models/llama2-7b-hf`，全部 32 层；RTX 4090，物理 GPU 4。
- 数据：[Salesforce/WikiText](https://huggingface.co/datasets/Salesforce/wikitext) 的 `wikitext-2-raw-v1` test，本地 Arrow 缓存，4,358 行。源 revision、文件和 token SHA256 保存在 JSON。
- 将全文用两个换行连接，原模型 slow LlamaTokenizer，默认特殊 token，共 341,469 tokens。
- 序列长 2048，使用全部 166 个完整不重叠分块，丢弃尾部 1,501 tokens。每块从空有效 KV 开始，预测位置 1–2047，共 **339,802 个计分 token**。
- FP32 next-token cross entropy，按实际计分 token 总数计算平均 NLL，再取 exp。三种模式使用同一 Flash SDPA。运行的是带真实文本输入的完整模型，包含 FP16 embedding、最终 norm 和 LM head。
- fake 是此性能原型的 PyTorch fake-quant reference：先反量化到 FP16，再运行浮点矩阵乘。CUDA 是 packed INT4 点积后应用 affine scale/zero-point correction。它们存在舍入次序差别。
- 两种量化模式都是相同的未校准 seed=42 随机双旋转和置换；W4A4，weight clip=0.8、activation clip=0.9、Q/K clip=0.9、V clip=1.0。没有运行原版校准、SmoothQuant 或 LWC，也未用 test 数据调参。
- **不是原仓库 `main.py` 经过校准的模型对照。** 这里回答的是先前测速的同一原型，其真实 INT4 与同参数 fake 路径的质量差异。

## 全量结果

| 模式 | PPL ↓ | next-token top-1 accuracy ↑ |
|---|---:|---:|
| FP16 | 5.472100 | 61.0132% |
| 同参数 fake W4A4 | 8.345393 | 54.5362% |
| 真实 INT4 CUDA W4A4 | 8.359285 | 54.5506% |

真实 INT4 相比 fake 的 PPL 增加 **0.013893（0.1665%）**；相比 FP16 增加 **52.76%**，fake 相比 FP16 也增加 **52.51%**。所以平均 PPL 的主要退化在 fake 路径中就已存在，不能将此前 2.54–2.60× 加速描述为精度无损加速。

这里的 accuracy 是在真实前缀条件下预测下一个 token 的正确率，**不是 QA accuracy**，也不是两种模式的预测一致率。

## 参数相同，输出仍有明显差异

fake 与真实 INT4 的两项 SHA256 完全相同：

- 全部层的旋转和置换：`bb9da39428706312d09237dfd58d4d8f30642c9b016f9f266bf099dff3d505d7`
- 从各自量化表示恢复的 FP16 权重：`be7c0f09d01c9ac173f2bd709ae9036176f7b57404a5e6220efbdcd8b82885ab`

对全部 339,802 个计分位置，真实 INT4 与 fake 的 **top-1 预测一致率只有 83.3647%**。在前四块各取 64 个等间距位置、共 256 个位置的 logits 诊断中：

- logits 相对 L2 误差：34.8772%；不是逐 bit 相同。
- 平均 `KL(fake || CUDA)`：0.125232 nats。

这些是 teacher-forcing 下同一真实输入前缀的比较；自由生成可能进一步分叉。全量 PPL 接近不能推导出逐样本输出相同，也不能保证其他任务质量相同。

按 166 个配对分块做 10,000 次 bootstrap，CUDA/fake PPL 比值的描述性 95% 区间为 `[0.99705, 1.00655]`。这不是正式等价性检验。

## 补充整数算术诊断

为检查是否仅由 fake 的逐元素 FP16 反量化舍入解释差异，额外评测了**前 4 块、8,188 个计分 token**。独立参考将整数 code 存为 FP32，计算精确整数点积，再应用 scale；norm、激活和 Hadamard 使用 PyTorch。该参考也经过同样的参数哈希核验。

| 前 4 块模式 | PPL | 与整数参考 top-1 一致率 |
|---|---:|---:|
| fake W4A4 | 9.195651 | 82.4621% |
| CUDA W4A4 | 9.756077 | 83.5125% |
| 独立整数算术参考 | 9.793481 | 100% |

CUDA 和整数参考在这些分块的 PPL 较近，但输出仍不一致。已有单算子测试支持量化分界和舍入次序会放大数值差异；**本次诊断不能证明全部偏差都来自 fake 的反量化舍入，也未完成各融合算子误差的逐层归因**。需要继续收紧数值一致性，并导入实际校准参数后重新评测。当前结论只限于本次数据上的平均指标。

## 复现与文件

新增全位置 logits 模式 `last_token_only=False`，默认的生成接口仍只计算最后位置 logits。测试覆盖 causal prefix 与全位置输出的对应关系、next-token shift、FP32 CE 和准确计分 token 数；连同现有 CUDA 测试，共 12 项通过。未修改 CUDA kernel。

```bash
export CUDA_VISIBLE_DEVICES=4
export CUDA_HOME=/usr/local/cuda-12.4
export CUTLASS_PATH=/mnt/yichen/wyc/QDLM/QuaRot/third-party/cutlass
export TRITON_CACHE_DIR=/tmp/duquant_triton
export DUQUANT_PYTHON=/home/yichen/miniconda3/envs/hb/bin/python

"$DUQUANT_PYTHON" benchmarks/eval_ppl.py --mode fp16 --output benchmarks/ppl_fp16.json
"$DUQUANT_PYTHON" benchmarks/eval_ppl.py --mode fake --output benchmarks/ppl_fake.json
"$DUQUANT_PYTHON" benchmarks/eval_ppl.py --mode cuda --output benchmarks/ppl_cuda.json
"$DUQUANT_PYTHON" benchmarks/eval_ppl.py --mode integer_reference --max-blocks 4 \
  --output benchmarks/ppl_integer_reference_4blocks.json
"$DUQUANT_PYTHON" benchmarks/summarize_ppl.py
```

从仓库根目录执行，每个模式独立进程。评测和加载耗时不作为新的正式速度 benchmark；速度和显存口径见 [MODEL.md](MODEL.md)。

- `benchmarks/ppl_{fp16,fake,cuda}.json`：全量逐块 NLL、计分数量、PPL、数据和参数指纹。
- `benchmarks/ppl_integer_reference_4blocks.json`：补充整数参考。
- `benchmarks/ppl_summary.json`：核验一致条件后的指标与诊断汇总。
- `benchmarks/cache/ppl_*.npz`：本地逐位置预测、target 和采样 logits，可重新计算诊断；位于 git 忽略的 cache 目录。
