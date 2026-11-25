# 校准后的 W4A4：精度、整数参考一致性、速度和显存

已接入原仓库的数据驱动双旋转、zigzag 置换和 smoothing，并修正三个会被后续量化放大的数值差异。LLaMA-2-7B 的真实 INT4 WikiText-2 PPL 从 **8.3593 降到 6.1638**，下降 **26.26%**；完整模型生成仍比 FP16 快 **2.52–2.57×**。

以下首先保留无 LWC 的结果；[LWC 续实验](#lwc-follow-up) 将真实 INT4 PPL 进一步降到 **6.0641**，端到端加速为 **2.51–2.55×**。最新的[五个 family 验证](#model-family-validation)覆盖 LLaMA-1/2/3、Vicuna 和 Mistral 的代表模型，全部不使用 LWC。

## 校准参数

`benchmarks/calibrate_model.py` 从 WikiText-2 **train** 抽取 128 个 2048-token 样本，seed=2。test 没有参与校准、参数搜索或 kernel 数值修正；数值诊断也使用 train 文本。

1. 逐层传播 FP16 校准输入，收集各 projection 输入的 per-channel absolute maximum。
2. 按原仓库 `smooth_and_let_inplace` 的零 shift 规则折叠四组 smoothing scale，alpha=0.6。QKV 和 gate/up 的 scale 折叠进 norm 与权重；O 和 down 的 scale 分别折叠进 V/O、up/down 权重，不引入在线 scaling kernel。
3. 直接调用原仓库 `UniformAffineQuantizer.online_duquant_cali`，每次 rotation 最多 256 步、block size=128、permutation_times=1。旋转代表输入是校准 embedding 的逐位置均值，并逐层传播。基础旋转按原 `get_rot.py` 的 Givens 构造生成，仅生成所需的 128 维版本。
4. 导出每层 smoothing scales、两个旋转和置换。加载时检查模型配置、manifest 和逐层文件哈希，再折叠原始权重并打包 INT4。

W4A4 的 weight clip=0.8、activation clip=0.9、Q/K clip=0.9、V clip=1.0。没有训练 LWC/LET。这是**将原仓库校准算法接入现有 CUDA 后端**；attention 使用该后端的 Flash SDPA 和 QK/V 量化路径，不是原 `main.py` 的 eager attention 逐行复现，也不是下载的论文 checkpoint。

参数包位于 `benchmarks/cache/calibrated_w4a4/`，约 16 MiB，受 git ignore 保护。manifest 的可审阅副本为 `benchmarks/calibrated/calibration_manifest.json`，记录训练数据和 token 哈希、128 个采样起点、参数和逐层文件 SHA256。manifest SHA256：

`7252d2f0c71598d66e95df9343b08788c4fd8a5d56180e4fcddc3a088bb5a39b`

## 全量 PPL

协议与首次评测相同：WikiText-2 raw test，slow LlamaTokenizer，全文以双换行连接；全部 166 个完整不重叠 2048-token 分块，实际计分 **339,802 tokens**。每块从 position=0 开始，FP32 next-token CE，使用实际计分 token 数作分母。各模式独立进程。

| 模式 | PPL ↓ | next-token 正确率 ↑ |
|---|---:|---:|
| 最终代码 FP16 基线 | 5.472090 | 61.0182% |
| 校准后 fake W4A4 | 6.156414 | 58.8587% |
| 校准后真实 INT4 CUDA | 6.163763 | 58.8384% |
| 校准后独立整数算术参考 | 6.163763 | 58.8384% |

CUDA 比 fake 的 PPL 高 **0.007349（0.1194%）**。相比 FP16，CUDA 的 PPL 仍高 **12.64%**，因此不是精度无损。这是 PPL 和 next-token 指标，没有运行 QA。

## 两种参考的区别与数值修正

旧式 fake 路径先将激活、权重逐元素反量化并舍入到 FP16，再做浮点 GEMM。整数参考则用 FP32 表示精确的 4-bit code，计算整数点积、零点修正，然后应用 scale。这里的 K≤11008，整数点积上界 `225*K < 2^24`，FP32 能精确表示；它独立于 INT4 kernel，也使用 PyTorch norm、Hadamard、SwiGLU。

相同参数下，旧式 fake 和真实 INT4 的舍入语义不同。本次全量 top-1 一致率为 **86.4854%**，采样 logits 相对 L2 差异仍为 27.03%。不能将平均 PPL 接近解读为与旧式 fake 输出逐 bit 等价。

逐算子诊断定位并修正：

- **RMSNorm**：复现当前 PyTorch FP32 mean 的四累加器、warp shuffle 归约顺序，并禁止乘加融合，保留相同 FP16 舍入。
- **Q/K Hadamard**：融合 kernel 内改用 Tensor Core dense dot，对齐参考 GEMM 的累加顺序，避免蝶形归约造成的舍入边界差异。
- **SwiGLU**：使用 CUDA libdevice 的 exp 和精确舍入除法，匹配 PyTorch 的指数及除法语义。

修正前，前四层在同一训练输入上的整层相对 L2 差异为 1.2894%、0.2923%、0.3490%、0.5684%；修正后这四个整层输出与整数参考逐 bit 相同。原始诊断数据为 `benchmarks/numeric_probe_train_{before,rms,rms_hadamard}.json`，最终为 `benchmarks/numeric_probe_train.json`。

**完整 32 层 CUDA 与整数参考**在全部 test 分块上进一步验证：

- 所有 339,802 个计分位置 top-1 一致率 **100%**。
- 全部 166 个分块 NLL 差值 **0**，因此最终 PPL 完全相同。
- 前四块各采样 64 个位置，共 256 个位置的全部词表 logits **逐 bit 一致**，相对 L2 和 KL 都为 0。

没有保存全量所有位置的完整 logits，因此全模型的逐 bit logits 结论只覆盖这 256 个采样位置。数据、旋转、置换和反量化后权重的 SHA256 也已核对一致。

## 最终代码的端到端速度和显存

RTX 4090（本机约 48 GiB），GPU 4，batch=1，每种配置独立进程，5 次测量取中位数。完整 32 层、自回归 token 反馈、真实 KV 更新，固定生成 32 tokens；从 GPU token 输入到生成 IDs 拷回 CPU 的 wall time。包括在线旋转、量化、packing、INT4 计算、norm、attention、LM head 和 argmax；排除加载、导出、tokenization、首次编译和 CUDA Graph capture。两种模式都使用同一套融合非线性算子及 Flash SDPA。

| 输入 → 输出 tokens | FP16 ms | 校准 INT4 ms | 加速比 | FP16 decode tok/s | INT4 decode tok/s |
|---|---:|---:|---:|---:|---:|
| 128 → 32 | 461.19 | 179.11 | **2.57×** | 69.9 | 180.5 |
| 2048 → 32 | 656.90 | 260.93 | **2.52×** | 65.0 | 150.6 |

GPU prefill 时间分别为 17.84 → 7.37 ms、179.89 → 54.92 ms。没有锁定 GPU 时钟；这些仍是固定形状 CUDA Graph 请求，非动态 serving benchmark。

显存单位 GiB。峰值 allocated 包括预热、capture 和推理临时张量，排除离线加载/导出和 HF 验证；reserved 包括 allocator 与 graph 池。设备占用包括 CUDA context 等开销。详细口径同 [MODEL.md](MODEL.md)。

| 输入长度 | 模式 | 模型张量 | KV | 峰值 allocated | reserved | 设备占用 |
|---|---|---:|---:|---:|---:|---:|
| 128 | FP16 | 12.55 | 0.08 | 12.67 | 12.68 | 13.26 |
| 128 | INT4 | 3.54 | 0.08 | 3.65 | 4.73 | 5.41 |
| 2048 | FP16 | 12.55 | 1.02 | 13.88 | 14.02 | 14.60 |
| 2048 | INT4 | 3.54 | 1.02 | 4.93 | 5.12 | 5.81 |

KV、embedding、LM head、attention 算术仍为 FP16。模型张量减少约 71.8%；不能将整机显存理解为理论上的四倍压缩。

## 检查和复现

14 项测试通过，包括新增 smoothing 折叠等价、RMSNorm 的多形状/残差逐 bit 检查，以及收紧为逐 bit 对照的整数参考整层测试。CUDA memcheck 为 **0 errors**。日志在 `benchmarks/calibrated/{tests,memcheck}.log`。最终 FP16 的短 prompt logits 相对 HF 的 L2 差异约 0.00698%，top-1 相同。

```bash
export CUDA_VISIBLE_DEVICES=4
export CUDA_HOME=/usr/local/cuda-12.4
export CUTLASS_PATH=/mnt/yichen/wyc/QDLM/QuaRot/third-party/cutlass
export TRITON_CACHE_DIR=/tmp/duquant_triton
export DUQUANT_PYTHON=/home/yichen/miniconda3/envs/hb/bin/python

# 重建参数时选择新目录，脚本拒绝覆盖已有 manifest。
"$DUQUANT_PYTHON" benchmarks/calibrate_model.py --output benchmarks/cache/calibrated_reproduced

# 使用本次固定导出参数进行复评。
"$DUQUANT_PYTHON" benchmarks/eval_ppl.py --mode fp16 --output benchmarks/calibrated/ppl_fp16.json
"$DUQUANT_PYTHON" benchmarks/eval_ppl.py --mode fake --calibration benchmarks/cache/calibrated_w4a4 --output benchmarks/calibrated/ppl_fake.json
"$DUQUANT_PYTHON" benchmarks/eval_ppl.py --mode cuda --calibration benchmarks/cache/calibrated_w4a4 --output benchmarks/calibrated/ppl_cuda.json
"$DUQUANT_PYTHON" benchmarks/eval_ppl.py --mode integer_reference --calibration benchmarks/cache/calibrated_w4a4 --output benchmarks/calibrated/ppl_integer_reference.json
"$DUQUANT_PYTHON" benchmarks/summarize_ppl.py --directory benchmarks/calibrated

"$DUQUANT_PYTHON" benchmarks/bench_model.py --mode fp16 --prompt 128 --new-tokens 32 --verify-hf --output benchmarks/calibrated/model_fp16_p128_g32.json
"$DUQUANT_PYTHON" benchmarks/bench_model.py --mode cuda --calibration benchmarks/cache/calibrated_w4a4 --prompt 128 --new-tokens 32 --output benchmarks/calibrated/model_cuda_p128_g32.json
"$DUQUANT_PYTHON" benchmarks/bench_model.py --mode fp16 --prompt 2048 --new-tokens 32 --output benchmarks/calibrated/model_fp16_p2048_g32.json
"$DUQUANT_PYTHON" benchmarks/bench_model.py --mode cuda --calibration benchmarks/cache/calibrated_w4a4 --prompt 2048 --new-tokens 32 --output benchmarks/calibrated/model_cuda_p2048_g32.json
"$DUQUANT_PYTHON" benchmarks/summarize_model.py --directory benchmarks/calibrated
```

每条命令单独启动并退出一个进程。正式结果是 `benchmarks/calibrated/ppl_summary.json` 和 `model_summary.json`；首次未校准结果保留在上一级 `benchmarks/`，不要混用。

## LWC follow-up

本次在上面的固定校准参数上训练 Learnable Weight Clipping。完整 32 层训练和评测均已完成，结果位于 `benchmarks/lwc/`，参数位于 `benchmarks/cache/lwc_w4a4/`。

### 训练设置与实现

- 复用同一组 WikiText-2 **train** 的 128 × 2048 tokens，seed=2；检查数据文件、token 和父 calibration manifest 的 SHA256。
- 每层 20 epochs，batch=1，AdamW，lr=0.01，weight decay=0。每个输出通道独立学习下界和上界的 sigmoid 系数，logit 初始化为 4.0。前层训练后的 fake 输出供给后层，目标是原始 FP16 模型对应层的输出，loss 为 FP32 MSE。
- 固定已有 smoothing（alpha=0.6）、旋转、置换和模型权重，不训练 LET。激活、Q/K、V 的 clipping 仍为 0.9、0.9、1.0。LWC 替换原来的固定 weight clip=0.8。
- 裁剪 logits 使用 FP32，sigmoid 系数转换为 FP16 后参与量化，forward 与部署的 FP16 quantizer 一致；round 使用 STE。每层导出后检查训练 forward 与 fake inference 输出完全一致，再传播量化输入。
- AMP 初始 scale=1024；发现梯度溢出时降低 scale 并重跑同一样本，不把跳过的更新计入训练步数。本次共 3 次重试，均在第 2 层。
- GPU 0 的前 8 层使用普通 PyTorch weight STE，后 24 层断点续训使用融合 Triton weight STE。完成层的文件和哈希保持不变，续训时重放 teacher/fake 输入。累计逐层用时 **72.38 分钟**，不包含开发验证和续训重放开销。

融合 weight STE 保留 FP16 舍入点与 ATen 梯度归约顺序。真实第一层、128 样本、2 epochs 的独立对照中，最终 logits 参数、裁剪系数和每轮 MSE 完全相同；层用时 22.22 → 15.04 秒。对应数据为 `fused_training_check.json`。这只是训练加速；推理仍使用原有 packed INT4 kernel，学习到的 clipping 在离线打包时生效。

参数 manifest SHA256：

`8c6a810d52eb4c3d37140569926637a954c241d8ac38f832f011a94f4807dc63`

可审阅的 manifest 和逐层训练记录分别为 `benchmarks/lwc/calibration_manifest.json`、`training.json`。训练代码为 `benchmarks/train_lwc.py`、`duquant_cuda/lwc.py` 和 `lwc_quant.py`。

### 全量 PPL 与数值核对

仍使用全部 166 个 2048-token test 分块，计分 339,802 tokens；协议与前述评测相同。FP16 PPL 复用未改变的基线结果。LWC fake、CUDA、整数参考分别在 GPU 0、1、2 的独立进程中重新评测。

| 模式 | 无 LWC PPL | LWC PPL | LWC next-token 正确率 |
|---|---:|---:|---:|
| FP16 | 5.472090 | 5.472090（基线） | 61.0182% |
| fake W4A4 | 6.156414 | **6.065405** | 58.9793% |
| 真实 INT4 CUDA | 6.163763 | **6.064058** | 59.0394% |
| 独立整数算术参考 | 6.163763 | **6.064058** | 59.0394% |

LWC 使真实 INT4 PPL 降低 **0.099705（1.6176%）**，fake PPL 降低 **1.4783%**。按相同 test 分块做 10,000 次 paired bootstrap，INT4 的 LWC/无 LWC PPL 比值描述性 95% 区间为 `[0.981342, 0.986284]`；不能据此推断其他数据集或任务的表现。前后对比记录为 `lwc_comparison.json`。

LWC CUDA 与 fake 的 PPL 相差 **−0.001347（−0.0222%）**，但 top-1 一致率为 **86.9977%**，仍不是逐 bit 等价；fake 的逐元素 FP16 反量化舍入语义没有改变。与独立整数参考相比，CUDA 在全部计分位置的 top-1 一致率为 **100%**，全部 166 块 NLL 差值为 **0**，256 个采样位置的完整词表 logits 逐 bit 一致。完整词表 logits 的逐 bit 结论仅覆盖这些采样位置。详见 `ppl_summary.json`。

[原论文 Table 1](https://arxiv.org/html/2406.01721v2#S4.T1) 报告 LLaMA-2-7B DuQuant+LWC WikiText-2 PPL 为 **6.08**，本次结果数值接近。但这是现有 CUDA 后端上的 LWC 实验：固定 alpha=0.6，而原仓库 `run_lwc.sh` 使用 alpha=0.5；attention 路径和部署一致的 FP16 训练量化语义也有区别。因此不是严格逐项复现论文，更不能根据这约 0.016 的差距宣称优于论文。未运行 C4 或 QA。

### LWC 模型的端到端速度与显存

RTX 4090，GPU 1，batch=1，5 次取中位数。FP16 与 LWC INT4 均在本次实验中重新测量，使用独立进程；CUDA Graph、自回归反馈、KV 更新及测量边界与前述基准相同。

| 输入 → 输出 tokens | FP16 ms | LWC INT4 ms | 端到端加速 | FP16 / INT4 decode tok/s |
|---|---:|---:|---:|---:|
| 128 → 32 | 462.11 | 181.26 | **2.55×** | 69.81 / 178.44 |
| 2048 → 32 | 664.46 | 264.57 | **2.51×** | 64.88 / 149.05 |

| 输入长度 | 模式 | 模型 GiB | KV GiB | 峰值 allocated GiB | reserved GiB | 设备占用 GiB |
|---|---|---:|---:|---:|---:|---:|
| 128 | FP16 | 12.55 | 0.08 | 12.67 | 12.68 | 13.26 |
| 128 | LWC INT4 | 3.54 | 0.08 | 3.65 | 4.73 | 5.41 |
| 2048 | FP16 | 12.55 | 1.02 | 13.88 | 14.02 | 14.60 |
| 2048 | LWC INT4 | 3.54 | 1.02 | 4.93 | 5.12 | 5.81 |

LWC 没有增加推理模型张量大小。KV、embedding、LM head 和 attention 算术仍为 FP16。结果是固定形状请求的端到端测试，不是动态 serving 吞吐；两条示例 prompt 的生成相同不代表所有生成相同。完整数据见 `model_summary.json`。

### 检查与复现 LWC

19 项回归测试通过，CUDA memcheck **0 errors**。新增测试覆盖与上游 FP16 LWC quantizer 对齐、训练梯度与导出、学习到的非对称裁剪在真实 GEMM/GEMV 上与整数参考对齐，以及融合 STE 的边界输入梯度和连续优化步骤。日志在 `benchmarks/lwc/{tests,memcheck}.log`。

使用前文的 Python/CUDA 环境变量：

```bash
# 完整重训选择新目录，避免覆盖本次参数；中断后可用相同参数加 --resume。
"$DUQUANT_PYTHON" benchmarks/train_lwc.py --calibration benchmarks/cache/calibrated_w4a4 --output benchmarks/cache/lwc_reproduced --epochs 20 --fused-weight-quant

# 使用本次已经完成的参数，输出到独立评测目录。
"$DUQUANT_PYTHON" benchmarks/eval_ppl.py --mode fp16 --output benchmarks/lwc_recheck/ppl_fp16.json
for mode in fake cuda integer_reference; do
    "$DUQUANT_PYTHON" benchmarks/eval_ppl.py --mode "$mode" --calibration benchmarks/cache/lwc_w4a4 --output "benchmarks/lwc_recheck/ppl_${mode}.json"
done
"$DUQUANT_PYTHON" benchmarks/summarize_ppl.py --directory benchmarks/lwc_recheck

# 速度/显存复测时为 FP16 与 CUDA 分别启动独立进程。
for prompt in 128 2048; do
    "$DUQUANT_PYTHON" benchmarks/bench_model.py --mode fp16 --prompt "$prompt" --new-tokens 32 --output "benchmarks/lwc_recheck/model_fp16_p${prompt}_g32.json"
    "$DUQUANT_PYTHON" benchmarks/bench_model.py --mode cuda --calibration benchmarks/cache/lwc_w4a4 --prompt "$prompt" --new-tokens 32 --output "benchmarks/lwc_recheck/model_cuda_p${prompt}_g32.json"
done
"$DUQUANT_PYTHON" benchmarks/summarize_model.py --directory benchmarks/lwc_recheck
```


<a id="model-family-validation"></a>
## 五个 model family 的端到端验证（无 LWC）

覆盖 README Model Zoo 的五个 family，各选择一个完整模型。全部使用同一套真实 W4A4 CUDA 后端，逐模型校准后重新运行完整 WikiText-2 test、独立整数参考及自回归生成。这里确认的是这五个具体 checkpoint，不代表该 family 的全部尺寸、变体或论文结果均已复现。

### 实现与校准范围

本次补齐 GQA：Q/K/V 可有不同投影宽度，RoPE/cache kernel 分别处理 query 和 KV heads，KV cache 按实际 KV head 数存储，Flash SDPA 使用 GQA。O 投影 smoothing 在共享同一个 V head 的 query-head 组内绑定，以便正确折叠到 V/O 权重；MHA 路径保持原有语义。模型加载器检查 head_dim=128，拒绝 scaled RoPE、sliding-window attention 和带 bias 的投影。

每个模型使用 WikiText-2 **train** 的 128×2048 tokens，seed=2、alpha=0.6、block=128、256 次旋转搜索；原仓库旋转搜索生成两次块旋转和 zigzag permutation，导出各模型独立的参数。没有新增 LWC/LET 训练。LLaMA-2 复用已完成的同协议无 LWC 校准，其余四个新做校准；五个模型的推理评测均重新运行。

旋转和 permutation 从校准文件读取，未硬编码成某一个模型的矩阵。旋转代表输入仍是传播的校准输入均值，attention 仍采用本后端的 Hadamard + QK/V qdq 语义；因此下文 fake 是**本后端相同参数的 PyTorch fake 参考**，不是重新运行上游完整评测入口，也不是论文数值的严格复现。

### 完整测试集质量

WikiText-2 test 按不重叠 2048-token 完整块计算，每块计分 2047 个位置。不同 tokenizer 的分块和 token 数不同，不跨模型直接比较 PPL。

| 模型 | Q/KV heads | 分块 / 计分 tokens | HF FP16 PPL | 后端 FP16 PPL | fake W4A4 PPL | CUDA INT4 PPL |
|---|---:|---:|---:|---:|---:|---:|
| LLaMA-1 7B | 32/32 | 166 / 339,802 | 5.677054 | 5.677063 | 6.313289 | 6.297440 |
| LLaMA-2 7B | 32/32 | 166 / 339,802 | 5.472053 | 5.472090 | 6.156414 | 6.163763 |
| LLaMA-3 8B | 32/8 | 141 / 288,627 | 6.135790 | 6.135898 | 8.115615 | 8.104349 |
| Vicuna-v1.5 7B | 32/32 | 166 / 339,802 | 6.783673 | 6.783619 | 7.578176 | 7.580546 |
| Mistral-7B-Instruct-v0.3 | 32/8 | 163 / 333,661 | 5.494614 | 5.494637 | 5.999756 | 5.998084 |

五个模型的真实 CUDA 与独立整数参考均满足：全部计分位置 top-1 **100% 一致**，全部分块 NLL 差值 **0**，每模型 256 个采样位置的完整词表 logits **逐 bit 一致**。完整 logits 的逐 bit 结论只覆盖采样位置。CUDA 与 FP16 fake 的舍入语义仍不同，PPL 接近不代表逐 token 等价。

HF 基线在完整测试集核对 PPL、所有预测位置及采样 logits，并额外检查长度 1/17/128 的 prefill 和各 3 步 cached decode。LLaMA-3 的 BOS-only 序列有一次 top-1 不同：HF 顶部 FP16 logits 恰好并列，后端候选在一个 FP16 ULP 内，已单独记录 `accepted_exact_hf_tie`，没有把它记成 top-1 一致。其他四个模型的 12 个短序列检查均 top-1 一致。完整 HF 数值差异见各目录的 `hf_comparison.json`。

### 整模型生成速度

同型 RTX 4090（约 48 GiB），单模型单 GPU、batch=1、5 次中位数。FP16 与 INT4 分别使用独立进程，均使用融合 block、Flash SDPA 和固定形状 CUDA Graph。包括 embedding、全部 32 层、LM head、真实 KV 更新、自回归 greedy 反馈及输出 token 回传；INT4 包括在线旋转、permutation 和量化。排除离线校准/打包、加载、首次编译、graph capture、tokenization 和输入 H2D。忽略 EOS，固定生成 32 tokens。为兼容 LLaMA-1 的 2048 上下文上限，统一测量 128/1024 输入长度。

| 模型 | 输入 → 输出 | FP16 ms | INT4 ms | 端到端加速 |
|---|---:|---:|---:|---:|
| LLaMA-1 7B | 128 → 32 | 461.27 | 179.48 | **2.57×** |
| LLaMA-1 7B | 1024 → 32 | 554.36 | 215.93 | **2.57×** |
| LLaMA-2 7B | 128 → 32 | 461.26 | 179.13 | **2.58×** |
| LLaMA-2 7B | 1024 → 32 | 555.67 | 215.91 | **2.57×** |
| LLaMA-3 8B | 128 → 32 | 519.00 | 218.05 | **2.38×** |
| LLaMA-3 8B | 1024 → 32 | 603.49 | 242.70 | **2.49×** |
| Vicuna-v1.5 7B | 128 → 32 | 461.32 | 179.42 | **2.57×** |
| Vicuna-v1.5 7B | 1024 → 32 | 553.16 | 215.65 | **2.57×** |
| Mistral-7B-Instruct-v0.3 | 128 → 32 | 492.33 | 191.13 | **2.58×** |
| Mistral-7B-Instruct-v0.3 | 1024 → 32 | 573.55 | 214.40 | **2.68×** |

### 显存

下表统一为 **1024 → 32**，单位 GiB，`FP16 → INT4`。模型张量统计包含旋转/量化参数；peak allocated 包含 graph capture/warmup 的峰值，reserved 为推理驻留池。设备总占用还包括 CUDA context 等，原始 JSON 另有记录。KV、embedding、LM head 和 attention 算术保持 FP16；LLaMA-3 的大词表使这些未量化权重占比更高。

| 模型 | 模型张量 | KV（两者相同） | 峰值 allocated | reserved |
|---|---:|---:|---:|---:|
| LLaMA-1 7B | 12.55 → 3.54 | 0.515 | 13.23 → 4.24 | 13.41 → 4.91 |
| LLaMA-2 7B | 12.55 → 3.54 | 0.515 | 13.23 → 4.25 | 13.39 → 4.91 |
| LLaMA-3 8B | 14.96 → 5.24 | 0.129 | 15.26 → 5.58 | 15.33 → 6.47 |
| Vicuna-v1.5 7B | 12.55 → 3.54 | 0.515 | 13.23 → 4.25 | 13.39 → 4.91 |
| Mistral-7B-Instruct-v0.3 | 13.52 → 3.80 | 0.129 | 13.82 → 4.14 | 13.89 → 5.15 |

### Checkpoint 来源和可复核记录

- LLaMA-1：ModelScope `skyline2006/llama-7b` 的原始 LLaMA-1 转换权重，位于 `benchmarks/cache/family_models/llama1-7b-ms`。所有下载文件按源 manifest SHA256 核验。旧包的 BOS/EOS ID 和空 special-token 字符串有误，按未修改的 SentencePiece 模型（BOS=1/EOS=2/UNK=0）修复，并补齐原有 2048 上下文配置；完整 before/after 保存在 `conversion.json`。33 个 bin 分片转换为 safetensors 后逐张量校验 dtype 和 bitwise equality。
- LLaMA-2：本地 `/mnt/public_models/llama2-7b-hf`。
- LLaMA-3：本地 `/mnt/public_models/Llama-3-8B`，使用其实际 `PreTrainedTokenizerFast`，没有用 LLaMA-2 tokenizer 代替。
- Vicuna：ModelScope `AI-ModelScope/vicuna-7b-v1.5`，位于 `benchmarks/cache/family_models/vicuna-7b-v1.5`。两个 bin 权重分片的 SHA256 均与 HF `lmsys/vicuna-7b-v1.5` commit `3321f76e3f527bd14065daf69dad9344000a201d` 一致；lossless safetensors 转换逐张量验证通过。
- Mistral：本地 `/mnt/public_models/Mistral-7B-Instruct-v0.3`，config 标识 `unsloth/Mistral-7B-Instruct-v0.3`，dense BF16 权重加载为 FP16。该 checkpoint 的 `sliding_window=null`；本次**未验证 Mistral-v0.1 的 sliding-window 路径**。

完整产物位于 `benchmarks/families/{llama1,llama2,llama3,vicuna,mistral}/`：`run.json`、校准 manifest、模型 config、5 种模式 PPL、预测/采样 logits 数组、HF 核对、速度/显存及日志。两份下载模型还记录源 manifest 和逐分片转换 hash。总表为 [`benchmarks/families/summary.json`](../benchmarks/families/summary.json)，汇总脚本会检查五个 family 均完成、整数参考覆盖完整测试集、校准协议匹配以及推理源码 hash 未改变。

22 项回归测试通过，CUDA memcheck **0 errors**，日志为 `benchmarks/families/tests.log` 和 `memcheck.log`。新增测试覆盖 4Q/1KV 与 4Q/2KV、prefill/多步 decode、compact cache、graph replay 改变输入、GQA smoothing 的浮点等价及非法未绑定 scale 的拒绝。

本次不包括更大参数尺寸、scaled RoPE、滑窗 attention、多 GPU、W6A6、QA/C4 或动态 serving。以上速度仅适用于测过的固定形状请求；质量结果也不能代替这些未测范围的结论。

### 复现

使用前文 Python/CUDA 环境。`download_family_models.py` 为两个缺失模型提供支持续传的 ModelScope 下载和 SHA256 核验；`convert_family_checkpoint.py` 做可核验的 lossless 格式转换。模型文件位于被 git 忽略的 `benchmarks/cache/`。

```bash
# 在仓库根目录运行；首次下载/转换时使用。
"$DUQUANT_PYTHON" benchmarks/download_family_models.py
"$DUQUANT_PYTHON" benchmarks/convert_family_checkpoint.py benchmarks/cache/family_models/vicuna-7b-v1.5
"$DUQUANT_PYTHON" benchmarks/convert_family_checkpoint.py benchmarks/cache/family_models/llama1-7b-ms --legacy-llama1

# 以 LLaMA-3 为例。重做校准请选择一个新目录。
"$DUQUANT_PYTHON" benchmarks/calibrate_model.py --model /mnt/public_models/Llama-3-8B --output benchmarks/cache/llama3_reproduced
# name 决定产物目录；复测选择新 name，避免复用已有结果。
"$DUQUANT_PYTHON" benchmarks/run_family.py --name llama3_recheck --model /mnt/public_models/Llama-3-8B --calibration benchmarks/cache/llama3_reproduced

# 核验本次五个 family 的已有产物并生成总表。
"$DUQUANT_PYTHON" benchmarks/summarize_families.py
```

### 五模型速度与显存的一键重测

`benchmarks/run_e2e_suite.py` 使用上述五份已验证的无 LWC 校准参数，在同一张空闲 GPU 上依次运行 20 组独立进程：五个模型 × FP16/INT4 × 128/1024-token 输入，均生成 32 tokens，默认每组重复 10 次。这里专门复测生成性能；完整 PPL 结果保留在 `benchmarks/families/`。

```bash
# 只检查校准配置、列出命令，不使用 GPU。
"$DUQUANT_PYTHON" benchmarks/run_e2e_suite.py --gpu 0 --output benchmarks/e2e_reproduced --dry-run
# 确认该 GPU 空闲后执行；输出目录必须不存在。
"$DUQUANT_PYTHON" benchmarks/run_e2e_suite.py --gpu 0 --output benchmarks/e2e_reproduced
```

该脚本在每组开始前检查 GPU 是否空闲，运行中每 2 秒检查一次其他计算进程；发现争用会将该轮标记为未完成，不生成成功总表。它不终止其他用户的任务，也不更改 GPU 时钟。校准与源码 SHA256、命令、运行日志、每次计时样本及结果均保存到新目录。`summary.json` 区分模型张量、KV、峰值 allocated、reserved 和包含 CUDA context 的设备占用；设备占用是快照，不声称为采样峰值。

首次追加重测的尝试记录在 `benchmarks/e2e_recheck/run.json`：当时所有 GPU 均被其他任务占用，空闲检查在模型加载前退出，完成 **0/20** 组。GPU 0 释放后的成功重测记录在下面；上文历史表格仍保留其原有 5 次中位数口径。


<a id="e2e-recheck"></a>
### 五模型端到端重测已完成（同一 GPU，各 10 次）

GPU 0 释放后，执行 `benchmarks/run_e2e_suite.py --gpu 0 --output benchmarks/e2e_fresh`，全部 **20/20** 组完成。五个模型的 FP16/INT4 均在同一张 RTX 4090 上依次运行，各组使用独立进程、10 次取中位数；轮询未发现该卡有其他计算进程。未修改推理内核、校准参数，也未新增 LWC。

本轮仍为 batch=1、输入 128/1024 tokens、固定生成 32 tokens，含全部 32 层、在线旋转/量化、attention、KV 更新、LM head 和自回归反馈。FP16 与 INT4 同样使用固定形状 CUDA Graph；排除加载、离线打包、编译、graph capture、tokenization 和输入 H2D，包含生成 token 的 D2H。20 组均通过 graph/eager 生成一致性和有限 logits 检查。

| 模型 | 输入→输出 | FP16 ms | INT4 ms | 端到端加速 | FP16 / INT4 decode tok/s |
|---|---:|---:|---:|---:|---:|
| LLaMA-1 7B | 128→32 | 461.24 | 179.10 | **2.58×** | 69.94 / 180.57 |
| LLaMA-1 7B | 1024→32 | 553.68 | 215.87 | **2.56×** | 67.17 / 163.06 |
| LLaMA-2 7B | 128→32 | 461.32 | 179.12 | **2.58×** | 69.92 / 180.54 |
| LLaMA-2 7B | 1024→32 | 552.51 | 216.06 | **2.56×** | 67.17 / 162.91 |
| LLaMA-3 8B | 128→32 | 518.10 | 215.90 | **2.40×** | 62.31 / 149.71 |
| LLaMA-3 8B | 1024→32 | 601.17 | 240.76 | **2.50×** | 61.78 / 147.06 |
| Vicuna-v1.5 7B | 128→32 | 461.37 | 179.14 | **2.58×** | 69.93 / 180.54 |
| Vicuna-v1.5 7B | 1024→32 | 554.49 | 215.98 | **2.57×** | 67.17 / 163.02 |
| Mistral-7B-Instruct-v0.3 | 128→32 | 492.19 | 190.02 | **2.59×** | 65.63 / 170.33 |
| Mistral-7B-Instruct-v0.3 | 1024→32 | 575.45 | 214.77 | **2.68×** | 65.03 / 166.91 |

显存统一列出 **1024→32**，单位 GiB，数值为 `FP16 → INT4`。峰值 allocated 包含 capture/warmup 峰值；设备占用为推理驻留快照（含 context），不是设备峰值。128 输入的完整显存指标同样保存在 JSON。

| 模型 | 模型张量 | KV（两者相同） | 峰值 allocated | reserved | 设备占用 |
|---|---:|---:|---:|---:|---:|
| LLaMA-1 7B | 12.55 → 3.54 | 0.515 | 13.23 → 4.24 | 13.41 → 4.91 | 13.99 → 5.59 |
| LLaMA-2 7B | 12.55 → 3.54 | 0.515 | 13.23 → 4.25 | 13.39 → 4.91 | 13.97 → 5.59 |
| LLaMA-3 8B | 14.96 → 5.24 | 0.129 | 15.26 → 5.58 | 15.33 → 6.47 | 15.91 → 7.15 |
| Vicuna-v1.5 7B | 12.55 → 3.54 | 0.515 | 13.23 → 4.25 | 13.39 → 4.91 | 13.97 → 5.59 |
| Mistral-7B-Instruct-v0.3 | 13.52 → 3.80 | 0.129 | 13.82 → 4.14 | 13.89 → 5.12 | 14.47 → 5.80 |

本次总表为 [`benchmarks/e2e_fresh/summary.json`](../benchmarks/e2e_fresh/summary.json)；`plan.json` 记录全部命令及校准 hash，`run.json` 记录 20/20 完成状态及推理/测量源码 hash，每个模型目录保留原始 10 次计时、显存分项、生成 token 和日志。此前因 GPU 占用而退出的 `benchmarks/e2e_recheck/` 保留为失败尝试记录，不用于本次结论。质量验证仍引用前述完整 WikiText-2 五模型结果，本次没有重复评测 PPL。


<a id="kv-modes"></a>
## 显式 W4A4KV16 与真实压缩 W4A4KV4

已实现并验证五个代表模型的两种显式 KV 模式。为了隔离 attention/KV 的影响，本次固定此前各模型的无 LWC 校准权重、双旋转、permutation 和 smoothing，没有重新校准或训练 LWC。

### 模式定义与实现

| 参数 | Q | K/V 数值处理 | KV 物理存储 | attention |
|---|---|---|---|---|
| 省略 `--kv-bits`（兼容旧路径） | Hadamard 后 Q4 QDQ | K4/V4 QDQ | FP16 | Flash SDPA |
| `--kv-bits 16` | FP16，无额外 QDQ/Hadamard | W4A4 投影输出，不再额外量化 | FP16 | Flash SDPA |
| `--kv-bits 4` | RoPE 后 Hadamard，保留 FP16 | K 经过配对 Hadamard 后量化；V 直接量化 | 真正 packed 4-bit + FP16 scale/zero | prefill Flash SDPA；decode 直接读压缩缓存 |

两种显式模式都保留 W4A4 的 QKV/O/MLP 线性层。KV16 仍然包含这些线性层的量化误差，不是原始 FP16 模型的 K/V。KV4 没有额外引入 QuaRot/SpinQuant 的 V/O 旋转，因此本实验不是那些方法的复现。

新代码 `duquant_cuda/kv4.py` 实现融合 RoPE、Q/K Hadamard、K/V 非对称量化和写 cache。每个 token、每个 KV head 的 128 维向量独立量化：K clip=0.9、V clip=1.0；两个 code 打包进一个 uint8，另存 FP16 scale 和 zero。每个 K/V 对由 512 bytes 变为 136 bytes，即 **3.7647×** 压缩（含量化参数，不含 decode 临时工作区）。

prefill 同时产生当前输入块的 FP16 反量化 K/V 临时张量供 Flash SDPA 使用；持久 cache 只有 packed codes 与量化参数。decode 分块在寄存器/共享内存内反量化，**不展开整份历史 FP16 cache**。MHA 使用 FP32 split-K 归约；GQA 让共享 KV 的 query-head 组复用解包结果，QK/PV 使用 Tensor Core，softmax 权重拆成 FP16 high/low 两部分计算 PV，减少单次 FP16 截断损失。softmax 与 partial/merge 累加为 FP32。tile 根据缓存容量选择 32/64/128，分派仅依赖形状。

### 完整 WikiText-2 PPL

仍为完整、不重叠 2048-token test 分块，与前述 family 评测相同的 tokenizer 和计分协议。表中的旧路径是此前真实 INT4 Linear + Q/K/V4 QDQ + FP16 物理 cache；不能把它称为显式 KV16 或真实压缩 KV4。

| 模型 | 原始 FP16 | 旧 Q4/K4/V4 QDQ 路径 | W4A4KV16 | W4A4KV4 |
|---|---:|---:|---:|---:|
| LLaMA-1 7B | 5.677063 | 6.297440 | **6.177697** | 6.255274 |
| LLaMA-2 7B | 5.472090 | 6.163763 | **5.999411** | 6.083101 |
| LLaMA-3 8B | 6.135898 | 8.104349 | **7.725156** | 7.938467 |
| Vicuna-v1.5 7B | 6.783619 | 7.580546 | **7.411816** | 7.501459 |
| Mistral-7B-Instruct-v0.3 | 5.494637 | 5.998084 | **5.897815** | 5.946837 |

在这五个 checkpoint 的完整 WikiText-2 评测上，KV16 均优于 KV4，也均优于旧路径。但这是同一份既有校准参数下的实测结论，不是对所有任务、样本或模型的保证；与原始 FP16 相比仍有量化损失。KV4 相比旧路径还同时取消了 Q4 量化，因此不能把二者精度差全部归因于 cache 格式。

两种新模式的全量 **prefill PPL** 均与独立整数算术参考匹配：所有计分位置预测相同、每块 NLL 差为 0、256 个采样位置的完整词表 logits 逐 bit 相同。

### 缓存解码的数值界限

上面的 PPL 以整块 prefill 计算，不能直接证明新的 cached-decode kernel 与另一种浮点归约实现逐 bit 等价。实际检查发现：FP32 attention 的细微舍入差异会在后续 W4A4 量化边界被放大。提高到 FP64 的隔离试验也未保证整模型一致，因此没有为了追求这种一致性而采用更慢的 FP64 路径。

最终对每个模型做了 **8,640 次**真实 cached-decode attention 检查：在完全相同的 Q/K/V 上，对照独立 PyTorch FP32 dense attention，再舍入到 FP16，全部满足 `abs(error) <= 0.001 + 0.002*abs(reference)`。另外保留长度 1/129/1024 的 prefill + 7 步解码的整模型 logits 诊断，没有隐去其差异。

以下是额外的真实缓存评测：从 test 中均匀选取 8 个窗口，各输入 128 tokens，再逐 token 喂入 32 个真实后续 token，共计分 256 tokens。它是有限诊断，不能与完整测试集 PPL 混用，也不能视为正式等价性检验。

| 模型 | KV16 CUDA/参考 PPL（相同） | KV4 CUDA PPL | KV4 dense 参考 PPL | KV4 与参考预测一致率 | 单次 attention 最大相对 L2 |
|---|---:|---:|---:|---:|---:|
| LLaMA-1 7B | 7.668679 | 8.004426 | 8.139375 | 91.80% | 0.010380% |
| LLaMA-2 7B | 7.400700 | 7.449666 | 7.558630 | 85.55% | 0.012697% |
| LLaMA-3 8B | 17.264989 | 16.974064 | 17.253970 | 82.42% | 0.023850% |
| Vicuna-v1.5 7B | 9.926909 | 9.625573 | 9.978095 | 86.33% | 0.024869% |
| Mistral-7B-Instruct-v0.3 | 6.561293 | 6.471767 | 6.740647 | 86.72% | 0.016411% |

**KV4 cached decode 的整模型输出不是逐 bit / 逐 token 等价于独立 dense 参考。** 小样本 PPL 更低也不能据此宣称 kernel 改善了量化方法。KV16 在本次 cached-decode 诊断中与参考完全匹配；KV4 当前应按实验性后端看待，其 kernel 的共同输入数值检查、完整 prefill PPL 和独立 cached 评测是三类不同证据。

### 端到端生成速度

同型 RTX 4090（约 48 GiB），每个模型的 FP16/KV16/KV4 对照使用同一张 GPU、各组独立进程，batch=1、固定生成 32 tokens、10 次中位数。测量边界与上文一致：全 32 层、真实反馈和 KV 写入、在线旋转/量化、输出 D2H 都计入；离线校准/打包、加载、编译、graph capture、tokenization、输入 H2D 不计。均使用固定形状 CUDA Graph。

| 模型 | 输入→输出 | FP16 ms | KV16 ms | KV4 ms | KV16 / KV4 相对 FP16 加速 |
|---|---:|---:|---:|---:|---:|
| LLaMA-1 7B | 128→32 | 461.40 | 175.54 | 179.64 | 2.63× / 2.57× |
| LLaMA-1 7B | 1024→32 | 554.76 | 211.58 | 208.53 | 2.62× / 2.66× |
| LLaMA-2 7B | 128→32 | 461.34 | 175.45 | 179.35 | 2.63× / 2.57× |
| LLaMA-2 7B | 1024→32 | 550.57 | 211.21 | 207.95 | 2.61× / 2.65× |
| LLaMA-3 8B | 128→32 | 518.22 | 209.22 | 212.35 | 2.48× / 2.44× |
| LLaMA-3 8B | 1024→32 | 598.62 | 232.22 | 238.08 | 2.58× / 2.51× |
| Vicuna-v1.5 7B | 128→32 | 461.17 | 175.40 | 179.59 | 2.63× / 2.57× |
| Vicuna-v1.5 7B | 1024→32 | 551.11 | 210.87 | 207.27 | 2.61× / 2.66× |
| Mistral-7B-Instruct-v0.3 | 128→32 | 492.41 | 183.78 | 187.03 | 2.68× / 2.63× |
| Mistral-7B-Instruct-v0.3 | 1024→32 | 573.72 | 206.61 | 212.50 | 2.78× / 2.70× |

KV4 的主要确定收益是缓存压缩，并不保证比 KV16 更快；本次 1024 输入下，MHA 模型 KV4 稍快，GQA 模型 KV16 稍快。为减少 GQA 重复解包开销，最终采用共享 KV 的 Tensor Core kernel；其 decode 与性能重新测量，未变的 prefill PPL、KV16、MHA 结果保留，`reused_outputs` 和前后源码 hash 记录了这一点。

### 显存与长上下文

下表均为 1024→32，单位 GiB，`KV16 → KV4`；模型张量不随 KV 位宽改变。峰值 allocated 包含 graph capture/warmup；reserved 是 PyTorch 的驻留内存池，不能等同于实际活跃张量。KV4 的 decode 工作区额外计入峰值，缓存列只统计持久 codes/scale/zero。

| 模型 | 模型张量 | KV 缓存 | KV4 decode 工作区 | 峰值 allocated | reserved | 设备占用快照 |
|---|---:|---:|---:|---:|---:|---:|
| LLaMA-1 7B | 3.537 | 0.515 → 0.137 | 0.0045 | 4.245 → 3.897 | 4.908 → 4.941 | 5.587 → 5.620 |
| LLaMA-2 7B | 3.537 | 0.515 → 0.137 | 0.0045 | 4.246 → 3.898 | 4.910 → 4.943 | 5.589 → 5.622 |
| LLaMA-3 8B | 5.244 | 0.129 → 0.034 | 0.0084 | 5.583 → 5.501 | 6.471 → 6.516 | 7.150 → 7.194 |
| Vicuna-v1.5 7B | 3.537 | 0.515 → 0.137 | 0.0045 | 4.246 → 3.898 | 4.910 → 4.943 | 5.589 → 5.622 |
| Mistral-7B-Instruct-v0.3 | 3.798 | 0.129 → 0.034 | 0.0084 | 4.138 → 4.056 | 5.123 → 5.168 | 5.802 → 5.847 |

Mistral-7B-Instruct-v0.3 另外测试原生 8192-token 输入、生成 32 tokens（无 sliding window）：

| 模式 | wall ms | 相对 FP16 加速 | KV GiB | 峰值 allocated GiB | reserved GiB |
|---|---:|---:|---:|---:|---:|
| fp16_kv16 | 1368.48 | 1.00× | 1.004 | 15.786 | 15.939 |
| cuda_kv16 | 564.83 | 2.42× | 1.004 | 6.427 | 6.697 |
| cuda_kv4 | 566.37 | 2.42× | 0.267 | 5.762 | 6.703 |

在这个 8192→32 场景，KV4 与 KV16 的延迟接近，KV 从约 1.004 降到 0.267 GiB，峰值 allocated 从约 6.43 降到 5.76 GiB；但 **reserved 两者都约 6.70 GiB**，allocator/graph pool 没有按活跃张量节省量等额释放驻留显存。不能把 KV 数据压缩倍数称为整模型总显存压缩倍数。旧 GQA scalar decode 约 637 ms 的结果保留在 `diagnostics_scalar_decode/`；最终版本约 566 ms。

### 检查与复现

最终 25 项测试通过，CUDA memcheck **0 errors**：`benchmarks/kv_modes/tests_final.log`、`memcheck_final.log`。新增覆盖 packed bytes/scale/zero、MHA/GQA、batch=2、31/32/33、63/64/65、127/128/129 等边界、连续追加、Graph 修改输入、完整 block 与 KV16。

完整总表为 [`benchmarks/kv_modes/summary.json`](../benchmarks/kv_modes/summary.json)。各 family 保存两种模式的完整 PPL/独立参考、cached-decode 逐 token NLL、性能原始 10 次样本、显存分项、源码和校准 hash。早期未通过严格整模型 logits 比较的诊断和未采用的 scalar GQA 性能明确保存在 diagnostics 目录，不作为最终成功结论。

```bash
# 沿用前述 Python/CUDA 环境；选择空闲 GPU。
# 使用相同的无 LWC 校准，显式选择真实 KV4。
"$DUQUANT_PYTHON" benchmarks/bench_model.py --model /mnt/public_models/llama2-7b-hf --mode cuda --kv-bits 4 --calibration benchmarks/cache/calibrated_w4a4 --prompt 1024 --new-tokens 32 --repeats 10 --output benchmarks/kv4_recheck_model.json
# KV16：将 --kv-bits 4 改为 --kv-bits 16。

# 在新目录重做该 family 的两种模式、PPL、缓存诊断和生成基准。
"$DUQUANT_PYTHON" benchmarks/run_kv_modes.py --family llama2 --output-root benchmarks/kv_modes_reproduced
# 中断后显式加 --reuse-existing；已有结果不会默认覆盖。
"$DUQUANT_PYTHON" benchmarks/summarize_kv_modes.py
```

适用范围仍限于已验证的这些 checkpoint、head_dim=128、当前 GPU/软件环境、固定形状生成；没有增加 paged serving、多 GPU、滑窗或 scaled RoPE 支持，也没有运行新的 QA/C4。


<a id="ptq-search"></a>
## 对称 W、非对称 A/K/V：clipping 搜索与 GPTQ 消融

本轮先验证 LLaMA-2 7B。固定此前无 LWC 校准的所有 DuQuant 双旋转、permutation、smoothing 和模型 checkpoint，只改变量化器及权重取整方法。新增配置属于 DuQuant 的 PTQ 扩展；没有重跑其他四个 family，也没有进行新的 QA 评测。

### 设置与数据隔离

- W：per-output-channel，对称有符号区间 [-8,7]，FP16 scale；存储 code=q+8、zero=8，复用已有 UINT4 GEMM/GEMV 和 affine epilogue。GPTQ 的最终整数 codes 直接导出、加载，不经第二次 RTN。
- A：动态 per-token 非对称；Q 保持 FP16。KV4：per-token/head 非对称 group=128、真实 packed cache，保留原有 post-RoPE Q/K Hadamard，没有新增 V/O 旋转。
- W 搜索：每个输出通道扫描 0.20–1.00、步长 0.01，以原始旋转后权重的 FP32 重构 MSE 选取 clipping，scale 最终存为 FP16。
- A/K/V 搜索：0.40–1.00、步长 0.025，以部署时 FP16 QDQ 的重构 MSE 选择；A 每层每个融合投影一项，K/V 各每层一项。每条训练序列均匀取 64 行，合计每层每类 8192 行。
- 数据：既有 WikiText-2 train 的 128×2048 样本、seed=2；核对原始 Arrow/token/校准 hash。RTN 搜索时逐层传播已搜索量化层的输出。测试集不参与任何 clipping 选择。
- GPTQ：固定与搜索 RTN 完全相同的 scale、zero、A/K/V clipping；block=128、damping=1%、不开 act-order。按 QKV→O→gate/up→down 真正顺序处理，使用自己的量化前缀和全部训练 token 的 A4 输入构造 Hessian。W clipping 不在 GPTQ 中重新搜索。
- 两种 KV 位宽使用相同导出权重和 A clipping，不为 KV16 单独重做校准。

第一轮较窄网格触及下界，因此仅根据训练重构误差扩大范围后重跑。初次诊断保留在 `benchmarks/ptq_llama2/`，最终结果位于 `benchmarks/ptq_llama2_final/`。

### 完整 WikiText-2 test PPL

与此前相同 tokenizer、2048 非重叠分块、166 blocks、339802 计分 tokens；表中均为真实 CUDA kernel 的完整 prefill PPL。原始 FP16 PPL 为 5.472090。

| 配置 | W4A4KV16 | W4A4KV4 |
|---|---:|---:|
| 原有非对称 W，固定 clipping，RTN | 5.999411 | 6.083101 |
| W 对称，固定 clipping，RTN | 6.057011 | 6.135865 |
| W 对称，搜索 W clipping，RTN | 5.949731 | 6.027398 |
| W 对称，搜索 W/A/K/V clipping，RTN | 5.896381 | 5.974302 |
| 相同搜索结果 + GPTQ | 5.822099 | 5.890531 |

单独改为对称 W、保留 W=0.8/A=0.9/K=0.9/V=1.0 的固定 clipping 略微变差；搜索 clipping 后改善。相同 clipping 下，GPTQ 相对 RTN 的 KV16/KV4 PPL 再降低约 0.0743/0.0838，因此本 checkpoint 上存在小幅正收益，不能泛化为所有模型都有同样收益。

全部 8 组（4 个配置×2 种 KV 位宽）的完整 PPL 均与独立整数参考相同：所有计分位置预测相同、逐块 NLL 差为 0、256 个采样位置的完整词表 logits 逐 bit 相同。脚本还校验同一配置的 KV16/KV4 权重完全相同、W-only 搜索与全搜索的权重完全相同，以及 RTN/GPTQ 的全部 scale/zero 完全相同。

| 投影 | W clipping 最小/平均/最大 | A clipping 取值 | GPTQ/RTN 局部 Hessian 加权误差均值 |
|---|---:|---|---:|
| qkv | 0.350 / 0.627 / 0.780 | 0.7 | 0.5125 |
| o | 0.400 / 0.625 / 0.770 | 0.675, 0.7 | 0.3425 |
| gate_up | 0.340 / 0.627 / 0.780 | 0.7 | 0.5693 |
| down | 0.370 / 0.589 / 0.720 | 0.625, 0.65, 0.675 | 0.5176 |

K clipping 取值：[0.925]；V clipping 取值：[0.925, 0.95]。局部加权重构误差的降低不能替代端到端 PPL。

### 端到端性能与显存

同一张 GPU 0 RTX 4090，独立进程，batch=1、1024→32 tokens、CUDA Graph、10 次中位数。所有在线旋转、量化、cache 更新与 token 反馈均计入，测量边界沿用此前报告；模型加载和离线搜索/GPTQ 不计入推理时间。

| 模式 | wall ms | 相对 FP16 加速 | KV GiB | 峰值 allocated GiB | reserved GiB |
|---|---:|---:|---:|---:|---:|
| fp16 | 553.35 | 1.000× | 0.515 | 13.235 | 13.389 |
| sym_search_kv4 | 207.83 | 2.662× | 0.137 | 3.897 | 4.293 |
| sym_search_kv16 | 211.55 | 2.616× | 0.515 | 4.245 | 4.436 |
| sym_gptq_kv4 | 208.20 | 2.658× | 0.137 | 3.897 | 4.293 |
| sym_gptq_kv16 | 211.41 | 2.617× | 0.515 | 4.245 | 4.436 |

RTN/GPTQ 使用相同运行时格式；GPTQ 不增加在线操作。两者延迟基本相同。allocated 与 reserved 含义不同，以上 reserved 为本轮 allocator/graph pool 状态，不应把相对旧进程的差异归因于量化方法。

### 缓存解码限制与测试

KV4 仍有前节记录的浮点归约敏感性：同输入 attention oracle 通过，并不等于整模型 cached decode 与 dense 参考逐 token 等价。额外 256-token teacher-forced 缓存诊断如下，不能混同完整测试集 PPL：

| 配置 | KV 位宽 | CUDA / 参考缓存 PPL | 预测一致率 | 同输入 attention 最大相对 L2 |
|---|---:|---:|---:|---:|
| sym_search | 4 | 7.731109 / 7.621565 | 87.89% | 0.015587% |
| sym_search | 16 | 7.573966 / 7.573966 | 100.00% | 0.000000% |
| sym_gptq | 4 | 7.142639 / 7.141445 | 88.28% | 0.015263% |
| sym_gptq | 16 | 7.045854 / 7.045854 | 100.00% | 0.000000% |

KV16 在本次缓存诊断中与参考完全一致；KV4 尚不能宣称与独立 dense 后端的整模型输出等价。搜索 RTN 的完整 prefill PPL 改善，也不代表其每个小样本缓存窗口 PPL 都改善。

28 项回归测试通过（`benchmarks/ptq_llama2/tests.log`）；新增 3 项 PTQ 测试的 CUDA memcheck 为 0 errors（同目录 `memcheck.log`）。GPTQ Triton block 与独立逐列 PyTorch oracle 对照，包含 dead feature 和对角 Hessian 退化为 RTN 的检查；另覆盖对称权重 GEMM/GEMV、搜索 clipping 的完整 block 与真实 KV 打包。

### 复现与产物

实现：`duquant_cuda/ptq.py`；搜索：`benchmarks/search_ptq.py`；评测：`benchmarks/run_ptq_eval.py`；总表：[`benchmarks/ptq_llama2_final/summary.json`](../benchmarks/ptq_llama2_final/summary.json)。模型 codes/scale/zero 位于被 git 忽略的 `benchmarks/cache/ptq_llama2_final/`。日志记录源码、父校准、导出层与数据 hash。

```bash
# 沿用前述 Python/CUDA 环境；选择空闲 GPU。新实验使用新的输出目录。
"$DUQUANT_PYTHON" benchmarks/search_ptq.py --method rtn --output benchmarks/cache/ptq_reproduced --report benchmarks/ptq_reproduced
"$DUQUANT_PYTHON" benchmarks/search_ptq.py --method gptq --output benchmarks/cache/ptq_reproduced --report benchmarks/ptq_reproduced
"$DUQUANT_PYTHON" benchmarks/run_ptq_eval.py --variant sym_gptq --calibration-root benchmarks/cache/ptq_reproduced --root benchmarks/ptq_reproduced
# 另外三组 variant: sym_fixed / sym_wsearch / sym_search。
# 推理必须显式给出 --kv-bits 4 或 16。
"$DUQUANT_PYTHON" benchmarks/bench_model.py --model /mnt/public_models/llama2-7b-hf --mode cuda --calibration benchmarks/cache/ptq_llama2_final/sym_gptq --kv-bits 4 --prompt 1024 --new-tokens 32 --repeats 10 --output benchmarks/ptq_gptq_recheck.json
```

<a id="qa-int4"></a>
## LLaMA-2 7B：六任务真实 INT4 对照

三种量化模型全部使用真实 W4A4 整数 GEMM、packed KV4 和同一计分程序。
W 对称 per-channel，A 非对称 per-token，Q16，K/V 非对称 per-head group=128；
共享前节的 clipping 搜索网格、重构 MSE 目标和 true-sequential GPTQ 规则。
搜索/GPTQ 使用完全相同的 128×2048 WikiText-2 train token 序列。
DuQuant 沿用无 LWC 的 `sym_gptq`；QuaRot 用 seed=0 的随机 Hadamard；
SpinQuant 使用官方 W16A4KV4 seed=0 学习旋转，再套用共同的 W4 GPTQ。
SpinQuant 旋转学习本身的数据和预算单独来自官方 checkpoint，不能称为三者整个优化预算相同。

完整 zero-shot 数据集，共 19,965 道题、67,106 个候选答案，无截断：
PIQA 1838、ARC-E 2376、ARC-C 1172、BoolQ 3270、HellaSwag 10042、WinoGrande 1267。
使用仓库旧版 lm-eval 的 prompt/独立 context-continuation tokenization，无 BOS/EOS。
ARC-C、HellaSwag 汇报按答案字符长度归一化的 accuracy，其余为 raw accuracy；平均为六项算术平均。

| 模型/方法 | PIQA | ARC-E | ARC-C | BoolQ | HellaSwag | WinoGrande | 平均 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 / KV16 | 78.40 | 69.32 | 40.61 | 71.10 | 72.95 | 67.09 | 66.58 |
| QuaRot W4A4KV4 | 74.97 | 67.42 | 38.74 | 68.44 | 68.35 | 63.61 | 63.59 |
| SpinQuant W4A4KV4 | 74.48 | 65.57 | 37.80 | 69.57 | 66.95 | 62.35 | 62.79 |
| DuQuant W4A4KV4 | 76.50 | 68.48 | 37.71 | 72.02 | 70.33 | 64.64 | 64.95 |

本设置下 DuQuant 平均比 QuaRot 高 1.36 个百分点、比 SpinQuant 高 2.16 个百分点，
比 FP16 低 1.63 个百分点。以上是统一后端与量化策略的对照；不能作为各论文默认配置的复现结论，
也不能泛化为各方法在其他 clipping、GPTQ、训练预算或评测协议下的排名。

DuQuant 实际推理计分约 263.7 秒，QuaRot 455.7 秒，SpinQuant 434.6 秒，FP16 657.5 秒。
这些是不同空闲 GPU 上本次评测的运行记录，含编译、padding 和 Python 调度，未含加载，
不作为受控推理加速比。比较器的 QuaRot/SpinQuant 线性层仍经过两个恒等 DuQuant 旋转步骤，
未对其在线变换做专门性能优化。没有运行全量 fake-quant QA。

验证包括：两个新增 QA 测试通过（独立 Hugging Face 计分/右 padding、比较器 INT4 与独立整数 oracle）；
两种比较器各 32 层的浮点旋转等价性检查；原始 7B FP16 的 12 题/36 候选 Hugging Face 抽样复核，
raw/normalized 最终选择全部一致，累计 log-likelihood 最大差 0.023634。
所有正式评测采用完整 causal prefill，前述 KV4 cached-decode 数值限制仍保留。

完整 [协议与复现](../benchmarks/qa_compare/README.md)、[汇总 JSON](../benchmarks/qa_compare/summary.json)
及 [表格](../benchmarks/qa_compare/table.md)。逐题预测、raw/normalized 两种 accuracy、源码/校准/数据 hash
均已保存。`benchmarks/summarize_qa.py` 只有在全部结果完整且协议一致时才会生成汇总。

### 相同配置追加 FlatQuant

保留官方 LLaMA-2 7B W4A4KV4 的可逆非正交变换、对角缩放和 Q/K/V 变换，
替换其预训练 clipping 为本次共同搜索，再运行相同 GPTQ。模型、训练 token、
六任务数据、计分源码和 batch 设置均与上表一致，使用真实 INT4 GEMM 和 packed KV4。
本次扩展没有更改前三种量化方法或原始计分代码。

| 方法 | PIQA | ARC-E | ARC-C | BoolQ | HellaSwag | WinoGrande | 平均 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FlatQuant W4A4KV4 | 77.37 | 69.02 | 38.31 | 68.90 | 70.78 | 65.90 | 65.05 |

FlatQuant 平均比 DuQuant 高 0.10 个百分点，数值非常接近；比 QuaRot 高 1.46、
比 SpinQuant 高 2.26、比 FP16 低 1.53 个百分点。它在其中五项更高，DuQuant 的 BoolQ 高 3.12 个百分点。
官方变换训练本身的预算与配置保持原样，以上不等于原论文默认量化设置的复现。

四项新增检查通过；全部 32 层浮点等价性检查通过（最大相对 L2 0.00185065）；
32 个候选答案的整模型 INT4/独立整数参考累计 log-likelihood 完全一致，最大差为 0。
实际六任务计分耗时 256.4 秒（GPU 0，不含加载）；与上轮 DuQuant 的 263.7 秒接近，
两次评测运行记录不能替代受控性能基准。

完整 [FlatQuant 协议与结果](../benchmarks/flatquant_compare/README.md)、
[五方法总表 JSON](../benchmarks/flatquant_compare/summary.json)。

<a id="qwen3-comparison"></a>
## Qwen3-1.7B-Base：同一 W4A4KV4 协议

本地适配 Qwen3 的 head-wise Q/K RMSNorm、GQA 和 tied embedding/head，
使用 FP16 基线（原始 BF16 权重转换）和真实整数 W4A4/packed KV4。
完整六任务共 19,965 题、67,106 候选，无 chat template；
ARC-C/HellaSwag 使用 normalized accuracy，其余为 raw accuracy。
三方法共享 128×2048 WikiText-2 train、同样的 clipping 搜索和 true-sequential GPTQ。

| 方法 | PIQA | ARC-E | ARC-C | BoolQ | HellaSwag | WinoGrande | 平均 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 75.90 | 73.78 | 45.05 | 79.08 | 66.52 | 64.17 | 67.42 |
| QuaRot | 68.61 | 64.39 | 37.80 | 65.96 | 55.86 | 53.83 | 57.74 |
| SpinQuant | 68.61 | 67.63 | 38.99 | 66.88 | 55.40 | 56.35 | 58.98 |
| DuQuant | 70.73 | 72.18 | 41.72 | 72.51 | 58.86 | 56.35 | 62.06 |

DuQuant 高于 QuaRot 4.32、高于 SpinQuant 3.08 个百分点，低于 FP16 5.36 个百分点。
这是本地适配结果，不等于作者官方 Qwen3 结果。SpinQuant 使用原始 Cayley SGDG，
W16A4KV4 STE causal loss，800 步，batch=1、长度2048、lr=1.5 cosine，
循环同一批128训练样本；其学习数据/并行训练配置与原论文不同。
DuQuant 使用原始双旋转/permutation、256步搜索、alpha=.6 smoothing，无LWC。

单层与 Hugging Face 对照通过；原生完整模型抽样六题/18候选的最终选择一致；
三种完整量化模型各四候选与独立整数参考计分完全一致。
SpinQuant 最终正交性最大偏差 1.19210e-6；QuaRot/SpinQuant全部28层变换等价性检查通过。
本次计分耗时 FP16/QuaRot/SpinQuant/DuQuant 分别为164.9/200.5/181.6/198.0秒，
包含未融合变换、编译、padding和调度；此评测路径没有体现Qwen3上的INT4加速，不能沿用LLaMA性能结论。

完整[实验协议、支持情况与复现](../benchmarks/qwen3_compare/README.md)、
[结果JSON](../benchmarks/qwen3_compare/summary.json)。此前KV4 cached-decode数值限制仍适用。

### Qwen3 同配置追加 FlatQuant

本地训练 FlatQuant 的非正交 SVD/Kronecker 变换、对角缩放及匹配 Q/K/V/O 变换，
使用官方模块，适配 Qwen3 的 Q/K norm 和 GQA。每层15 epochs、batch4、
相同128×2048训练样本；训练阶段包含LWC/LAC，导出后替换为共同clipping搜索和GPTQ。
各方法的变换学习预算不同，不能称为相同训练预算的比较，也不是作者官方Qwen3结果。

| 方法 | PIQA | ARC-E | ARC-C | BoolQ | HellaSwag | WinoGrande | 平均 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FlatQuant W4A4KV4 | 71.55 | 69.32 | 40.61 | 70.34 | 63.05 | 60.06 | 62.49 |

相对DuQuant平均高0.43个百分点，双方各三项更高；相对FP16低4.93个百分点。
28层浮点变换检查通过（最大相对L2约0.2603%）；三项新增测试通过；
32候选的整模型INT4与独立整数参考累计log-likelihood完全一致，最大差0。
完整真实INT4计分171.0秒，不含加载，包含编译和调度，不作为受控速度比较。
详见[FlatQuant Qwen3报告](../benchmarks/qwen3_compare/flatquant/README.md)与
[五方法结果JSON](../benchmarks/qwen3_compare/flatquant/summary.json)。

## 两模型追加 LAMBADA、WikiText-2 与 C4 PPL

补齐 FlatQuant 默认任务构成中此前缺少的英文 `lambada_openai` test（全部5153条）。
原先测的是BoolQ，FlatQuant默认列表用LAMBADA替换它。所有方法沿用此前checkpoint，
不重新学习变换、搜索clipping或做GPTQ；量化行均为真实W4A4KV4。
同时测完整WikiText-2 test的2048-token完整块，以及FlatQuant代码的C4-new验证切片：
验证首分片前1100篇文本以单空格拼接，取256×2048 tokens；C4不是整个语料库。

| 模型 | 方法 | LAMBADA acc % | LAMBADA末词PPL | WikiText-2 PPL | C4 PPL | 替换BoolQ后六任务均分 % |
|---|---|---:|---:|---:|---:|---:|
| LLaMA-2 7B | FP16 | 73.92 | 3.395 | 5.472 | 7.263 | 67.05 |
| LLaMA-2 7B | QuaRot | 70.83 | 4.084 | 6.430 | 8.875 | 63.99 |
| LLaMA-2 7B | SpinQuant | 69.92 | 4.509 | 6.666 | 9.118 | 62.85 |
| LLaMA-2 7B | DuQuant | 72.56 | 3.702 | 5.891 | 8.011 | 65.04 |
| LLaMA-2 7B | FlatQuant | 72.91 | 3.576 | 5.774 | 7.799 | 65.72 |
| Qwen3-1.7B-Base | FP16 | 63.77 | 5.609 | 9.388 | 15.408 | 64.86 |
| Qwen3-1.7B-Base | QuaRot | 48.11 | 13.037 | 15.200 | 26.694 | 54.77 |
| Qwen3-1.7B-Base | SpinQuant | 48.55 | 13.349 | 14.180 | 24.198 | 55.92 |
| Qwen3-1.7B-Base | DuQuant | 51.52 | 10.820 | 12.361 | 20.995 | 58.56 |
| Qwen3-1.7B-Base | FlatQuant | 57.31 | 8.068 | 11.547 | 19.527 | 60.32 |

LAMBADA准确率要求末词所有subword均为对应位置贪心预测；末词PPL以5153个词为分母，
并另存target-token PPL。使用全文与context编码的严格前缀/后缀切分，防止单独编码
SentencePiece前导空格产生额外目标token；两模型全部5153条边界通过检查。
WikiText/C4以shifted token为分母；不同PPL列不能混同。

新的六任务均分对齐的是任务构成，保留原先五项QA的冻结提示词和旧版独立context/choice编码，
不等于FlatQuant官方整套评测协议复现。原BoolQ六任务均分、七任务均分也保存于汇总。
本组中FlatQuant相对DuQuant的新六任务均分在LLaMA-2高约0.68、Qwen3高约1.76个百分点，
同时两模型的WikiText/C4 PPL更低。变换训练预算和原先Qwen3本地适配限制继续适用。

新增末词计分测试通过；LLaMA-2 FP16/DuQuant WikiText PPL精确复现此前5.472090/5.890531。
全部10组LAMBADA/WikiText和10组C4完成且通过完整性、源码/校准/数据hash、逐题预测、
计分token分母校验。详见[完整报告](../benchmarks/lambada_compare/README.md)与
[汇总JSON](../benchmarks/lambada_compare/summary.json)。
