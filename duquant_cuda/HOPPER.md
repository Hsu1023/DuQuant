# Hopper（sm_90 / GH200）后端

同一套 `DuQuantLinearCUDA` 接口在 Hopper 上自动切换到本页描述的 kernel，量化格式、checkpoint 与 Ada 路径完全相同。分派只依赖设备 compute capability 和形状。

## 为什么需要单独的 Hopper 路径

Hopper 没有 INT4 Tensor Core（Ada 路径依赖的 `IMMA.16864.U4.U4` 在 sm_90 上不存在），但有 INT8 `wgmma`。因此：

- **权重**仍以 packed u4 存放（显存占用不变），在 kernel 内从 shared memory 读入寄存器后用 `__byte_perm` 拆成 s8（0..15），作为 `wgmma` 的寄存器操作数（CUTLASS sm90 mixed-input，RS 模式）。
- **激活**由量化 kernel 直接输出 s8 码（0..15，一字节一个），作为 `wgmma` 的 shared-memory 操作数。
- 整数累加结果与 Ada 的 u4×u4 路径逐位相同；零点修正、两个 scale 与 bias 在 epilogue 中按与 Ada 相同的表达式和运算顺序计算（CUTLASS sm90 EVT）。

## 实现

| 文件 | 内容 |
|---|---|
| `gemm_sm90.cu` | INT8 wgmma GEMM（TMA + warp-specialized cooperative）。`wgmma` 的寄存器操作数必须是 A，所以问题被转置：`D'[N,M] = W[N,K]·X[M,K]^T`，列主序 `D'` 即行主序 `Y[M,N]`。token 维 tile 16/32/64/128/256（data-parallel）与 16/32/64/128（Stream-K）共 9 个配置。|
| `linear_small.cuh` | decode（M ≤ 4）时整个 linear 一次 cooperative launch：见下节。|
| `small_m.cuh` | 小 M 的 dp4a u4 GEMV 与公共工具函数。|
| `backend.py` | 设备检测、编译（`-arch=sm_90a`）、形状分派。|
| `kv4.py` | Hopper 上 MHA 的 KV4 decode 使用 16-token tile / 2 warps（128×128 tile 在 sm_90 上严重溢出寄存器）。|

### decode linear（M ≤ 4）：单次 launch

原路径每个 linear 4 次 launch（两次旋转、量化、GEMV），在 H100 级带宽下固定开销占主导。新 kernel 一次完成：

1. worker CTA b（每个 128 宽的块一个）直接由 shared memory 中的整行 x 计算本块需要的 `t[perm[128b+i]]`（与块旋转同样的计算量，省去 t 的全局往返和一道屏障），再乘 R2 得到 z；发布每块的行 min/max。
2. 屏障 1 之后，按 `backend._quant` 的 FP16 舍入序列计算行 scale/zero，量化打包本块，发布 codes 与块内 code 和。除法 `div_rn` 用倒数乘法快速路径实现，只有商落在 .5 附近时才回退到 IEEE 除法，`nearbyint` 结果与 `div_rn` 逐位一致。
3. 屏障 2 之后，全部 CTA 做持久化 dp4a GEMV（两条 nibble 平面直接 `dp4a`，无需解包）和融合 epilogue。
4. 权重不依赖激活：每个 CTA 在屏障前就把自己第一组权重读进寄存器、并把更多权重预取进 L2；本层流式读取的权重使用 `evict_first` 策略，同时把**下一个** linear（模型按 qkv→o→gate_up→down 顺序串起）的旋转矩阵、置换和权重开头预取进 L2（`evict_last`）。

屏障是 64 位单调计数器（从不复位，launch epoch 由到达计数推出），每个（M 桶、配置、是否带 GEMV）一个计数器槽，因为 epoch 依赖 grid 大小。cooperative launch 只用来保证所有 CTA 同时驻留。

两次旋转都在 Tensor Core 上计算（`mma.m16n8k16`，FP32 累加，k 升序），与 `tl.dot`/cuBLAS 参考路径的舍入逐位一致。第一次旋转只算本块需要的列：每个 mma 的 n 维取 8 个连续输出，16 行放"各输出的源块 × token"，只保留对角元素。早期版本用标量 FMA 顺序累加，会在 2–4% 的行上让一个激活码翻转（相对误差 ~0.5%），已弃用。

## 正确性

- `tests/test_cuda_layer.py` 新增：
  - s8 激活码与 packed 码逐位一致；
  - sm90 GEMM（9 个配置，含尾块）与 float64 参考逐位相等；
  - 小 M 路径的激活码、scale、zero、code 和与 Triton 路径逐位相等，GEMV 与整数参考一致。
- 2000 行随机输入（K=4096/11008）上，小 M 路径与 torch 参考的激活码 0 差异。
- 测试结果：除需要校准 checkpoint 的测试外全部通过。`test_cuda_layer.py` 原来会全局关闭梯度并泄漏到其他测试模块，已改为在 `tearDown` 中恢复；`test_flatquant.py` 有同样的问题，未改。

## 校准与质量（GH200 上重新校准）

在 GH200 上从头跑了 LLaMA-2-7B 的完整校准流程：calibrate（55 s）→ RTN/GPTQ clipping 搜索（3 + 3 min）→ 重建训练（3 个 shard 在同一张卡上并发，48 min）→ 导出。
- 训练 token 与 4090 那次逐位相同（哈希、采样起点一致）。
- 各层校准文件因浮点不可复现而不同。
- 训练 MSE 与 4090 记录相近。
- 产物在 `benchmarks/cache/`（git ignore），报告在 `benchmarks/hopper/calibration/`。

WikiText-2 test，seqlen 2048，全部 166 块（`benchmarks/eval_hopper_reconstruct.py`，token 哈希与仓库记录的协议一致）：

| 模型 | 4090 (Ada) | GH200 (Hopper) |
|---|---:|---:|
| FP16 | 5.4721 | 5.4721 |
| 重建 RTN | 5.8179 | 5.8170 |
| 重建 GPTQ | 5.7909 | 5.8002 |

两个 export 在前 8 个块上：CUDA 后端与独立整数参考的逐块 NLL **完全相同**（差为 0）。

LAMBADA 与六任务 QA 未复测：它们的数据准备脚本依赖 4090 机器上 `/mnt/yichen` 下的原始数据。

## 测量（GH200 120GB，PyTorch 2.9.1+cu128，Triton 3.5.1，CUDA 12.9，CUTLASS 3.9.2）

### 端到端（与 README 同口径）

口径与 Ada 表完全一致：同一脚本与参数，逐字段核对过 `scope`/`excluded`/显存统计字段与采样数。
- LLaMA-2-7B，batch 1，1024-token prompt + 生成 32 token，含全部 32 层和 LM head。
- CUDA Graph，10 次中位数；总计取 wall time（与 README 表相同）。
- 汇总见 `benchmarks/hopper/performance/summary.json`。

| | prefill | decode / token | 总计 (wall) | 加速 | 模型存储 | 峰值 allocated |
|---|---:|---:|---:|---:|---:|---:|
| FP16 | 23.45 ms | 5.06 ms | 180.25 ms | — | 12.55 GiB | 13.28 GiB |
| W4A4KV4 重建 RTN | 19.02 ms | 3.93 ms | 140.73 ms | **1.28×** | 3.55 GiB | 3.98 GiB（−70.1%）|

模型存储比 Ada 的 3.54 GiB 多 11.16 MiB，其中：
- 11.09 MiB 是 Hopper 专用的 R1ᵀ/R2ᵀ 填充副本与 int32 置换；
- 0.07 MiB 是 decode kernel 的屏障计数器。

生成的 token 与 FP16 相同。

与 4090（2.66×）相比加速比小得多，原因有两点：
- GH200 的 FP16 基线本身很强：decode 每 token 5.06 ms，而 4090 是 14.9 ms。
- W4A4 每个 linear 的激活变换（两次旋转、置换、逐 token 量化）是延迟主导的固定开销，在 H100 级带宽下占比很高。

decode 时每层 4 个 linear 合计约 88 µs（冷 L2），而权重流按带宽只需约 30 µs。

### 单层（LLaMA-2 第一层真实权重，`benchmarks/hopper/single_layer_final.json`）

热缓存 / 冷缓存（每次前清 256 MiB）对 FP16 的加速比：

| K → N | M=1 | M=16 | M=128 | M=2048 |
|---|---|---|---|---|
| 4096 → 4096 | 0.79 / 1.15 | 0.48 / 0.80 | 0.55 / 0.69 | 1.24 / 1.00 |
| 4096 → 11008 | 1.63 / 1.57 | 1.44 / 1.34 | 1.24 / 1.17 | 1.61 / 1.23 |
| 11008 → 4096 | 1.60 / 1.49 | 0.94 / 1.00 | 0.89 / 0.90 | 1.22 / 1.04 |

GEMM 本身在大 M 时约 1.2–1.4 POPS（FP16 cuBLAS 的 2.1–2.4 倍）。

中等 M 的配置扫描见 `benchmarks/hopper/gemm_config_sweep.json`：Stream-K 对 down 投影（输出 tile 少、K 长）有效，M≤16 时 37.6→30.6 µs，M=128 时 53.9→44.0 µs。

## 已知限制 / 后续

- 中等 M（2–512）时 W4A4 仍慢于 FP16：激活变换走 3 个 Triton kernel，且 CUTLASS mixed-input GEMM 在小 tile 下的固定延迟约 12 µs。
- decode linear 的激活阶段约 9–11 µs，主要是延迟：冷 R 矩阵装载 2.7–5 µs，加上两道全局屏障。可能的方向：
  - 用 thread block cluster + DSMEM 取代全局屏障；
  - 用 PDL 与上一个 kernel 重叠序言；
  - 用 L2 persistence 常驻全部旋转矩阵（8 MB）。
- prefill 的激活变换（M=2048 时 40 µs，K=11008 时 117 µs）仍是分开的 3 个 kernel。
- KV4 decode attention 在 1k 上下文时约 14 µs/层，带宽利用很低，值得重写成 CUDA。

## 复现

```bash
source /work/nvme/bfne/duquant/env_hopper.sh  # conda env、CUDA 12.9、CUTLASS、模型/数据路径
python -m unittest discover -s tests
python benchmarks/bench_single_layer.py --model $DUQUANT_LLAMA2 --tokens 1,2,4,16,128,512,2048 --cold --output benchmarks/hopper/single_layer_final.json
python benchmarks/bench_model.py --mode fp16 --kv-bits 16 --prompt 1024 --new-tokens 32 --repeats 10 --output benchmarks/hopper/performance/fp16.json
PYTHONPATH=benchmarks python benchmarks/bench_llama_reconstruct.py --mode cuda --kv-bits 4 --prompt 1024 --new-tokens 32 --repeats 10 --output benchmarks/hopper/performance/rtn_kv4.json
python benchmarks/eval_hopper_reconstruct.py --output benchmarks/hopper/reconstruct_wikitext.json
```

校准流程（单卡即可）：`slurm_scripts/calib_stage12.sh` 与 `slurm_scripts/calib_stage3.sh`（位于仓库外的 `/work/nvme/bfne/duquant/`），对应仓库内 `calibrate_model.py` → `search_ptq.py --method rtn/gptq` → `prepare_llama_reconstruct.py` → `train_llama_reconstruct.py --shard 0/1/2` → `finish_llama_reconstruct.py`。
