# Calibrated W4A4: accuracy, integer-reference consistency, speed and memory

The data-driven dual rotations, zigzag permutation and smoothing of the original repository are now wired in. Three numeric differences that later quantization would amplify have been fixed. The real INT4 WikiText-2 PPL of LLaMA-2-7B drops from **8.3593 to 6.1638**, a **26.26%** reduction; full-model generation is still **2.52–2.57×** faster than FP16.

The results without LWC come first. The [LWC follow-up](#lwc-follow-up) lowers the real INT4 PPL further to **6.0641**, with an end-to-end speedup of **2.51–2.55×**. The latest [five-family validation](#model-family-validation) covers representative LLaMA-1/2/3, Vicuna and Mistral models, all without LWC.

## Calibration parameters

`benchmarks/calibrate_model.py` draws 128 samples of 2048 tokens from WikiText-2 **train**, seed=2. The test split was not used for calibration, parameter search or kernel numeric fixes; the numeric diagnostics also use train text.

1. Propagate the FP16 calibration inputs layer by layer and collect the per-channel absolute maximum of each projection input.
2. Fold four groups of smoothing scales with the zero-shift rule of the original `smooth_and_let_inplace`, alpha=0.6. The QKV and gate/up scales fold into the norms and weights; the O and down scales fold into the V/O and up/down weights respectively. No online scaling kernel is introduced.
3. Call the original `UniformAffineQuantizer.online_duquant_cali` directly: at most 256 steps per rotation, block size=128, permutation_times=1. The representative input for the rotations is the per-position mean of the calibration embeddings, propagated layer by layer. The base rotation is built with the Givens construction of the original `get_rot.py`, generating only the required 128-dimensional version.
4. Export the per-layer smoothing scales, both rotations and the permutation. At load time, the model config, manifest and per-layer file hashes are checked; then the original weights are folded and packed to INT4.

W4A4 uses weight clip=0.8, activation clip=0.9, Q/K clip=0.9, V clip=1.0. No LWC/LET is trained. This **wires the original repository's calibration algorithm into the existing CUDA backend**. Attention uses this backend's Flash SDPA and QK/V quantization path. It is neither a line-by-line reproduction of the eager attention in the original `main.py` nor a downloaded paper checkpoint.

The parameter bundle is in `benchmarks/cache/calibrated_w4a4/`, about 16 MiB, and git-ignored. A reviewable copy of the manifest is `benchmarks/calibrated/calibration_manifest.json`. It records the training data and token hashes, the 128 sampling starts, the parameters and the per-layer file SHA256. Manifest SHA256:

`7252d2f0c71598d66e95df9343b08788c4fd8a5d56180e4fcddc3a088bb5a39b`

## Full PPL

Same protocol as the first evaluation: WikiText-2 raw test, slow LlamaTokenizer, full text joined with double newlines; all 166 complete non-overlapping 2048-token blocks, **339,802 tokens** actually scored. Each block starts at position=0, FP32 next-token CE, with the actual number of scored tokens as the denominator. Each mode runs in its own process.

| mode | PPL ↓ | next-token accuracy ↑ |
|---|---:|---:|
| FP16 baseline, final code | 5.472090 | 61.0182% |
| calibrated fake W4A4 | 6.156414 | 58.8587% |
| calibrated real INT4 CUDA | 6.163763 | 58.8384% |
| calibrated independent integer-arithmetic reference | 6.163763 | 58.8384% |

CUDA PPL is **0.007349 (0.1194%)** above fake. CUDA PPL is still **12.64%** above FP16, so this is not lossless. These are PPL and next-token metrics; no QA was run.

## The two references and the numeric fixes

The old-style fake path dequantizes activations and weights element-wise and rounds them to FP16 before a floating-point GEMM. The integer reference represents the exact 4-bit codes in FP32, computes the integer dot products and zero-point correction, then applies the scales. Here K≤11008, so the integer dot products are bounded by `225*K < 2^24` and FP32 represents them exactly. It is independent of the INT4 kernel and uses PyTorch norm, Hadamard and SwiGLU.

With the same parameters, old-style fake and real INT4 have different rounding semantics. Here the full-set top-1 agreement is **86.4854%**, and the sampled logits still differ by 27.03% relative L2. Close average PPL cannot be read as bit equivalence with the old-style fake output.

Per-operator diagnostics located and fixed:

- **RMSNorm**: reproduce the four accumulators and warp-shuffle reduction order of the current PyTorch FP32 mean, and disable multiply-add fusion to keep the same FP16 rounding.
- **Q/K Hadamard**: the fused kernel now uses a Tensor Core dense dot, matching the accumulation order of the reference GEMM and avoiding rounding-boundary differences from the butterfly reduction.
- **SwiGLU**: use CUDA libdevice exp and correctly rounded division, matching PyTorch's exponential and division semantics.

Before the fixes, the full-layer relative L2 differences of the first four layers on the same training input were 1.2894%, 0.2923%, 0.3490% and 0.5684%. After the fixes, these four full-layer outputs are bit-identical to the integer reference. The raw diagnostic data are `benchmarks/numeric_probe_train_{before,rms,rms_hadamard}.json`; the final data are `benchmarks/numeric_probe_train.json`.

The **full 32-layer CUDA model versus the integer reference** was further validated on every test block:

- Top-1 agreement over all 339,802 scored positions is **100%**.
- The NLL difference is **0** for all 166 blocks, so the final PPL is identical.
- For 64 sampled positions in each of the first four blocks (256 positions), the logits over the whole vocabulary are **bit-identical**; relative L2 and KL are both 0.

Full logits for every position of the whole set were not saved, so the whole-model bit-identity claim on logits covers only these 256 sampled positions. The SHA256 hashes of the data, rotations, permutations and dequantized weights were also checked to match.

## End-to-end speed and memory of the final code

RTX 4090 (about 48 GiB on this machine), GPU 4, batch=1, each configuration in its own process, median of 5 measurements. Full 32 layers, autoregressive token feedback, real KV updates, a fixed 32 generated tokens; wall time from the GPU token input until the generated IDs are copied back to the CPU. Includes online rotation, quantization, packing, INT4 compute, norms, attention, LM head and argmax; excludes loading, export, tokenization, first compilation and CUDA Graph capture. Both modes use the same fused nonlinear operators and Flash SDPA.

| input → output tokens | FP16 ms | calibrated INT4 ms | speedup | FP16 decode tok/s | INT4 decode tok/s |
|---|---:|---:|---:|---:|---:|
| 128 → 32 | 461.19 | 179.11 | **2.57×** | 69.9 | 180.5 |
| 2048 → 32 | 656.90 | 260.93 | **2.52×** | 65.0 | 150.6 |

GPU prefill time goes from 17.84 → 7.37 ms and 179.89 → 54.92 ms respectively. GPU clocks were not locked; these are still fixed-shape CUDA Graph requests, not a dynamic serving benchmark.

Memory in GiB. Peak allocated includes warmup, capture and inference temporaries, and excludes offline loading/export and HF validation. Reserved includes the allocator and graph pools. Device usage includes the CUDA context and other overheads. The detailed accounting is the same as in [MODEL.md](MODEL.md).

| input length | mode | model tensors | KV | peak allocated | reserved | device usage |
|---|---|---:|---:|---:|---:|---:|
| 128 | FP16 | 12.55 | 0.08 | 12.67 | 12.68 | 13.26 |
| 128 | INT4 | 3.54 | 0.08 | 3.65 | 4.73 | 5.41 |
| 2048 | FP16 | 12.55 | 1.02 | 13.88 | 14.02 | 14.60 |
| 2048 | INT4 | 3.54 | 1.02 | 4.93 | 5.12 | 5.81 |

KV, embedding, LM head and attention arithmetic are still FP16. Model tensors shrink by about 71.8%; this should not be read as a theoretical 4× compression of whole-device memory.

## Checks and reproduction

14 tests pass. They include the new smoothing-folding equivalence test, multi-shape/residual bit-identity checks of RMSNorm, and the full-layer integer-reference test tightened to a bit-exact comparison. CUDA memcheck reports **0 errors**. Logs are in `benchmarks/calibrated/{tests,memcheck}.log`. The short-prompt logits of the final FP16 model differ from HF by about 0.00698% relative L2, with the same top-1.

```bash
export CUDA_VISIBLE_DEVICES=4
export CUDA_HOME=/usr/local/cuda-12.4
export CUTLASS_PATH=/mnt/yichen/wyc/QDLM/QuaRot/third-party/cutlass
export TRITON_CACHE_DIR=/tmp/duquant_triton
export DUQUANT_PYTHON=/home/yichen/miniconda3/envs/hb/bin/python

# Choose a new directory when regenerating parameters; the script refuses to overwrite an existing manifest.
"$DUQUANT_PYTHON" benchmarks/calibrate_model.py --output benchmarks/cache/calibrated_reproduced

# Re-evaluate with the fixed exported parameters used here.
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

Each command starts and exits its own process. The official results are `benchmarks/calibrated/ppl_summary.json` and `model_summary.json`. The first uncalibrated results remain in the parent `benchmarks/`; do not mix them.

## LWC follow-up

Learnable Weight Clipping is trained on top of the fixed calibration parameters above. Full 32-layer training and evaluation are complete; results are in `benchmarks/lwc/`, parameters in `benchmarks/cache/lwc_w4a4/`.

### Training setup and implementation

- Reuses the same 128 × 2048 tokens of WikiText-2 **train**, seed=2, and checks the SHA256 of the data file, the tokens and the parent calibration manifest.
- 20 epochs per layer, batch=1, AdamW, lr=0.01, weight decay=0. Each output channel learns its own sigmoid coefficients for the lower and upper bounds, with logits initialized to 4.0. The fake output of each trained layer feeds the next layer; the target is the output of the corresponding layer of the original FP16 model, and the loss is FP32 MSE.
- The existing smoothing (alpha=0.6), rotations, permutations and model weights are fixed; LET is not trained. Activation, Q/K and V clipping stay at 0.9, 0.9 and 1.0. LWC replaces the former fixed weight clip=0.8.
- The clipping logits are FP32; the sigmoid coefficients are converted to FP16 before quantization, so the forward matches the deployed FP16 quantizer; rounding uses STE. After each layer is exported, the training forward is checked to equal the fake-inference output exactly before the quantized inputs are propagated.
- AMP starts with scale=1024. When gradient overflow is detected, the scale is lowered and the same sample is rerun; skipped updates are not counted as training steps. There were 3 retries in total, all in layer 2.
- On GPU 0, the first 8 layers used the plain PyTorch weight STE. The remaining 24 layers were resumed with the fused Triton weight STE. Files and hashes of completed layers are unchanged; the teacher/fake inputs are replayed on resume. The cumulative per-layer time is **72.38 minutes**, excluding development validation and the replay overhead of resuming.

The fused weight STE keeps the FP16 rounding points and the ATen gradient reduction order. In an independent comparison on the real first layer with 128 samples and 2 epochs, the final logit parameters, clipping coefficients and per-epoch MSE are identical. The layer time drops from 22.22 to 15.04 seconds. The data are in `fused_training_check.json`. This only speeds up training; inference still uses the existing packed INT4 kernels, and the learned clipping takes effect during offline packing.

Parameter manifest SHA256:

`8c6a810d52eb4c3d37140569926637a954c241d8ac38f832f011a94f4807dc63`

The reviewable manifest and the per-layer training records are `benchmarks/lwc/calibration_manifest.json` and `training.json`. The training code is `benchmarks/train_lwc.py`, `duquant_cuda/lwc.py` and `lwc_quant.py`.

### Full PPL and numeric checks

All 166 2048-token test blocks are used again, with 339,802 scored tokens; the protocol is the same as above. The FP16 PPL reuses the unchanged baseline result. LWC fake, CUDA and the integer reference were re-evaluated in separate processes on GPUs 0, 1 and 2.

| mode | PPL without LWC | PPL with LWC | next-token accuracy with LWC |
|---|---:|---:|---:|
| FP16 | 5.472090 | 5.472090 (baseline) | 61.0182% |
| fake W4A4 | 6.156414 | **6.065405** | 58.9793% |
| real INT4 CUDA | 6.163763 | **6.064058** | 59.0394% |
| independent integer-arithmetic reference | 6.163763 | **6.064058** | 59.0394% |

LWC lowers the real INT4 PPL by **0.099705 (1.6176%)** and the fake PPL by **1.4783%**. A 10,000-sample paired bootstrap over the same test blocks gives a descriptive 95% interval of `[0.981342, 0.986284]` for the INT4 LWC / no-LWC PPL ratio; no conclusion about other datasets or tasks follows from it. The before/after comparison is recorded in `lwc_comparison.json`.

LWC CUDA and fake differ in PPL by **−0.001347 (−0.0222%)**, but their top-1 agreement is **86.9977%**, so they are still not bit-equivalent; fake's element-wise FP16 dequantization rounding semantics are unchanged. Against the independent integer reference, CUDA agrees on top-1 at **100%** of scored positions, the NLL difference is **0** for all 166 blocks, and the full-vocabulary logits at the 256 sampled positions are bit-identical. The bit-identity claim on full-vocabulary logits covers only these sampled positions. See `ppl_summary.json`.

[Table 1 of the original paper](https://arxiv.org/html/2406.01721v2#S4.T1) reports a WikiText-2 PPL of **6.08** for LLaMA-2-7B DuQuant+LWC, numerically close to the result here. However, this is an LWC experiment on the existing CUDA backend, and it differs from the paper:
- it uses a fixed alpha=0.6, while the original `run_lwc.sh` uses alpha=0.5;
- the attention path and the deployment-consistent FP16 training quantization semantics also differ.

So this is not a strict item-by-item reproduction of the paper, and the ~0.016 gap cannot be used to claim an improvement over it. C4 and QA were not run.

### End-to-end speed and memory of the LWC model

RTX 4090, GPU 1, batch=1, median of 5. FP16 and LWC INT4 were both re-measured in this experiment in separate processes; CUDA Graphs, autoregressive feedback, KV updates and the measurement boundaries are the same as the benchmarks above.

| input → output tokens | FP16 ms | LWC INT4 ms | end-to-end speedup | FP16 / INT4 decode tok/s |
|---|---:|---:|---:|---:|
| 128 → 32 | 462.11 | 181.26 | **2.55×** | 69.81 / 178.44 |
| 2048 → 32 | 664.46 | 264.57 | **2.51×** | 64.88 / 149.05 |

| input length | mode | model GiB | KV GiB | peak allocated GiB | reserved GiB | device usage GiB |
|---|---|---:|---:|---:|---:|---:|
| 128 | FP16 | 12.55 | 0.08 | 12.67 | 12.68 | 13.26 |
| 128 | LWC INT4 | 3.54 | 0.08 | 3.65 | 4.73 | 5.41 |
| 2048 | FP16 | 12.55 | 1.02 | 13.88 | 14.02 | 14.60 |
| 2048 | LWC INT4 | 3.54 | 1.02 | 4.93 | 5.12 | 5.81 |

LWC does not increase the size of the inference model tensors. KV, embedding, LM head and attention arithmetic are still FP16. These are end-to-end tests of fixed-shape requests, not dynamic serving throughput; identical generations on the two example prompts do not mean all generations are identical. Full data in `model_summary.json`.

### Checking and reproducing LWC

19 regression tests pass, and CUDA memcheck reports **0 errors**. The new tests cover:
- alignment with the upstream FP16 LWC quantizer;
- training gradients and export;
- learned asymmetric clipping on the real GEMM/GEMV, aligned with the integer reference;
- boundary-input gradients and consecutive optimizer steps of the fused STE.

Logs are in `benchmarks/lwc/{tests,memcheck}.log`.

Use the Python/CUDA environment variables from above:

```bash
# For a full retrain choose a new directory to avoid overwriting these parameters; after an interruption, rerun with the same arguments plus --resume.
"$DUQUANT_PYTHON" benchmarks/train_lwc.py --calibration benchmarks/cache/calibrated_w4a4 --output benchmarks/cache/lwc_reproduced --epochs 20 --fused-weight-quant

# Use the completed parameters and write to a separate evaluation directory.
"$DUQUANT_PYTHON" benchmarks/eval_ppl.py --mode fp16 --output benchmarks/lwc_recheck/ppl_fp16.json
for mode in fake cuda integer_reference; do
    "$DUQUANT_PYTHON" benchmarks/eval_ppl.py --mode "$mode" --calibration benchmarks/cache/lwc_w4a4 --output "benchmarks/lwc_recheck/ppl_${mode}.json"
done
"$DUQUANT_PYTHON" benchmarks/summarize_ppl.py --directory benchmarks/lwc_recheck

# For speed/memory re-measurement, start separate processes for FP16 and CUDA.
for prompt in 128 2048; do
    "$DUQUANT_PYTHON" benchmarks/bench_model.py --mode fp16 --prompt "$prompt" --new-tokens 32 --output "benchmarks/lwc_recheck/model_fp16_p${prompt}_g32.json"
    "$DUQUANT_PYTHON" benchmarks/bench_model.py --mode cuda --calibration benchmarks/cache/lwc_w4a4 --prompt "$prompt" --new-tokens 32 --output "benchmarks/lwc_recheck/model_cuda_p${prompt}_g32.json"
done
"$DUQUANT_PYTHON" benchmarks/summarize_model.py --directory benchmarks/lwc_recheck
```


<a id="model-family-validation"></a>
## End-to-end validation on five model families (no LWC)

This covers the five families of the README Model Zoo, with one full model each. All use the same real W4A4 CUDA backend. Each model is calibrated separately, then the full WikiText-2 test, the independent integer reference and autoregressive generation are rerun. What is confirmed here is these five specific checkpoints; it does not mean all sizes, variants or paper results of each family have been reproduced.

### Implementation and calibration scope

GQA support was completed for this:
- Q/K/V can have different projection widths, and the RoPE/cache kernels handle query and KV heads separately.
- The KV cache is stored with the actual number of KV heads, and Flash SDPA runs in GQA mode.
- The O-projection smoothing is tied within each group of query heads that shares one V head, so it folds correctly into the V/O weights; the MHA path keeps its original semantics.
- The model loader checks head_dim=128 and rejects scaled RoPE, sliding-window attention and projections with bias.

Each model uses 128×2048 tokens of WikiText-2 **train**, seed=2, alpha=0.6, block=128, 256 rotation-search steps. The original repository's rotation search produces two block rotations and a zigzag permutation, and each model exports its own parameters. No new LWC/LET training was added. LLaMA-2 reuses the completed no-LWC calibration with the same protocol; the other four were newly calibrated. Inference evaluation was rerun for all five models.

Rotations and permutations are read from the calibration files, not hard-coded to any one model's matrices. The representative input for the rotations is still the propagated mean of the calibration inputs, and attention still uses this backend's Hadamard + QK/V qdq semantics. The fake below is therefore **a PyTorch fake reference of this backend with the same parameters**. It is neither a rerun of the upstream evaluation entry point nor a strict reproduction of the paper's numbers.

### Full test-set quality

WikiText-2 test is computed over complete non-overlapping 2048-token blocks, scoring 2047 positions per block. Tokenizers produce different blocks and token counts, so PPL is not compared directly across models.

| model | Q/KV heads | blocks / scored tokens | HF FP16 PPL | backend FP16 PPL | fake W4A4 PPL | CUDA INT4 PPL |
|---|---:|---:|---:|---:|---:|---:|
| LLaMA-1 7B | 32/32 | 166 / 339,802 | 5.677054 | 5.677063 | 6.313289 | 6.297440 |
| LLaMA-2 7B | 32/32 | 166 / 339,802 | 5.472053 | 5.472090 | 6.156414 | 6.163763 |
| LLaMA-3 8B | 32/8 | 141 / 288,627 | 6.135790 | 6.135898 | 8.115615 | 8.104349 |
| Vicuna-v1.5 7B | 32/32 | 166 / 339,802 | 6.783673 | 6.783619 | 7.578176 | 7.580546 |
| Mistral-7B-Instruct-v0.3 | 32/8 | 163 / 333,661 | 5.494614 | 5.494637 | 5.999756 | 5.998084 |

For all five models, real CUDA versus the independent integer reference satisfies:
- top-1 is **100% identical** at every scored position;
- the NLL difference is **0** for every block;
- the full-vocabulary logits at 256 sampled positions per model are **bit-identical**.

The bit-identity claim on full logits covers only the sampled positions. CUDA and FP16 fake still have different rounding semantics, so close PPL does not mean token-by-token equivalence.

The HF baseline was checked on the full test set (PPL, all predicted positions and the sampled logits). In addition, prefill at lengths 1/17/128, each followed by 3 cached decode steps, was checked. The LLaMA-3 BOS-only sequence has one top-1 difference: the top HF FP16 logits are exactly tied, and the backend's choice is within one FP16 ULP. It is recorded separately as `accepted_exact_hf_tie` and not counted as top-1 agreement. All 12 short-sequence checks of the other four models agree on top-1. Full HF numeric differences are in each directory's `hf_comparison.json`.

### Full-model generation speed

Same-model RTX 4090 (about 48 GiB), one model per GPU, batch=1, median of 5. FP16 and INT4 run in separate processes; both use fused blocks, Flash SDPA and fixed-shape CUDA Graphs.
- Included: embedding, all 32 layers, LM head, real KV updates, autoregressive greedy feedback and the output-token copy back; for INT4 also the online rotation, permutation and quantization.
- Excluded: offline calibration/packing, loading, first compilation, graph capture, tokenization and the input H2D.
- EOS is ignored and 32 tokens are always generated. To stay within LLaMA-1's 2048-token context limit, input lengths of 128/1024 are used throughout.

| model | input → output | FP16 ms | INT4 ms | end-to-end speedup |
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

### Memory

The table below is for **1024 → 32**, in GiB, as `FP16 → INT4`.
- Model tensors include the rotation/quantization parameters.
- Peak allocated includes the graph capture/warmup peak; reserved is the resident inference pool.
- Total device usage also includes the CUDA context and similar overheads, and is recorded separately in the raw JSON.
- KV, embedding, LM head and attention arithmetic stay FP16. LLaMA-3's large vocabulary makes these unquantized weights a larger share.

| model | model tensors | KV (same for both) | peak allocated | reserved |
|---|---:|---:|---:|---:|
| LLaMA-1 7B | 12.55 → 3.54 | 0.515 | 13.23 → 4.24 | 13.41 → 4.91 |
| LLaMA-2 7B | 12.55 → 3.54 | 0.515 | 13.23 → 4.25 | 13.39 → 4.91 |
| LLaMA-3 8B | 14.96 → 5.24 | 0.129 | 15.26 → 5.58 | 15.33 → 6.47 |
| Vicuna-v1.5 7B | 12.55 → 3.54 | 0.515 | 13.23 → 4.25 | 13.39 → 4.91 |
| Mistral-7B-Instruct-v0.3 | 13.52 → 3.80 | 0.129 | 13.82 → 4.14 | 13.89 → 5.15 |

### Checkpoint sources and auditable records

- LLaMA-1: the converted original LLaMA-1 weights from ModelScope `skyline2006/llama-7b`, in `benchmarks/cache/family_models/llama1-7b-ms`.
  - Every downloaded file was verified against the SHA256 of the source manifest.
  - The old package had wrong BOS/EOS IDs and empty special-token strings. They were fixed according to the unmodified SentencePiece model (BOS=1/EOS=2/UNK=0), and the original 2048-token context config was restored; the full before/after is saved in `conversion.json`.
  - The 33 bin shards were converted to safetensors, and each tensor was checked for dtype and bitwise equality.
- LLaMA-2: local `/mnt/public_models/llama2-7b-hf`.
- LLaMA-3: local `/mnt/public_models/Llama-3-8B`, with its actual `PreTrainedTokenizerFast`; the LLaMA-2 tokenizer was not substituted.
- Vicuna: ModelScope `AI-ModelScope/vicuna-7b-v1.5`, in `benchmarks/cache/family_models/vicuna-7b-v1.5`. The SHA256 of both bin weight shards matches HF `lmsys/vicuna-7b-v1.5` commit `3321f76e3f527bd14065daf69dad9344000a201d`; the lossless safetensors conversion passed per-tensor verification.
- Mistral: local `/mnt/public_models/Mistral-7B-Instruct-v0.3`, config identifier `unsloth/Mistral-7B-Instruct-v0.3`; the dense BF16 weights are loaded as FP16. This checkpoint has `sliding_window=null`; **the sliding-window path of Mistral-v0.1 was not validated**.

All artifacts are in `benchmarks/families/{llama1,llama2,llama3,vicuna,mistral}/`: `run.json`, the calibration manifest, the model config, PPL for 5 modes, prediction/sampled-logits arrays, the HF check, speed/memory and logs. The two downloaded models also record the source manifest and per-shard conversion hashes. The overall table is [`benchmarks/families/summary.json`](../benchmarks/families/summary.json). The summary script checks that:
- all five families are complete;
- the integer reference covers the full test set;
- the calibration protocols match;
- the inference source hashes are unchanged.

22 regression tests pass and CUDA memcheck reports **0 errors**; logs are `benchmarks/families/tests.log` and `memcheck.log`. The new tests cover:
- 4Q/1KV and 4Q/2KV, prefill/multi-step decode and the compact cache;
- graph replay with changed inputs;
- floating-point equivalence of GQA smoothing and rejection of illegal untied scales.

Not covered here: larger parameter sizes, scaled RoPE, sliding-window attention, multi-GPU, W6A6, QA/C4 and dynamic serving. The speeds above apply only to the fixed-shape requests that were measured, and the quality results do not stand in for these untested areas.

### Reproduction

Use the Python/CUDA environment from above. `download_family_models.py` provides resumable ModelScope downloads with SHA256 verification for the two missing models; `convert_family_checkpoint.py` performs a verifiable lossless format conversion. The model files live in the git-ignored `benchmarks/cache/`.

```bash
# Run from the repository root; needed for the first download/conversion.
"$DUQUANT_PYTHON" benchmarks/download_family_models.py
"$DUQUANT_PYTHON" benchmarks/convert_family_checkpoint.py benchmarks/cache/family_models/vicuna-7b-v1.5
"$DUQUANT_PYTHON" benchmarks/convert_family_checkpoint.py benchmarks/cache/family_models/llama1-7b-ms --legacy-llama1

# LLaMA-3 as an example. Choose a new directory when redoing calibration.
"$DUQUANT_PYTHON" benchmarks/calibrate_model.py --model /mnt/public_models/Llama-3-8B --output benchmarks/cache/llama3_reproduced
# name determines the artifact directory; pick a new name for a re-check to avoid reusing existing results.
"$DUQUANT_PYTHON" benchmarks/run_family.py --name llama3_recheck --model /mnt/public_models/Llama-3-8B --calibration benchmarks/cache/llama3_reproduced

# Verify the existing artifacts of the five families and build the overall table.
"$DUQUANT_PYTHON" benchmarks/summarize_families.py
```

### One-command re-measurement of speed and memory for the five models

`benchmarks/run_e2e_suite.py` uses the five verified no-LWC calibrations above and runs 20 separate processes one after another on the same idle GPU. The 20 runs are five models × FP16/INT4 × 128/1024-token inputs, each generating 32 tokens and repeated 10 times by default. It re-measures generation performance only; the full PPL results remain in `benchmarks/families/`.

```bash
# Only checks the calibration config and lists the commands; does not use the GPU.
"$DUQUANT_PYTHON" benchmarks/run_e2e_suite.py --gpu 0 --output benchmarks/e2e_reproduced --dry-run
# Run after confirming the GPU is idle; the output directory must not exist.
"$DUQUANT_PYTHON" benchmarks/run_e2e_suite.py --gpu 0 --output benchmarks/e2e_reproduced
```

The script checks that the GPU is idle before each run, and checks for other compute processes every 2 seconds while running. If it detects contention, it marks the round as incomplete and does not produce a successful summary. It neither kills other users' jobs nor changes GPU clocks. The calibration and source SHA256, commands, run logs, every timing sample and the results are saved in the new directory. `summary.json` separates model tensors, KV, peak allocated, reserved and device usage including the CUDA context. Device usage is a snapshot and is not claimed to be a sampled peak.

The first attempt at an additional re-measurement is recorded in `benchmarks/e2e_recheck/run.json`. At the time all GPUs were busy with other jobs, the idle check exited before loading the model, and **0/20** runs completed. The successful re-measurement after GPU 0 was freed is recorded below. The historical tables above keep their original median-of-5 protocol.


<a id="e2e-recheck"></a>
### Five-model end-to-end re-measurement completed (same GPU, 10 runs each)

After GPU 0 was freed, `benchmarks/run_e2e_suite.py --gpu 0 --output benchmarks/e2e_fresh` was run, and all **20/20** runs completed. FP16 and INT4 of all five models ran one after another on the same RTX 4090, each run in its own process, median of 10; polling found no other compute process on that GPU. The inference kernels and calibration parameters were not changed, and no LWC was added.

This round is again batch=1, inputs of 128/1024 tokens and a fixed 32 generated tokens.
- Included: all 32 layers, online rotation/quantization, attention, KV updates, LM head, autoregressive feedback and the D2H of the generated tokens.
- FP16 and INT4 both use fixed-shape CUDA Graphs.
- Excluded: loading, offline packing, compilation, graph capture, tokenization and the input H2D.

All 20 runs passed the graph/eager generation-consistency and finite-logits checks.

| model | input→output | FP16 ms | INT4 ms | end-to-end speedup | FP16 / INT4 decode tok/s |
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

Memory is listed for **1024→32**, in GiB, as `FP16 → INT4`. Peak allocated includes the capture/warmup peak. Device usage is a snapshot of the resident inference state (including the context), not the device peak. The full memory metrics for 128-token inputs are also saved in the JSON.

| model | model tensors | KV (same for both) | peak allocated | reserved | device usage |
|---|---:|---:|---:|---:|---:|
| LLaMA-1 7B | 12.55 → 3.54 | 0.515 | 13.23 → 4.24 | 13.41 → 4.91 | 13.99 → 5.59 |
| LLaMA-2 7B | 12.55 → 3.54 | 0.515 | 13.23 → 4.25 | 13.39 → 4.91 | 13.97 → 5.59 |
| LLaMA-3 8B | 14.96 → 5.24 | 0.129 | 15.26 → 5.58 | 15.33 → 6.47 | 15.91 → 7.15 |
| Vicuna-v1.5 7B | 12.55 → 3.54 | 0.515 | 13.23 → 4.25 | 13.39 → 4.91 | 13.97 → 5.59 |
| Mistral-7B-Instruct-v0.3 | 13.52 → 3.80 | 0.129 | 13.82 → 4.14 | 13.89 → 5.12 | 14.47 → 5.80 |

Records of this run:
- The overall table is [`benchmarks/e2e_fresh/summary.json`](../benchmarks/e2e_fresh/summary.json).
- `plan.json` records all commands and the calibration hashes; `run.json` records the 20/20 completion status and the inference/measurement source hashes.
- Each model directory keeps the raw 10 timings, the memory breakdown, the generated tokens and the logs.

The earlier `benchmarks/e2e_recheck/`, which exited because the GPU was busy, is kept as a record of the failed attempt and is not used for these conclusions. Quality validation still refers to the full five-model WikiText-2 results above; PPL was not re-evaluated here.


<a id="kv-modes"></a>
## Explicit W4A4KV16 and real compressed W4A4KV4

Two explicit KV modes were implemented and validated on the five representative models. To isolate the effect of attention/KV, each model's earlier no-LWC calibrated weights, dual rotations, permutation and smoothing are kept fixed; no recalibration or LWC training was done.

### Mode definitions and implementation

| argument | Q | K/V numerics | KV physical storage | attention |
|---|---|---|---|---|
| `--kv-bits` omitted (legacy-compatible path) | Q4 QDQ after Hadamard | K4/V4 QDQ | FP16 | Flash SDPA |
| `--kv-bits 16` | FP16, no extra QDQ/Hadamard | W4A4 projection output, not quantized further | FP16 | Flash SDPA |
| `--kv-bits 4` | Hadamard after RoPE, kept in FP16 | K quantized after a paired Hadamard; V quantized directly | real packed 4-bit + FP16 scale/zero | prefill Flash SDPA; decode reads the compressed cache directly |

Both explicit modes keep the W4A4 QKV/O/MLP linear layers. KV16 still contains the quantization error of these linear layers; it is not the K/V of the original FP16 model. KV4 does not add the V/O rotations of QuaRot/SpinQuant, so this experiment is not a reproduction of those methods.

The new `duquant_cuda/kv4.py` implements fused RoPE, Q/K Hadamard, asymmetric K/V quantization and the cache write. The 128-dimensional vector of each token and each KV head is quantized independently: K clip=0.9, V clip=1.0. Two codes are packed into one uint8, with an FP16 scale and zero stored separately. Each K/V pair goes from 512 bytes to 136 bytes, a **3.7647×** compression (including the quantization parameters, excluding the temporary decode workspace).

Prefill also produces temporary FP16 dequantized K/V tensors of the current input chunk for Flash SDPA; the persistent cache holds only the packed codes and quantization parameters. Decode dequantizes tile by tile in registers/shared memory and **never materializes the whole FP16 history cache**.
- MHA uses an FP32 split-K reduction.
- In GQA, each group of query heads that shares a KV head reuses the unpacked values, and QK/PV run on Tensor Cores. The softmax weights are split into FP16 high/low parts to compute PV, reducing the loss of a single FP16 truncation.
- Softmax and the partial/merge accumulation are FP32.
- The tile is chosen as 32/64/128 from the cache capacity; dispatch depends only on the shape.

### Full WikiText-2 PPL

Again complete non-overlapping 2048-token test blocks, with the same tokenizer and scoring protocol as the family evaluation above. The legacy path in the table is the earlier real INT4 Linear + Q/K/V4 QDQ + FP16 physical cache; it must not be called explicit KV16 or real compressed KV4.

| model | original FP16 | legacy Q4/K4/V4 QDQ path | W4A4KV16 | W4A4KV4 |
|---|---:|---:|---:|---:|
| LLaMA-1 7B | 5.677063 | 6.297440 | **6.177697** | 6.255274 |
| LLaMA-2 7B | 5.472090 | 6.163763 | **5.999411** | 6.083101 |
| LLaMA-3 8B | 6.135898 | 8.104349 | **7.725156** | 7.938467 |
| Vicuna-v1.5 7B | 6.783619 | 7.580546 | **7.411816** | 7.501459 |
| Mistral-7B-Instruct-v0.3 | 5.494637 | 5.998084 | **5.897815** | 5.946837 |

On the full WikiText-2 evaluation of these five checkpoints, KV16 beats KV4 and also beats the legacy path every time. But this is a measured result under one fixed set of existing calibration parameters, not a guarantee for all tasks, samples or models, and there is still quantization loss relative to the original FP16. Compared with the legacy path, KV4 also drops Q4 quantization at the same time, so the accuracy difference between the two cannot be attributed entirely to the cache format.

For both new modes, the full **prefill PPL** matches the independent integer-arithmetic reference:
- the predictions at every scored position are identical;
- the NLL difference of every block is 0;
- the full-vocabulary logits at 256 sampled positions are bit-identical.

### Numeric bounds of cached decode

The PPL above is computed with whole-block prefill and does not directly prove that the new cached-decode kernel is bit-equivalent to another floating-point reduction. The checks showed that tiny rounding differences in FP32 attention are amplified at later W4A4 quantization boundaries. An isolated experiment raising the precision to FP64 did not guarantee whole-model agreement either, so the slower FP64 path was not adopted to chase such agreement.

The final check runs **8,640** real cached-decode attention cases per model. On identical Q/K/V, each is compared with an independent PyTorch FP32 dense attention rounded to FP16, and all satisfy `abs(error) <= 0.001 + 0.002*abs(reference)`. Whole-model logits diagnostics for prefill at lengths 1/129/1024 followed by 7 decode steps are also kept, and their differences are not hidden.

Below is an additional real-cache evaluation:
- 8 windows are taken evenly from the test set, each with a 128-token input.
- 32 real following tokens are then fed one token at a time, for 256 scored tokens in total.

It is a limited diagnostic: it cannot be mixed with the full test-set PPL and is not a formal equivalence test.

| model | KV16 CUDA/reference PPL (identical) | KV4 CUDA PPL | KV4 dense reference PPL | KV4 prediction agreement with reference | max relative L2 of a single attention |
|---|---:|---:|---:|---:|---:|
| LLaMA-1 7B | 7.668679 | 8.004426 | 8.139375 | 91.80% | 0.010380% |
| LLaMA-2 7B | 7.400700 | 7.449666 | 7.558630 | 85.55% | 0.012697% |
| LLaMA-3 8B | 17.264989 | 16.974064 | 17.253970 | 82.42% | 0.023850% |
| Vicuna-v1.5 7B | 9.926909 | 9.625573 | 9.978095 | 86.33% | 0.024869% |
| Mistral-7B-Instruct-v0.3 | 6.561293 | 6.471767 | 6.740647 | 86.72% | 0.016411% |

**The whole-model output of KV4 cached decode is not bit-/token-equivalent to the independent dense reference.** A lower PPL on this small sample is no basis for claiming that the kernel improves the quantization method. KV16 matches the reference exactly in this cached-decode diagnostic. KV4 should currently be treated as an experimental backend. Its evidence comes in three distinct kinds:
- numeric checks of the kernel on shared inputs;
- full prefill PPL;
- the independent cached evaluation.

### End-to-end generation speed

Same-model RTX 4090 (about 48 GiB). For each model, the FP16/KV16/KV4 comparison uses the same GPU with a separate process per run, batch=1, a fixed 32 generated tokens, median of 10. The measurement boundaries are the same as above:
- counted: all 32 layers, real feedback and KV writes, online rotation/quantization and the output D2H;
- not counted: offline calibration/packing, loading, compilation, graph capture, tokenization and the input H2D.

All runs use fixed-shape CUDA Graphs.

| model | input→output | FP16 ms | KV16 ms | KV4 ms | KV16 / KV4 speedup over FP16 |
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

The main definite benefit of KV4 is cache compression; it is not guaranteed to be faster than KV16. Here, with 1024-token inputs, KV4 is slightly faster on the MHA models and KV16 slightly faster on the GQA models. To reduce the repeated unpacking overhead in GQA, the final version uses the shared-KV Tensor Core kernel. Its decode and performance were re-measured, while the unchanged prefill PPL, KV16 and MHA results were kept; `reused_outputs` and the before/after source hashes record this.

### Memory and long context

The table below is for 1024→32, in GiB, as `KV16 → KV4`; model tensors do not change with the KV bit width.
- Peak allocated includes graph capture/warmup.
- Reserved is PyTorch's resident memory pool and is not the same as live tensors.
- The KV4 decode workspace is counted in the peak, while the cache column covers only the persistent codes/scale/zero.

| model | model tensors | KV cache | KV4 decode workspace | peak allocated | reserved | device usage snapshot |
|---|---:|---:|---:|---:|---:|---:|
| LLaMA-1 7B | 3.537 | 0.515 → 0.137 | 0.0045 | 4.245 → 3.897 | 4.908 → 4.941 | 5.587 → 5.620 |
| LLaMA-2 7B | 3.537 | 0.515 → 0.137 | 0.0045 | 4.246 → 3.898 | 4.910 → 4.943 | 5.589 → 5.622 |
| LLaMA-3 8B | 5.244 | 0.129 → 0.034 | 0.0084 | 5.583 → 5.501 | 6.471 → 6.516 | 7.150 → 7.194 |
| Vicuna-v1.5 7B | 3.537 | 0.515 → 0.137 | 0.0045 | 4.246 → 3.898 | 4.910 → 4.943 | 5.589 → 5.622 |
| Mistral-7B-Instruct-v0.3 | 3.798 | 0.129 → 0.034 | 0.0084 | 4.138 → 4.056 | 5.123 → 5.168 | 5.802 → 5.847 |

Mistral-7B-Instruct-v0.3 was additionally tested with a native 8192-token input and 32 generated tokens (no sliding window):

| mode | wall ms | speedup over FP16 | KV GiB | peak allocated GiB | reserved GiB |
|---|---:|---:|---:|---:|---:|
| fp16_kv16 | 1368.48 | 1.00× | 1.004 | 15.786 | 15.939 |
| cuda_kv16 | 564.83 | 2.42× | 1.004 | 6.427 | 6.697 |
| cuda_kv4 | 566.37 | 2.42× | 0.267 | 5.762 | 6.703 |

In this 8192→32 scenario, KV4 and KV16 have similar latency.
- KV drops from about 1.004 to 0.267 GiB, and peak allocated from about 6.43 to 5.76 GiB.
- However, **reserved is about 6.70 GiB for both**: the allocator/graph pool does not release resident memory in proportion to the live-tensor savings.

The compression ratio of the KV data cannot be called the compression ratio of total model memory. The earlier GQA scalar decode result of about 637 ms is kept in `diagnostics_scalar_decode/`; the final version takes about 566 ms.

### Checks and reproduction

The final 25 tests pass, and CUDA memcheck reports **0 errors**: `benchmarks/kv_modes/tests_final.log` and `memcheck_final.log`. New coverage:
- packed bytes/scale/zero;
- MHA/GQA and batch=2;
- boundaries such as 31/32/33, 63/64/65 and 127/128/129;
- consecutive appends and graphs with modified inputs;
- the full block and KV16.

The full table is [`benchmarks/kv_modes/summary.json`](../benchmarks/kv_modes/summary.json). Each family saves, for both modes:
- the full PPL and the independent reference;
- the per-token NLL of cached decode;
- the raw 10 performance samples and the memory breakdown;
- the source and calibration hashes.

Earlier diagnostics that did not pass the strict whole-model logits comparison, and the unadopted scalar GQA performance, are kept explicitly in diagnostics directories and are not used as final success claims.

```bash
# Reuse the Python/CUDA environment from above; choose an idle GPU.
# Use the same no-LWC calibration and explicitly select real KV4.
"$DUQUANT_PYTHON" benchmarks/bench_model.py --model /mnt/public_models/llama2-7b-hf --mode cuda --kv-bits 4 --calibration benchmarks/cache/calibrated_w4a4 --prompt 1024 --new-tokens 32 --repeats 10 --output benchmarks/kv4_recheck_model.json
# KV16: change --kv-bits 4 to --kv-bits 16.

# Redo both modes, PPL, cache diagnostics and generation benchmarks of this family in a new directory.
"$DUQUANT_PYTHON" benchmarks/run_kv_modes.py --family llama2 --output-root benchmarks/kv_modes_reproduced
# After an interruption, add --reuse-existing explicitly; existing results are not overwritten by default.
"$DUQUANT_PYTHON" benchmarks/summarize_kv_modes.py
```

The scope is still limited to these validated checkpoints, head_dim=128, the current GPU/software environment and fixed-shape generation. No support was added for paged serving, multi-GPU, sliding windows or scaled RoPE, and no new QA/C4 was run.


<a id="ptq-search"></a>
## Symmetric W, asymmetric A/K/V: clipping search and GPTQ ablation

This round validates LLaMA-2 7B first. All DuQuant dual rotations, permutations, smoothing and the model checkpoint from the earlier no-LWC calibration are kept fixed; only the quantizer and the weight rounding method change. The new configurations are a PTQ extension of DuQuant. The other four families were not rerun, and no new QA evaluation was done.

### Setup and data isolation

- W: per-output-channel, symmetric signed range [-8,7], FP16 scale. Stored as code=q+8 and zero=8, reusing the existing UINT4 GEMM/GEMV and affine epilogue. GPTQ's final integer codes are exported and loaded directly, without a second RTN.
- A: dynamic per-token asymmetric; Q stays FP16. KV4: per-token/head asymmetric with group=128 and a real packed cache, keeping the existing post-RoPE Q/K Hadamard and adding no V/O rotation.
- W search: each output channel scans 0.20–1.00 in steps of 0.01, picking the clipping by the FP32 reconstruction MSE of the original rotated weights; the scale is finally stored as FP16.
- A/K/V search:
  - range 0.40–1.00 in steps of 0.025, selected by the reconstruction MSE of the deployed FP16 QDQ;
  - A has one value per fused projection per layer, and K and V one value each per layer;
  - 64 rows are taken evenly from each training sequence, 8192 rows per layer per category in total.
- Data: the existing 128×2048 samples of WikiText-2 train, seed=2, with the raw Arrow/token/calibration hashes checked. During the RTN search, the outputs of already-searched quantized layers are propagated layer by layer. The test set is not used for any clipping choice.
- GPTQ:
  - uses exactly the same scale, zero and A/K/V clipping as the searched RTN; block=128, damping=1%, act-order off;
  - processes layers in the true QKV→O→gate/up→down order, building the Hessian from its own quantized prefix and the A4 inputs of all training tokens;
  - W clipping is not re-searched within GPTQ.
- Both KV bit widths use the same exported weights and A clipping; KV16 is not recalibrated separately.

The first, narrower grid hit its lower bound, so the range was widened based on the training reconstruction error alone and rerun. The first diagnostics are kept in `benchmarks/ptq_llama2/`; the final results are in `benchmarks/ptq_llama2_final/`.

### Full WikiText-2 test PPL

Same tokenizer, 2048-token non-overlapping blocks, 166 blocks and 339802 scored tokens as before; every entry is the full prefill PPL of the real CUDA kernels. The original FP16 PPL is 5.472090.

| configuration | W4A4KV16 | W4A4KV4 |
|---|---:|---:|
| original asymmetric W, fixed clipping, RTN | 5.999411 | 6.083101 |
| symmetric W, fixed clipping, RTN | 6.057011 | 6.135865 |
| symmetric W, searched W clipping, RTN | 5.949731 | 6.027398 |
| symmetric W, searched W/A/K/V clipping, RTN | 5.896381 | 5.974302 |
| same search results + GPTQ | 5.822099 | 5.890531 |

Switching only to symmetric W with the fixed clipping W=0.8/A=0.9/K=0.9/V=1.0 is slightly worse; searching the clipping improves it. With the same clipping, GPTQ lowers the KV16/KV4 PPL by a further ~0.0743/0.0838 relative to RTN. So there is a small positive gain on this checkpoint, which cannot be generalized to the same gain on all models.

For all 8 groups (4 configurations × 2 KV bit widths), the full PPL matches the independent integer reference:
- the predictions at every scored position are identical;
- the per-block NLL difference is 0;
- the full-vocabulary logits at 256 sampled positions are bit-identical.

The script also verifies that:
- the KV16/KV4 weights of the same configuration are identical;
- the W-only-search and full-search weights are identical;
- all scales/zeros of RTN and GPTQ are identical.

| projection | W clipping min/mean/max | A clipping values | mean GPTQ/RTN local Hessian-weighted error |
|---|---:|---|---:|
| qkv | 0.350 / 0.627 / 0.780 | 0.7 | 0.5125 |
| o | 0.400 / 0.625 / 0.770 | 0.675, 0.7 | 0.3425 |
| gate_up | 0.340 / 0.627 / 0.780 | 0.7 | 0.5693 |
| down | 0.370 / 0.589 / 0.720 | 0.625, 0.65, 0.675 | 0.5176 |

K clipping values: [0.925]; V clipping values: [0.925, 0.95]. A lower local weighted reconstruction error does not replace end-to-end PPL.

### End-to-end performance and memory

Same GPU 0 RTX 4090, separate processes, batch=1, 1024→32 tokens, CUDA Graphs, median of 10. All online rotation, quantization, cache updates and token feedback are counted, with the same measurement boundaries as the earlier reports. Model loading and the offline search/GPTQ are not counted in inference time.

| mode | wall ms | speedup over FP16 | KV GiB | peak allocated GiB | reserved GiB |
|---|---:|---:|---:|---:|---:|
| fp16 | 553.35 | 1.000× | 0.515 | 13.235 | 13.389 |
| sym_search_kv4 | 207.83 | 2.662× | 0.137 | 3.897 | 4.293 |
| sym_search_kv16 | 211.55 | 2.616× | 0.515 | 4.245 | 4.436 |
| sym_gptq_kv4 | 208.20 | 2.658× | 0.137 | 3.897 | 4.293 |
| sym_gptq_kv16 | 211.41 | 2.617× | 0.515 | 4.245 | 4.436 |

RTN and GPTQ use the same runtime format; GPTQ adds no online operations, and their latencies are essentially the same. Allocated and reserved mean different things. The reserved values above are the allocator/graph-pool state of this round, and differences from older processes should not be attributed to the quantization method.

### Cached-decode limitations and tests

KV4 still has the floating-point reduction sensitivity recorded in the previous section. Passing the same-input attention oracle does not mean whole-model cached decode is token-equivalent to the dense reference. An additional 256-token teacher-forced cache diagnostic follows; it must not be confused with the full test-set PPL:

| configuration | KV bits | CUDA / reference cache PPL | prediction agreement | max relative L2 of same-input attention |
|---|---:|---:|---:|---:|
| sym_search | 4 | 7.731109 / 7.621565 | 87.89% | 0.015587% |
| sym_search | 16 | 7.573966 / 7.573966 | 100.00% | 0.000000% |
| sym_gptq | 4 | 7.142639 / 7.141445 | 88.28% | 0.015263% |
| sym_gptq | 16 | 7.045854 / 7.045854 | 100.00% | 0.000000% |

KV16 matches the reference exactly in this cache diagnostic; KV4 cannot yet be claimed to be equivalent to the whole-model output of an independent dense backend. The improved full prefill PPL of the searched RTN also does not mean that the PPL of every small cache window improves.

28 regression tests pass (`benchmarks/ptq_llama2/tests.log`), and CUDA memcheck of the 3 new PTQ tests reports 0 errors (`memcheck.log` in the same directory). The new coverage:
- the GPTQ Triton block is compared against an independent column-by-column PyTorch oracle, including dead features and a diagonal Hessian degenerating to RTN;
- symmetric-weight GEMM/GEMV;
- the full block with searched clipping;
- real KV packing.

### Reproduction and artifacts

- Implementation: `duquant_cuda/ptq.py`.
- Search: `benchmarks/search_ptq.py`.
- Evaluation: `benchmarks/run_ptq_eval.py`.
- Overall table: [`benchmarks/ptq_llama2_final/summary.json`](../benchmarks/ptq_llama2_final/summary.json).

The model codes/scale/zero are in the git-ignored `benchmarks/cache/ptq_llama2_final/`. The logs record the hashes of the sources, the parent calibration, the exported layers and the data.

```bash
# Reuse the Python/CUDA environment from above; choose an idle GPU. Use new output directories for new experiments.
"$DUQUANT_PYTHON" benchmarks/search_ptq.py --method rtn --output benchmarks/cache/ptq_reproduced --report benchmarks/ptq_reproduced
"$DUQUANT_PYTHON" benchmarks/search_ptq.py --method gptq --output benchmarks/cache/ptq_reproduced --report benchmarks/ptq_reproduced
"$DUQUANT_PYTHON" benchmarks/run_ptq_eval.py --variant sym_gptq --calibration-root benchmarks/cache/ptq_reproduced --root benchmarks/ptq_reproduced
# The other three variants: sym_fixed / sym_wsearch / sym_search.
# Inference must specify --kv-bits 4 or 16 explicitly.
"$DUQUANT_PYTHON" benchmarks/bench_model.py --model /mnt/public_models/llama2-7b-hf --mode cuda --calibration benchmarks/cache/ptq_llama2_final/sym_gptq --kv-bits 4 --prompt 1024 --new-tokens 32 --repeats 10 --output benchmarks/ptq_gptq_recheck.json
```

<a id="qa-int4"></a>
## LLaMA-2 7B: six-task comparison with real INT4

All three quantized models use real W4A4 integer GEMMs, packed KV4 and the same scoring program.
- Quantization: symmetric per-channel W, asymmetric per-token A, Q16, asymmetric per-head K/V with group=128.
- They share the clipping search grid, the reconstruction-MSE objective and the true-sequential GPTQ rules of the previous section.
- Search/GPTQ use exactly the same 128×2048 WikiText-2 train token sequences.
- DuQuant uses the no-LWC `sym_gptq`; QuaRot uses a seed=0 random Hadamard; SpinQuant uses the official W16A4KV4 seed=0 learned rotations, followed by the common W4 GPTQ.

The data and budget of SpinQuant's rotation learning come separately from the official checkpoint, so the three methods cannot be said to have the same total optimization budget.

Full zero-shot datasets: 19,965 questions and 67,106 candidate answers, no truncation. PIQA 1838, ARC-E 2376, ARC-C 1172, BoolQ 3270, HellaSwag 10042, WinoGrande 1267. The prompts and independent context-continuation tokenization of the repository's old lm-eval are used, without BOS/EOS. ARC-C and HellaSwag report accuracy normalized by answer character length, the others raw accuracy; the average is the arithmetic mean of the six.

| model/method | PIQA | ARC-E | ARC-C | BoolQ | HellaSwag | WinoGrande | average |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 / KV16 | 78.40 | 69.32 | 40.61 | 71.10 | 72.95 | 67.09 | 66.58 |
| QuaRot W4A4KV4 | 74.97 | 67.42 | 38.74 | 68.44 | 68.35 | 63.61 | 63.59 |
| SpinQuant W4A4KV4 | 74.48 | 65.57 | 37.80 | 69.57 | 66.95 | 62.35 | 62.79 |
| DuQuant W4A4KV4 | 76.50 | 68.48 | 37.71 | 72.02 | 70.33 | 64.64 | 64.95 |

In this setup DuQuant averages 1.36 points above QuaRot and 2.16 points above SpinQuant, and 1.63 points below FP16. This is a comparison under a unified backend and quantization strategy. It is not a reproduction of each paper's default configuration, and it does not generalize to a ranking of the methods under other clipping, GPTQ, training budgets or evaluation protocols.

The actual inference scoring took about 263.7 s for DuQuant, 455.7 s for QuaRot, 434.6 s for SpinQuant and 657.5 s for FP16. These are run records of this evaluation on different idle GPUs, including compilation, padding and Python scheduling but excluding loading. They are not controlled inference speedups. The QuaRot/SpinQuant linear layers of the comparison still go through two identity DuQuant rotation steps, and their online transforms were not specifically optimized. No full fake-quant QA was run.

Validation includes:
- two new QA tests pass (independent Hugging Face scoring/right padding, and comparator INT4 versus an independent integer oracle);
- floating-point rotation-equivalence checks over 32 layers for both comparators;
- a spot check of the original 7B FP16 against Hugging Face on 12 questions/36 candidates: the raw/normalized final choices all agree, and the maximum cumulative log-likelihood difference is 0.023634.

All official evaluations use full causal prefill; the KV4 cached-decode numeric limitations above still apply.

See the full [protocol and reproduction](../benchmarks/qa_compare/README.md), the [summary JSON](../benchmarks/qa_compare/summary.json) and the [table](../benchmarks/qa_compare/table.md). Per-question predictions, both raw/normalized accuracies, and the source/calibration/data hashes are all saved. `benchmarks/summarize_qa.py` generates the summary only when all results are complete and the protocols agree.

### FlatQuant added with the same configuration

The official LLaMA-2 7B W4A4KV4 invertible non-orthogonal transforms, diagonal scaling and Q/K/V transforms are kept. Their pretrained clipping is replaced with the common search used here, followed by the same GPTQ. The model, training tokens, six-task data, scoring source and batch settings are the same as in the table above, using real INT4 GEMMs and packed KV4. This extension did not change the three earlier quantization methods or the original scoring code.

| method | PIQA | ARC-E | ARC-C | BoolQ | HellaSwag | WinoGrande | average |
|---|---:|---:|---:|---:|---:|---:|---:|
| FlatQuant W4A4KV4 | 77.37 | 69.02 | 38.31 | 68.90 | 70.78 | 65.90 | 65.05 |

FlatQuant averages 0.10 points above DuQuant, very close. It is 1.46 points above QuaRot, 2.26 above SpinQuant and 1.53 below FP16. It is higher on five of the tasks, while DuQuant is 3.12 points higher on BoolQ. The budget and configuration of the official transform training are kept as is, so this is not a reproduction of the original paper's default quantization setup.

Four new checks pass, and the floating-point equivalence check of all 32 layers passes (max relative L2 0.00185065). For 32 candidate answers, the cumulative log-likelihoods of the whole-model INT4 and the independent integer reference are identical, with a maximum difference of 0. The actual six-task scoring took 256.4 s (GPU 0, excluding loading), close to the 263.7 s of the earlier DuQuant run; these two evaluation run records do not replace a controlled performance benchmark.

See the full [FlatQuant protocol and results](../benchmarks/flatquant_compare/README.md) and the [five-method summary JSON](../benchmarks/flatquant_compare/summary.json).

<a id="qwen3-comparison"></a>
## Qwen3-1.7B-Base: the same W4A4KV4 protocol

Qwen3's head-wise Q/K RMSNorm, GQA and tied embedding/head are adapted locally. The FP16 baseline is converted from the original BF16 weights, and the quantized models use real integer W4A4/packed KV4.
- Full six tasks: 19,965 questions and 67,106 candidates, no chat template.
- ARC-C/HellaSwag use normalized accuracy, the others raw accuracy.
- All three methods share the 128×2048 WikiText-2 train data, the same clipping search and true-sequential GPTQ.

| method | PIQA | ARC-E | ARC-C | BoolQ | HellaSwag | WinoGrande | average |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 75.90 | 73.78 | 45.05 | 79.08 | 66.52 | 64.17 | 67.42 |
| QuaRot | 68.61 | 64.39 | 37.80 | 65.96 | 55.86 | 53.83 | 57.74 |
| SpinQuant | 68.61 | 67.63 | 38.99 | 66.88 | 55.40 | 56.35 | 58.98 |
| DuQuant | 70.73 | 72.18 | 41.72 | 72.51 | 58.86 | 56.35 | 62.06 |

DuQuant is 4.32 points above QuaRot and 3.08 above SpinQuant, and 5.36 points below FP16. These are local-adaptation results, not the authors' official Qwen3 results.
- SpinQuant uses the original Cayley SGDG with a W16A4KV4 STE causal loss: 800 steps, batch=1, length 2048, lr=1.5 cosine, cycling over the same 128 training samples. Its learning data and parallel training configuration differ from the original paper.
- DuQuant uses the original dual rotation/permutation, a 256-step search and alpha=.6 smoothing, without LWC.

Checks:
- The single-layer comparison with Hugging Face passes.
- A spot check of six questions/18 candidates on the native full model gives the same final choices.
- For each of the three full quantized models, four candidates score identically against the independent integer reference.
- SpinQuant's final maximum orthogonality deviation is 1.19210e-6, and the transform-equivalence checks over all 28 layers pass for QuaRot/SpinQuant.

The scoring times here were 164.9/200.5/181.6/198.0 s for FP16/QuaRot/SpinQuant/DuQuant. They include unfused transforms, compilation, padding and scheduling. This evaluation path does not show an INT4 speedup on Qwen3, and the LLaMA performance conclusions cannot be carried over.

See the full [experiment protocol, support status and reproduction](../benchmarks/qwen3_compare/README.md) and the [result JSON](../benchmarks/qwen3_compare/summary.json). The earlier KV4 cached-decode numeric limitations still apply.

### FlatQuant added on Qwen3 with the same configuration

FlatQuant's non-orthogonal SVD/Kronecker transforms, diagonal scaling and matching Q/K/V/O transforms are trained locally. Training uses the official modules, adapted to Qwen3's Q/K norm and GQA: 15 epochs per layer, batch 4, the same 128×2048 training samples. The training stage includes LWC/LAC; after export, these are replaced by the common clipping search and GPTQ. The methods have different transform-learning budgets, so this cannot be called a comparison at equal training budget, and it is not the authors' official Qwen3 result.

| method | PIQA | ARC-E | ARC-C | BoolQ | HellaSwag | WinoGrande | average |
|---|---:|---:|---:|---:|---:|---:|---:|
| FlatQuant W4A4KV4 | 71.55 | 69.32 | 40.61 | 70.34 | 63.05 | 60.06 | 62.49 |

The average is 0.43 points above DuQuant, with each method higher on three tasks; it is 4.93 points below FP16.

Checks:
- The 28-layer floating-point transform check passes (max relative L2 about 0.2603%), and three new tests pass.
- For 32 candidates, the cumulative log-likelihoods of the whole-model INT4 and the independent integer reference are identical, with a maximum difference of 0.

The full real-INT4 scoring took 171.0 s, excluding loading but including compilation and scheduling; it is not a controlled speed comparison. See the [FlatQuant Qwen3 report](../benchmarks/qwen3_compare/flatquant/README.md) and the [five-method result JSON](../benchmarks/qwen3_compare/flatquant/summary.json).

## LAMBADA, WikiText-2 and C4 PPL added for both models

This adds the English `lambada_openai` test (all 5153 examples), which was missing from FlatQuant's default task mix: it replaces BoolQ, which was measured before. All methods reuse their earlier checkpoints, without relearning transforms, re-searching clipping or redoing GPTQ; the quantized rows are all real W4A4KV4.

Also measured:
- the complete 2048-token blocks of the full WikiText-2 test;
- the C4-new validation slice of the FlatQuant code. The first 1100 documents of the first validation shard are joined with single spaces, and 256×2048 tokens are taken; C4 is not the whole corpus.

| model | method | LAMBADA acc % | LAMBADA last-word PPL | WikiText-2 PPL | C4 PPL | six-task average with LAMBADA instead of BoolQ % |
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

LAMBADA scoring details:
- Accuracy requires every subword of the last word to be the greedy prediction at its position.
- The last-word PPL uses the 5153 words as the denominator, and a target-token PPL is saved separately.
- A strict prefix/suffix split of the jointly encoded full text and context is used, to avoid extra target tokens from the SentencePiece leading space when the last word is encoded separately. All 5153 boundaries pass the check for both models.

WikiText/C4 use shifted tokens as the denominator; the different PPL columns must not be mixed.

The new six-task average aligns the task mix only. It keeps the frozen prompts and the old independent context/choice encoding of the five earlier QA tasks, so it is not a reproduction of FlatQuant's full official evaluation protocol. The original six-task average with BoolQ and the seven-task average are also saved in the summary. In this group, FlatQuant's new six-task average is about 0.68 points above DuQuant on LLaMA-2 and about 1.76 points above on Qwen3, and FlatQuant also has lower WikiText/C4 PPL on both models. The transform-training budget and the earlier Qwen3 local-adaptation limitations still apply.

Checks:
- The new last-word scoring test passes, and the LLaMA-2 FP16/DuQuant WikiText PPL exactly reproduce the earlier 5.472090/5.890531.
- All 10 LAMBADA/WikiText groups and 10 C4 groups completed. They passed the completeness, source/calibration/data hash, per-question prediction and scoring-token denominator checks.

See the [full report](../benchmarks/lambada_compare/README.md) and the [summary JSON](../benchmarks/lambada_compare/summary.json).
