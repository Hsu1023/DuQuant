# Qwen3-1.7B-Base, FP16 versus W4A4KV4

The [FlatQuant extension](flatquant/README.md) now adds a locally trained fifth
row under the same protocol: mean 62.49%, versus DuQuant 62.06%. Full five-row
results and training-budget differences are documented there; the original
four-row experiment below is preserved.

Local checkpoint: `/mnt/public_models/qwen3-1.7b-base`. The original BF16 weights
are converted to FP16 for this comparison, including the full precision row.
This is a base-model continuation-likelihood evaluation, without chat templates
or generated thinking traces.

## Complete results

Accuracy in percent. All four rows completed the same full splits:

| Method | PIQA | ARC-E | ARC-C | BoolQ | HellaSwag | WinoGrande | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 | 75.90 | 73.78 | 45.05 | 79.08 | 66.52 | 64.17 | 67.42 |
| QuaRot W4A4KV4 | 68.61 | 64.39 | 37.80 | 65.96 | 55.86 | 53.83 | 57.74 |
| SpinQuant W4A4KV4 | 68.61 | 67.63 | 38.99 | 66.88 | 55.40 | 56.35 | 58.98 |
| DuQuant W4A4KV4 | 70.73 | 72.18 | 41.72 | 72.51 | 58.86 | 56.35 | 62.06 |

DuQuant's mean is 4.32 percentage points above QuaRot and 3.08 above SpinQuant,
and 5.36 below FP16. These differences describe this checkpoint and local
adaptations, not an ordering of methods across arbitrary models or settings.
Scoring wall times were 164.9 / 200.5 / 181.6 / 198.0 seconds for
FP16 / QuaRot / SpinQuant / DuQuant, respectively. This evaluation pipeline
includes unfused transforms, compilation, cache allocation, padding and Python
overhead; it does not demonstrate an INT4 speedup on this model. The Qwen3 path
has not received a separate generation-kernel performance optimization pass.

`summary.json` passed completeness, matching protocol/source/configuration and
score-integrity checks. Each row retains 19965 question predictions and 67106
candidate scores. The FP16 preliminary and final-adapter reruns have the same
accuracy; the final row uses `results/fp16_final` so all reported runs share the
same recorded source hashes.

The common protocol retains the six complete tasks from the LLaMA experiment:
19965 questions and 67106 candidate continuations. They are retokenized with the
Qwen3 tokenizer. No BOS/EOS or truncation is added. ARC-C/HellaSwag use
character-normalized accuracy; PIQA/ARC-E/BoolQ/WinoGrande use raw accuracy.
Both metrics and per-question predictions are retained. Mean is unweighted.

All three quantized rows use real integer GEMM, symmetric per-output-channel W4,
asymmetric per-token A4, Q16 and packed asymmetric K4/V4 per head (group 128).
The embedding, final norm, head and Q/K norm remain FP16. K/V quantization is
applied during prefill as well as decode. A dedicated Qwen3 preparation path
applies Q/K head RMSNorm before RoPE and Q/K Hadamard, followed by packed KV4.
Existing LLaMA/FlatQuant experiment sources are preserved.

Calibration uses the same 128 x 2048 WikiText-2 train tokens for all methods,
seed 2. W clipping searches 0.20–1.00 by .01; A/K/V 0.40–1.00 by .025. The
objective is reconstruction MSE. GPTQ uses block 128, damping .01, no act-order,
fixed searched scales and true-sequential A4 Hessians from its own quantized
prefix. Evaluation questions are not used for calibration or rotation learning.

## Method adaptations

DuQuant uses the repository's dual block rotation/permutation search (block
128, 256 greedy steps) and alpha .6 smoothing, without LWC. Its four transforms
are learned from propagated mean calibration activations. QuaRot uses seed-0
randomized residual Hadamard, per-head V Hadamard, across-head O Hadamard and
online down-projection Hadamard. GQA V/O transforms respect the shared KV heads.
Both QuaRot/SpinQuant fold RMSNorm scales into adjacent weights and rotate the
embedding/head as well as the residual stream.

SpinQuant is a **local Qwen3 adaptation**, with actual learned R1 and per-layer
R2; it is not a random-rotation row renamed SpinQuant. The optimizer is the
original repository's Cayley SGDG, learning rate 1.5, momentum .9, cosine decay,
800 updates, batch 1, length 2048. Learning uses W16A4KV4 STE and causal language
model loss, then applies the common searched W4 GPTQ. The same 128 train samples
are cycled during optimization. This differs from the original paper's data
and distributed training schedule; all settings, losses and hashes are recorded.

## Public support audit

The original DuQuant repository lists LLaMA/Vicuna/Mistral; its newer DuQuant++
release uses MXFP4, a different format from this INT4 comparison. The original
SpinQuant release is LLaMA-focused; we did not find an author-released Qwen3
rotation checkpoint to reuse. Qwen3 support in another package is not sufficient
to conclude that original learned SpinQuant has been reproduced:

- [AIMET SpinQuant documentation](https://qualcomm.github.io/aimet-pages/releases/2.40.0/ptq_techniques/spinquant.html)
  supports Qwen3 but explicitly implements fixed R1/R2 Hadamard with **no
  optimization**.
- [QuaRot-on-RDNA4](https://github.com/NTHUQuantization/QuaRot-on-RDNA4) is a
  third-party dense-Qwen3 W4A4KV4 adaptation.
- [Original DuQuant](https://github.com/Hsu1023/DuQuant),
  [original SpinQuant](https://github.com/facebookresearch/SpinQuant), and
  [FlatQuant](https://github.com/ruikangliu/FlatQuant) do not establish that all
  these methods have author-reported Qwen3 results. FlatQuant's documented model
  zoo includes LLaMA, Qwen2.5 and DeepSeek.
- [AMD Quark rotation training](https://quark.docs.amd.com/release-0.11/pytorch/tutorial_rotation.html)
  explicitly validates trained rotations on Qwen3/Qwen3-MoE.
- [Intel AutoRound rotation documentation](https://github.com/intel/auto-round/blob/main/docs/rotation_details.md)
  provides optional trainable SpinQuant rotations with Cayley SGD and its own
  KL/MSE objectives. These establish broader Qwen3-era ecosystem support,
  without implying identical algorithms or evaluation settings.

This experiment tests adapted methods on one Qwen3 checkpoint. It is not an
original-paper-default reproduction or a claim of equal optimization budget.

## Validation and reproduction

`tests/test_qwen3.py` compares the Q/K-normalized GQA block to native Hugging Face
at batch/length 1/1, 2/17 and 1/65. Maximum relative L2 was approximately .001002.
`check_qwen3.py` additionally compares native full-model answer scores and checks
quantized full-model scores against an independent integer-dot reference.

Native Hugging Face checks cover six questions / 18 candidates: all raw and
normalized answer choices agree, with maximum cumulative log-likelihood
difference 0.052960. Each quantized model additionally matches an independent
integer-dot full-model reference exactly on four candidates (maximum score
difference zero); see `correctness.json`. SpinQuant completed 800 learning
updates with final rotation orthogonality error 1.19210e-6. Both QuaRot and
SpinQuant passed floating-point transformation-equivalence assertions for all
28 layers before their weights were quantized.

The existing `hb` PyTorch/CUDA environment is used. Transformers 4.51.3 and
Tokenizers .21.4 are isolated under `benchmarks/cache/qwen3_deps` because the
original environment's Transformers predates Qwen3; existing packages are not
replaced. The cached fast Hadamard extension and CUTLASS setup are reused.

```bash
python benchmarks/prepare_qwen3.py
python -m unittest discover -s tests -p test_qwen3.py -v
python benchmarks/train_spin_qwen3.py
python benchmarks/calibrate_qwen3.py --method duquant
python benchmarks/calibrate_qwen3.py --method quarot
python benchmarks/calibrate_qwen3.py --method spinquant
python benchmarks/eval_qwen3.py --method duquant --output benchmarks/qwen3_compare/results/duquant
# Repeat for quarot/spinquant; final FP16 output directory is results/fp16_final.
python benchmarks/check_qwen3.py
python benchmarks/summarize_qwen3.py
```

Calibration refuses existing manifests and evaluation requires explicit resume
with matching hashes. Wall times include compilation, padding and Python
scheduling and are not controlled latency benchmarks. Complete causal prefill
QA does not resolve the previously documented packed KV4 cached-decode numerical
limitation. Large model exports stay under ignored cache directories.
