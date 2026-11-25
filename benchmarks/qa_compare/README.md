# LLaMA-2 7B: six-task INT4 comparison

This experiment compares DuQuant, QuaRot and SpinQuant with the same deployed
W4A4KV4 format and evaluation code. All three quantized models use true integer
GEMM and packed KV4. Original FP16/KV16 is a separate reference row.

## Protocol

- Original checkpoint: `/mnt/public_models/llama2-7b-hf`.
- W: symmetric per-output-channel, signed range [-8, 7], stored as UINT4
  code = signed value + 8. A: asymmetric per-token. Q: FP16.
- K/V: asymmetric per-token/head, group size 128, FP16 scale/zero metadata.
- Common clipping search: W 0.20–1.00 by 0.01; A/K/V 0.40–1.00 by 0.025.
  Reconstruction MSE objective. A/K/V use 8192 sampled activation rows per layer.
- Common GPTQ: block size 128, damping 1%, no activation ordering, fixed searched
  scales/clips, true-sequential A4 Hessians and each method's own quantized prefix.
- Common search/GPTQ calibration: the same 128 x 2048 WikiText-2 **train** token
  sequences. Evaluation questions are not used for quantization or selection.
- Zero-shot, complete splits: PIQA 1838, ARC-E 2376, ARC-C 1172, BoolQ 3270,
  HellaSwag 10042, WinoGrande 1267; total 19965 questions, 67106 continuations.
- Frozen legacy repository lm-eval prompts; context and continuation tokenized
  separately with no BOS/EOS. No request exceeds 4096 tokens; no truncation.
- Primary metrics: raw accuracy for PIQA, ARC-E, BoolQ and WinoGrande;
  character-length-normalized accuracy for ARC-C and HellaSwag. Both forms are
  retained. Mean is the unweighted mean of the six primary metrics.
- Score complete continuations by full causal prefill, with shared FP16 final
  norm/head and FP32 log-softmax. Padding positions are excluded from scoring.

## Method-specific transforms

DuQuant retains the existing calibrated smoothing, two learned block rotations
and permutation, using the `sym_gptq` export without LWC. QuaRot uses a seed-0
randomized residual Hadamard, head Hadamard for V, an online across-head Hadamard
before O and an online full Hadamard before down-projection. SpinQuant uses the
official learned residual and head rotations, with online down-projection
Hadamard. The comparator online O/down transforms run in FP32 then cast to FP16.
The common backend performs post-RoPE Q/K Hadamard for all three methods.

SpinQuant rotation source is the official `LLaMA-2-7B/7B_W16A4KV4_lr_1.5_seed_0/R.bin`
release, Google Drive file ID `193O_aCS6MaObW8JosZP5CASwHAVU259E`.
Its rotation-learning budget/data are separate from the shared subsequent
calibration. This comparison controls quantization and evaluation; it does not
control the entire rotation-optimization budget or reproduce paper defaults.

Reference implementations: QuaRot commit
`5008669b08c1f11f9b64d52d16fddd47ca754c5a`; SpinQuant commit
`8f47aa3f00e8662caf1a484153920a07e5281c3a`.
All 32 comparator layers undergo a full-precision rotation-invariance check
before quantization. `tests/test_qa.py` independently checks the scorer against
Hugging Face, padding invariance and the comparator INT4 arithmetic against
an integer oracle, including the 11008-dimensional online Hadamard.

The real 7B FP16 scorer also agrees with native Hugging Face on all raw and
normalized answer choices for a fixed check of two questions per task (12
questions, 36 continuations). Maximum cumulative log-likelihood difference is
0.023634; see `hf_scorer_check.json` and `benchmarks/check_qa_hf.py`. This small
check validates scorer alignment; it is not a full fake-quant evaluation.

## Outputs and reproduction

`tasks.json` pins source files, prompts, tokenization and request hashes.
Each `results/<method>/run.json` records settings, source hashes, all task metrics
and completeness; `scores.npy` and `predictions.jsonl` preserve per-choice and
per-question results. `summary.json` and `table.md` are generated only after all
four runs complete and their protocols agree.

Use the existing `hb` Python environment, CUDA 12.4 and CUTLASS setup from
`duquant_cuda/CALIBRATED.md`. The comparator also requires the locally built
`fast_hadamard_transform` under the ignored `benchmarks/cache/qa_deps` directory.

```bash
python benchmarks/prepare_qa.py
# Each comparator export refuses to overwrite an existing manifest.
python benchmarks/calibrate_comparison.py --method quarot
python benchmarks/calibrate_comparison.py --method spinquant
python benchmarks/eval_qa.py --method duquant --mode cuda --output benchmarks/qa_compare/results/duquant
# Repeat evaluation for quarot, spinquant and fp16 on available GPUs.
python benchmarks/summarize_qa.py
```

Resume only with explicit `--resume`; source and protocol hashes must match.
Model exports, downloaded rotations and raw datasets stay under ignored cache
directories. Evaluation wall times include compilation and batching overhead
and are not a controlled latency benchmark. This prefill evaluation does not
resolve the separately documented KV4 cached-decode numerical limitation.
