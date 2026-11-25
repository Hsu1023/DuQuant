# DuQuant reconstruction experiment inspired by FlatQuant

This completed experiment tests one specific adaptation on Qwen3-1.7B-Base.
The results apply to the measured configuration.
The prior DuQuant/FlatQuant checkpoints, measurements and inference sources are
preserved.

## Completed Qwen3 results

Both predeclared exports completed every dataset and passed full-model integer
reference checks on 32 continuations (exact log-likelihoods, maximum difference0).

| Method | Original six mean (%) | LAMBADA six mean (%) | LAMBADA acc (%) | WikiText PPL | C4 PPL |
|---|---:|---:|---:|---:|---:|
| fp16 | 67.42 | 64.86 | 63.77 | 9.388 | 15.408 |
| duquant | 62.06 | 58.56 | 51.52 | 12.361 | 20.995 |
| flatquant | 62.49 | 60.32 | 57.31 | 11.547 | 19.527 |
| reconstruct_rtn | 64.45 | 61.54 | 57.66 | 10.700 | 17.626 |
| reconstruct_gptq | 63.24 | 61.12 | 59.09 | 10.820 | 17.824 |

The RTN extension improves the original six-task mean by 2.39 points over its
parent and 1.97 over the previously measured FlatQuant configuration. The GPTQ
extension improves by 1.18 and .75 points, respectively. Both extensions have
lower WikiText/C4 PPL and higher LAMBADA accuracy than that FlatQuant baseline.
RTN gives the better six-task mean and PPL here; GPTQ gives better LAMBADA.
This compares the recorded configurations and does not establish superiority
over FlatQuant's untouched author-default quantizer or on every individual task.

## Predeclared change

Keep the parent DuQuant block rotations and permutations frozen. Starting from
its folded FP16 weights and common clipping initialization, learn additional
positive per-channel folding scales plus symmetric per-channel W clipping and
one scalar A clip per fused projection / one K and V clip per layer. O scales
are tied across repeated query heads for each GQA V head. Scale folding retains
FP16 boundaries in training and export, and adds no online transforms.

Scale range is exp([-2,2]); W clip range [.2,1], A/K/V clip range [.4,1]. The
existing deployment format remains symmetric signed W4 encoded at zero8,
asymmetric per-token A4, Q16 and packed asymmetric per-head K4/V4 group128.
The existing kernels accept the learned exports directly.

Training uses each original frozen FP16 layer's output as the reconstruction
MSE target, on original FP16 prefixes, the same 128 x 2048 WikiText-2 train
sequences as the prior comparison. Per-layer 15 epochs, batch4, AdamW with
scale lr .005 and clipping lr .05, cosine decay to 5e-6. Objective is normalized
by its detached value, following the FlatQuant training loop. Grad norm is
clipped at10. Layer shards are disjoint and each independently computes the
same original FP16 prefix. The final epoch is used; no task/PPL-based selection.

A numerical diagnostic attempt failed with FP16 backward overflow and is
preserved in `diagnostic_fp16_backward/`. The completed experiment retains
FP16-rounded quantizer forward values but uses FP32 STE intermediate gradients,
including detached integer zero-point computation, avoiding `0 * infinity`
during half-precision division backward. The degenerate-input quantizer test
checks exact forward values against the deployed QDQ rule and finite gradients.
The failed attempt is excluded from final claims.

## Export and evaluation

Both RTN and GPTQ are predeclared and reported. RTN directly exports the learned
weights/scales/clips. GPTQ keeps those exact scales/clips and uses block128,
damping1%, no act-order, true-sequential A4 inputs and its own propagated
quantized prefix. No new clipping search overwrites the learned parameters.

All final measurements use real INT4 and packed KV4, with the unchanged scorers
and frozen requests: complete original six tasks, LAMBADA OpenAI, full WikiText-2
test blocks and the existing 256-block C4 validation slice. Both the original
BoolQ six-task mean and the LAMBADA six-task mean are reported. RTN/GPTQ
full-model integer-oracle checks precede summary generation.

The comparison target is the previously measured local Qwen3 FlatQuant export,
whose learned clipping was replaced by common search/GPTQ. This is a comparison
of explicit deployment configurations, not proof of superiority over untouched
FlatQuant defaults or of equal total optimization budget. Report all metrics
and both variants, including regressions, rather than selecting favorable tasks.

## Reproduction

```bash
python -m unittest discover -s tests -p test_reconstruct.py -v
python benchmarks/train_duquant_reconstruct.py --shard 0  # available GPU 0
python benchmarks/train_duquant_reconstruct.py --shard 1  # available GPU 4
python benchmarks/train_duquant_reconstruct.py --shard 2  # available GPU 7
python benchmarks/finish_duquant_reconstruct.py           # merge, export, GPTQ QA
python benchmarks/eval_duquant_reconstruct.py --variant rtn --output benchmarks/qwen3_compare/duquant_reconstruct/rtn/results
python benchmarks/eval_reconstruct_quality.py --variant rtn
python benchmarks/eval_reconstruct_quality.py --variant gptq
python benchmarks/check_reconstruct_integer.py
python benchmarks/summarize_reconstruct.py
```

All export parameters and codes are in ignored cache directories. Reports
retain training losses, manifests/hashes, full predictions and loss denominators.
Inherited packed KV4 cached-decode numerical limitations remain applicable;
full causal prefill evaluations do not validate cached-decode equivalence.
