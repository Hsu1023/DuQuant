"""Compare PPL and saved paired diagnostics; reject mismatched evaluation runs."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", type=Path, default=Path("benchmarks"))
    args = p.parse_args()
    runs = {
        mode: json.loads((args.directory / f"ppl_{mode}.json").read_text())
        for mode in ("fp16", "fake", "cuda")
    }
    baseline = runs["fp16"]
    for mode, run in runs.items():
        assert run["status"] == "complete", mode
        for key in (
            "model",
            "dataset",
            "seqlen",
            "block_count",
            "scored_tokens",
            "protocol",
        ):
            assert run[key] == baseline[key], (mode, key)
    assert runs["fake"]["quantization"] == runs["cuda"]["quantization"], (
        "Different quantization parameters/weights"
    )
    arrays = {}
    for mode, run in runs.items():
        with np.load(run["diagnostic_arrays"]) as data:
            arrays[mode] = dict(data)
        assert arrays[mode]["predictions"].size == run["scored_tokens"]
        assert np.array_equal(arrays[mode]["targets"], arrays["fp16"]["targets"])
        assert np.array_equal(
            arrays[mode]["sample_indices"], arrays["fp16"]["sample_indices"]
        )
    summary = dict(
        model=baseline["model"],
        dataset=baseline["dataset"],
        seqlen=baseline["seqlen"],
        block_count=baseline["block_count"],
        scored_tokens=baseline["scored_tokens"],
        quantization=runs["cuda"]["quantization"],
        metrics={
            mode: {k: run[k] for k in ("ppl", "mean_nll", "next_token_accuracy")}
            for mode, run in runs.items()
        },
        comparisons={},
        notes=[
            (
                "Train-calibrated W4A4 backend; see exported manifest and quantization fields for LWC settings; no LET optimization."
                if runs["cuda"]["quantization"]["calibrated"]
                else "Uncalibrated fixed-random-rotation W4A4 prototype, not the paper's calibrated DuQuant checkpoint."
            ),
            "fake is the PyTorch fake-quant reference for this same prototype, not the original repository calibration pipeline.",
            "Logit diagnostics use 64 evenly spaced scored positions in each of the first four blocks; prediction agreement uses all scored positions.",
            "Bootstrap resamples the paired disjoint blocks; descriptive interval, not a formal equivalence test or a guarantee on other tasks.",
        ],
    )
    for candidate, reference in (("cuda", "fake"), ("fake", "fp16"), ("cuda", "fp16")):
        a, b = arrays[candidate], arrays[reference]
        x = torch.from_numpy(a["sample_logits"].copy()).float()
        y = torch.from_numpy(b["sample_logits"].copy()).float()
        logx, logy = x.log_softmax(-1), y.log_softmax(-1)
        delta = np.array(
            [
                aa["nll_sum"] / aa["scored_tokens"]
                - bb["nll_sum"] / bb["scored_tokens"]
                for aa, bb in zip(runs[candidate]["blocks"], runs[reference]["blocks"])
            ]
        )
        indices = np.random.default_rng(42).integers(
            0, len(delta), size=(10000, len(delta))
        )
        ci = np.exp(np.quantile(delta[indices].mean(1), [0.025, 0.975]))
        summary["comparisons"][f"{candidate}_vs_{reference}"] = dict(
            ppl_delta=runs[candidate]["ppl"] - runs[reference]["ppl"],
            ppl_ratio=runs[candidate]["ppl"] / runs[reference]["ppl"],
            mean_nll_delta=float(delta.mean()),
            paired_block_bootstrap_ppl_ratio_95pct=ci.tolist(),
            bootstrap_samples=10000,
            bootstrap_seed=42,
            top1_prediction_agreement=float(
                np.mean(a["predictions"] == b["predictions"])
            ),
            sampled_logit_rows=len(x),
            sampled_logit_relative_l2=float((x - y).norm() / y.norm()),
            sampled_kl_reference_to_candidate=float(
                (logy.exp() * (logy - logx)).sum(-1).mean()
            ),
            sampled_logits_bitwise_equal=bool(torch.equal(x, y)),
        )
    integer_path = args.directory / "ppl_integer_reference.json"
    if not integer_path.exists():
        integer_path = args.directory / "ppl_integer_reference_4blocks.json"
    if integer_path.exists():
        ref = json.loads(integer_path.read_text())
        assert ref["status"] == "complete"
        assert ref["quantization"] == runs["cuda"]["quantization"]
        assert ref["dataset"] == baseline["dataset"]
        assert ref["seqlen"] == baseline["seqlen"]
        assert ref["block_count"] <= baseline["block_count"]
        with np.load(ref["diagnostic_arrays"]) as data:
            ref_arrays = dict(data)
        n = ref["scored_tokens"]
        count = ref["block_count"]
        diagnostic = dict(
            block_count=count,
            scored_tokens=n,
            integer_reference_ppl=ref["ppl"],
            description="Aligned blocks (count given above). Independent FP32 dot of exact integer codes, affine scales applied after dot; PyTorch reference norms/activations/Hadamard.",
            comparisons={},
        )
        y = torch.from_numpy(ref_arrays["sample_logits"].copy()).float()
        for mode in ("fake", "cuda"):
            assert np.array_equal(ref_arrays["targets"], arrays[mode]["targets"][:n])
            assert np.array_equal(
                ref_arrays["sample_indices"], arrays[mode]["sample_indices"]
            )
            x = torch.from_numpy(arrays[mode]["sample_logits"].copy()).float()
            logx, logy = x.log_softmax(-1), y.log_softmax(-1)
            diagnostic["comparisons"][mode] = dict(
                ppl_on_same_blocks=math.exp(
                    math.fsum(b["nll_sum"] for b in runs[mode]["blocks"][:count]) / n
                ),
                top1_agreement_with_integer_reference=float(
                    np.mean(
                        arrays[mode]["predictions"][:n] == ref_arrays["predictions"]
                    )
                ),
                sampled_logit_relative_l2=float((x - y).norm() / y.norm()),
                sampled_logits_bitwise_equal=bool(torch.equal(x, y)),
                max_per_block_nll_delta=max(
                    abs(a["nll_sum"] - b["nll_sum"])
                    for a, b in zip(runs[mode]["blocks"][:count], ref["blocks"])
                ),
                sampled_kl_integer_reference_to_candidate=float(
                    (logy.exp() * (logy - logx)).sum(-1).mean()
                ),
            )
        summary["integer_reference_diagnostic"] = diagnostic
    path = args.directory / "ppl_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
