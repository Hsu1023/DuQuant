"""One GPU per family; each evaluation/benchmark gets an isolated process."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--calibration", required=True)
    args = p.parse_args()
    model = str(Path(args.model).resolve())
    root = Path("benchmarks/families") / args.name
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def run(label, script, argv, output=None):
        if output is not None and output.exists():
            result = json.loads(output.read_text())
            if result.get("status", "complete") != "complete":
                raise RuntimeError(f"Incomplete result: {output}")
            if result.get("model") != model:
                raise ValueError(f"Wrong model in {output}")
            print(f"{args.name}: verified existing {label}", flush=True)
            return
        print(f"{args.name}: starting {label}", flush=True)
        with (root / f"{label}.log").open("w") as log:
            subprocess.run([sys.executable, script, *argv], stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        print(f"{args.name}: completed {label}", flush=True)

    cal = Path(args.calibration)
    manifest = json.loads((cal / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest.get("lwc_epochs", 0):
        raise ValueError("Expected completed calibration without LWC")
    shutil.copy2(cal / "manifest.json", root / "calibration_manifest.json")
    for mode in ("fp16", "hf", "cuda", "fake", "integer_reference"):
        output = root / f"ppl_{mode}.json"
        argv = ["--model", model, "--mode", mode, "--output", str(output)]
        if mode not in ("fp16", "hf"):
            argv += ["--calibration", str(cal)]
        run(f"ppl_{mode}", "benchmarks/eval_ppl.py", argv, output)
    for mode in ("fp16", "cuda"):
        for prompt in (128, 1024):
            output = root / f"model_{mode}_p{prompt}_g32.json"
            argv = ["--model", model, "--mode", mode, "--prompt", str(prompt),
                    "--new-tokens", "32", "--output", str(output)]
            if mode == "cuda":
                argv += ["--calibration", str(cal)]
            elif prompt == 128:
                argv += ["--verify-hf"]
            run(output.stem, "benchmarks/bench_model.py", argv, output)
    run("ppl_summary", "benchmarks/summarize_ppl.py", ["--directory", str(root)])
    run("model_summary", "benchmarks/summarize_model.py",
        ["--directory", str(root), "--prompts", "128", "1024"])
    hf = json.loads((root / "ppl_hf.json").read_text())
    fp = json.loads((root / "ppl_fp16.json").read_text())
    for key in ("model", "dataset", "seqlen", "block_count", "scored_tokens", "protocol"):
        assert hf[key] == fp[key], key
    with np.load(hf["diagnostic_arrays"]) as a, np.load(fp["diagnostic_arrays"]) as b:
        assert np.array_equal(a["targets"], b["targets"])
        agreement = float(np.mean(a["predictions"] == b["predictions"]))
        x, y = b["sample_logits"].astype(np.float64), a["sample_logits"].astype(np.float64)
        relative_l2 = float(np.linalg.norm(x-y) / np.linalg.norm(y))
    delta = fp["ppl"] / hf["ppl"] - 1
    passed = abs(delta) < 0.005 and agreement > 0.99 and relative_l2 < 0.01
    check = dict(hf_ppl=hf["ppl"], fp16_ppl=fp["ppl"], ppl_relative_delta=delta,
                 full_top1_agreement=agreement, sampled_logit_relative_l2=relative_l2,
                 acceptance="abs(PPL ratio - 1) < 0.5%; full top1 agreement > 99%; sampled logits relative L2 < 1%",
                 passed=passed)
    (root / "hf_comparison.json").write_text(json.dumps(check, indent=2) + "\n")
    if not passed:
        raise AssertionError(f"HF full-model validation failed: {check}")
    quality = json.loads((root / "ppl_summary.json").read_text())
    ref = quality["integer_reference_diagnostic"]["comparisons"]["cuda"]
    if (ref["top1_agreement_with_integer_reference"] != 1
            or ref["max_per_block_nll_delta"] != 0
            or not ref["sampled_logits_bitwise_equal"]):
        raise AssertionError("CUDA differs from independent integer reference")
    source_files = [*Path("duquant_cuda").glob("*.py"), *Path("duquant_cuda").glob("*.cu"),
                    *Path("duquant_cuda").glob("*.cuh"), Path(__file__)]
    (root / "run.json").write_text(json.dumps(dict(
        status="complete", family=args.name, model=model,
        seconds=time.perf_counter()-started,
        source_sha256={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files},
    ), indent=2) + "\n")
    print(f"{args.name}: all full-model checks complete", flush=True)


if __name__ == "__main__":
    main()
