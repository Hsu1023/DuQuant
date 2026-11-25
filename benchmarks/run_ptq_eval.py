"""Evaluate one predeclared PTQ variant; never fit on evaluation tokens."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=["sym_fixed", "sym_wsearch", "sym_search", "sym_gptq"])
    p.add_argument("--root", default="benchmarks/ptq_llama2_final")
    p.add_argument("--calibration-root", default="benchmarks/cache/ptq_llama2_final")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    out = Path(args.root)/args.variant
    cal = Path(args.calibration_root)/args.variant
    meta = json.loads((cal/"manifest.json").read_text())
    if meta["status"] != "complete":
        raise ValueError("Calibration incomplete")
    out.mkdir(parents=True, exist_ok=True)
    source = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in
              [*Path("duquant_cuda").glob("*.py"), Path(__file__), Path("benchmarks/eval_ppl.py"),
               Path("benchmarks/verify_kv_decode.py"), Path("benchmarks/bench_model.py")]}
    record = dict(status="running", variant=args.variant, model=meta["model"],
                  gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                  calibration_manifest_sha256=hashlib.sha256((cal/"manifest.json").read_bytes()).hexdigest(),
                  source_sha256=source, results={})
    if (out/"summary.json").exists():
        if not args.resume:
            raise FileExistsError("Use --resume explicitly")
        previous = json.loads((out/"summary.json").read_text())
        for key in ("source_sha256", "calibration_manifest_sha256"):
            if previous[key] != record[key]:
                raise ValueError("Cannot resume results from different code/calibration")
    (out/"summary.json").write_text(json.dumps(record, indent=2)+"\n")

    def run(script, name, extra):
        target = out/f"{name}.json"
        if target.exists() and args.resume:
            value = json.loads(target.read_text())
            if value.get("status") == "complete" or value.get("graph_matches_eager"):
                return value
        command = [sys.executable, f"benchmarks/{script}.py", "--model", meta["model"],
                   "--calibration", str(cal), "--output", str(target), *extra]
        print("Running " + " ".join(command), flush=True)
        with (out/f"{name}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        return json.loads(target.read_text())

    for bits in (4, 16):
        value = {}
        for mode in ("cuda", "integer_reference"):
            value[mode] = run("eval_ppl", f"ppl_{mode}_kv{bits}", ["--mode", mode, "--kv-bits", str(bits)])
        a, b = value["cuda"], value["integer_reference"]
        with np.load(a["diagnostic_arrays"]) as aa, np.load(b["diagnostic_arrays"]) as bb:
            comparison = dict(predictions_equal=np.array_equal(aa["predictions"], bb["predictions"]),
                              sampled_logits_equal=np.array_equal(aa["sample_logits"], bb["sample_logits"]))
        comparison["max_block_nll_delta"] = max(abs(x["nll_sum"]-y["nll_sum"]) for x, y in zip(a["blocks"], b["blocks"]))
        if (not comparison["predictions_equal"] or not comparison["sampled_logits_equal"]
                or comparison["max_block_nll_delta"] != 0):
            raise AssertionError(comparison)
        value["integer_comparison"] = comparison
        if args.variant in ("sym_search", "sym_gptq"):
            value["decode"] = run("verify_kv_decode", f"decode_kv{bits}", ["--kv-bits", str(bits)])
        if args.benchmark:
            value["benchmark"] = run("bench_model", f"model_kv{bits}_p1024_g32",
                ["--mode", "cuda", "--kv-bits", str(bits), "--prompt", "1024", "--new-tokens", "32", "--repeats", "10"])
        record["results"][str(bits)] = value
        (out/"summary.json").write_text(json.dumps(record, indent=2)+"\n")
        print(f"{args.variant} KV{bits} PPL={a['ppl']:.6f}; full integer-reference agreement", flush=True)
    record["status"] = "complete"
    (out/"summary.json").write_text(json.dumps(record, indent=2)+"\n")


if __name__ == "__main__":
    main()
