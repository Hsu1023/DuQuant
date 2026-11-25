"""Same-device generation check for searched RTN/GPTQ and FP16."""
import os
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    root = Path("benchmarks/ptq_llama2_final/performance")
    root.mkdir(parents=True, exist_ok=True)
    jobs = [("fp16", "fp16", 16, None)]
    for variant in ("sym_search", "sym_gptq"):
        for bits in (4, 16):
            jobs.append((f"{variant}_kv{bits}", "cuda", bits,
                         str(Path("benchmarks/cache/ptq_llama2_final")/variant)))
    source = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [*Path("duquant_cuda").glob("*.py"), Path("benchmarks/bench_model.py"), Path(__file__)]}
    result = dict(status="running", source_sha256=source, results={})
    if (root/"summary.json").exists():
        raise FileExistsError("Preserve completed performance runs; use a fresh directory")
    for name, mode, bits, calibration in jobs:
        command = [sys.executable, "benchmarks/bench_model.py", "--model", os.environ.get("DUQUANT_LLAMA2", "/mnt/public_models/llama2-7b-hf"),
                   "--mode", mode, "--kv-bits", str(bits), "--prompt", "1024", "--new-tokens", "32",
                   "--repeats", "10", "--output", str(root/f"{name}.json")]
        if calibration:
            command += ["--calibration", calibration]
        print(f"Benchmarking {name}", flush=True)
        with (root/f"{name}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        value = json.loads((root/f"{name}.json").read_text())
        assert value["graph_matches_eager"] and value["finite_logits"] and len(value["samples"]) == 10
        result["results"][name] = value
        (root/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    baseline = result["results"]["fp16"]
    for value in result["results"].values():
        assert value["visible_devices"] == baseline["visible_devices"]
        value["speedup_vs_fp16"] = baseline["medians"]["wall_ms"]/value["medians"]["wall_ms"]
    result["status"] = "complete"
    (root/"summary.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
