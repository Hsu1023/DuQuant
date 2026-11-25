"""Rerun five complete models on one idle GPU; preserve all previous results."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from summarize_model import summarize

ROOT = Path(__file__).resolve().parents[1]
CALIBRATIONS = {
    "llama1": "family_llama1", "llama2": "calibrated_w4a4",
    "llama3": "family_llama3", "vicuna": "family_vicuna",
    "mistral": "family_mistral",
}


def gpu_state(gpu):
    def query(args):
        return subprocess.check_output(["nvidia-smi", *args, "--format=csv,noheader,nounits"], text=True)

    row = query([f"--id={gpu}", "--query-gpu=uuid,name,memory.used,utilization.gpu"]).strip().split(", ")
    processes = []
    for line in query(["--query-compute-apps=gpu_uuid,pid"]).splitlines():
        uuid, pid = line.strip().split(", ")
        if uuid == row[0]:
            processes.append(int(pid))
    return dict(uuid=row[0], name=row[1], used_mib=int(row[2]),
                utilization=int(row[3]), compute_pids=processes)


def require_idle(state):
    if state["compute_pids"] or state["used_mib"] > 256 or state["utilization"] > 5:
        raise RuntimeError(f"GPU is occupied; no benchmark started: {state}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, help="Physical GPU index or UUID")
    parser.add_argument("--output", type=Path, required=True, help="New result directory")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print all commands without using a GPU")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    output = args.output.resolve()
    if output.exists():
        parser.error("Choose a new output directory; existing results are preserved")
    jobs = []
    manifests = {}
    for family, folder in CALIBRATIONS.items():
        calibration = ROOT / "benchmarks/cache" / folder
        manifest_file = calibration / "manifest.json"
        manifest = json.loads(manifest_file.read_text())
        model = Path(manifest["model"])
        assert manifest["status"] == "complete" and manifest["lwc_epochs"] == 0
        assert hashlib.sha256((model / "config.json").read_bytes()).hexdigest() == manifest["config_sha256"]
        manifests[family] = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
        for prompt in (128, 1024):
            for mode in ("fp16", "cuda"):
                result = output / family / f"model_{mode}_p{prompt}_g32.json"
                command = [sys.executable, str(ROOT / "benchmarks/bench_model.py"),
                           "--model", str(model), "--mode", mode, "--prompt", str(prompt),
                           "--new-tokens", "32", "--repeats", str(args.repeats),
                           "--output", str(result)]
                if mode == "cuda":
                    command += ["--calibration", str(calibration)]
                jobs.append(dict(family=family, mode=mode, prompt=prompt,
                                 output=str(result), command=command))
    plan = dict(gpu=args.gpu, repeats=args.repeats, calibration_sha256=manifests, jobs=jobs)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    sources = [*ROOT.glob("duquant_cuda/*.py"), *ROOT.glob("duquant_cuda/*.cu"),
               *ROOT.glob("duquant_cuda/*.cuh"), Path(__file__),
               ROOT / "benchmarks/bench_model.py", ROOT / "benchmarks/summarize_model.py"]
    run = dict(status="running", completed_jobs=0, started_unix=time.time(),
               source_sha256={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    status_file = output / "run.json"

    def save():
        status_file.write_text(json.dumps(run, indent=2) + "\n")

    save()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
    try:
        for job in jobs:
            state = gpu_state(args.gpu)
            # Utilization can briefly describe the previous completed worker.
            for _ in range(5):
                if state["compute_pids"] or state["used_mib"] > 256 or state["utilization"] <= 5:
                    break
                time.sleep(2)
                state = gpu_state(args.gpu)
            require_idle(state)
            run["gpu"] = state
            result = Path(job["output"])
            result.parent.mkdir(exist_ok=True)
            print(f"Starting {job['family']} {job['mode']} {job['prompt']} -> 32", flush=True)
            foreign_pids = set()
            with result.with_suffix(".log").open("w") as log:
                process = subprocess.Popen(job["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while process.poll() is None:
                        foreign_pids.update(set(gpu_state(args.gpu)["compute_pids"]) - {process.pid})
                        time.sleep(2)
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
            if process.returncode:
                raise RuntimeError(f"Benchmark failed; see {result.with_suffix('.log')}")
            if foreign_pids:
                raise RuntimeError(f"Concurrent GPU users detected; timing is invalid: {sorted(foreign_pids)}")
            record = json.loads(result.read_text())
            assert record["graph_matches_eager"] and record["finite_logits"]
            assert record["layers"] == 32 and record["lwc_epochs"] == 0
            assert len(record["samples"]) == args.repeats
            if job["mode"] == "cuda":
                assert record["calibration_manifest_sha256"] == manifests[job["family"]]
            run["completed_jobs"] += 1
            save()
        summaries = {family: summarize(output / family, (128, 1024)) for family in CALIBRATIONS}
        (output / "summary.json").write_text(json.dumps(dict(
            status="complete", gpu=run["gpu"], repeats=args.repeats,
            protocol="Batch 1; fixed-shape CUDA Graphs; all 32 layers; real KV updates and greedy feedback; wall time includes generated-token D2H; offline work excluded",
            memory="model tensors, compact KV, capture/warmup-inclusive peak allocated, replay reserved, device used including context; device used is a snapshot, not a peak",
            families=summaries), indent=2) + "\n")
        run["status"] = "complete"
    except BaseException as error:
        run.update(status="incomplete", error=str(error))
        raise
    finally:
        run["finished_unix"] = time.time()
        save()


if __name__ == "__main__":
    main()
