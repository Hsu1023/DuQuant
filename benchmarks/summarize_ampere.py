"""Summarize the Ampere end-to-end runs (same fields as benchmarks/hopper/performance/summary.json).

Usage: python benchmarks/summarize_ampere.py a100 a40
Reads benchmarks/ampere/<gpu>/performance/{fp16,rtn_kv4}.json and writes summary.json next to them.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for gpu in sys.argv[1:]:
        root = ROOT / "benchmarks/ampere" / gpu / "performance"
        fp16, w4 = (json.loads((root / f"{m}.json").read_text()) for m in ("fp16", "rtn_kv4"))
        assert fp16["gpu"] == w4["gpu"] and fp16["prompt_tokens"] == w4["prompt_tokens"] == 1024
        assert fp16["new_tokens"] == w4["new_tokens"] == 32 and len(fp16["samples"]) == len(w4["samples"]) == 10
        peak = lambda r: r["memory"]["inference_peak_allocated_bytes"]  # includes capture/warmup and graph pool
        store = lambda r: r["memory"]["model_storage_bytes"]
        summary = dict(
            status="complete",
            model="meta-llama/Llama-2-7b-hf (HF snapshot 01c7f73d771dfac7d292323805ebc428287df4f9)",
            gpu=w4["gpu"],
            checkpoint="DuQuant reconstruction RTN W4A4KV4 (the GH200 calibration export, unchanged)",
            speedup=fp16["medians"]["wall_ms"] / w4["medians"]["wall_ms"],
            peak_allocated_reduction=1 - peak(w4) / peak(fp16),
            model_storage_reduction=1 - store(w4) / store(fp16),
            table={
                name: dict(
                    prefill_ms=r["medians"]["prefill_ms"],
                    decode_ms_per_token=r["decode_ms_per_token"],
                    wall_ms=r["medians"]["wall_ms"],
                    model_storage_gib=store(r) / 2**30,
                    peak_allocated_gib=peak(r) / 2**30,
                    generated_token_ids=r["generated_token_ids"],
                )
                for name, r in (("fp16", fp16), ("rtn_kv4", w4))
            },
            same_generated_tokens=fp16["generated_token_ids"] == w4["generated_token_ids"],
            results=dict(fp16=fp16, rtn_kv4=w4),
        )
        (root / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        t = summary["table"]
        print(gpu, w4["gpu"], f"speedup {summary['speedup']:.2f}x",
              f"peak -{summary['peak_allocated_reduction']:.1%}", f"storage -{summary['model_storage_reduction']:.1%}",
              "same tokens" if summary["same_generated_tokens"] else "DIFFERENT tokens",
              {k: {kk: round(vv, 3) for kk, vv in v.items() if kk != "generated_token_ids"} for k, v in t.items()})


if __name__ == "__main__":
    main()
