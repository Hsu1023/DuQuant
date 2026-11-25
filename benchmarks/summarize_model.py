"""Summarize matching separate-process full-model benchmark runs."""

import json
import argparse
from pathlib import Path


def summarize(root, prompts=(128, 2048)):
    rows = []
    for prompt in prompts:
        fp = json.loads((root / f"model_fp16_p{prompt}_g32.json").read_text())
        quant = json.loads((root / f"model_cuda_p{prompt}_g32.json").read_text())
        for key in ("model", "layers", "batch", "prompt_tokens", "new_tokens", "gpu"):
            if fp[key] != quant[key]:
                raise ValueError(f"Incomparable runs: {key}")

        def memory(d):
            m = d["memory"]
            gib = 1024**3
            return dict(
                model_gib=m["model_storage_bytes"] / gib,
                kv_gib=m["kv_cache_bytes"] / gib,
                peak_allocated_gib=m["inference_peak_allocated_bytes"] / gib,
                reserved_gib=m["inference_reserved_bytes"] / gib,
                device_used_gib=m["inference"]["device_used_bytes"] / gib,
            )

        rows.append(
            dict(
                prompt_tokens=prompt,
                new_tokens=fp["new_tokens"],
                batch=fp["batch"],
                wall_ms=dict(
                    fp16=fp["medians"]["wall_ms"], w4a4=quant["medians"]["wall_ms"]
                ),
                end_to_end_speedup=fp["medians"]["wall_ms"]
                / quant["medians"]["wall_ms"],
                prefill_ms=dict(
                    fp16=fp["medians"]["prefill_ms"],
                    w4a4=quant["medians"]["prefill_ms"],
                ),
                decode_tokens_per_second=dict(
                    fp16=fp["decode_tokens_per_second"],
                    w4a4=quant["decode_tokens_per_second"],
                ),
                decode_speedup=fp["medians"]["decode_ms"]
                / quant["medians"]["decode_ms"],
                memory=dict(fp16=memory(fp), w4a4=memory(quant)),
                reserved_reduction_fraction=1
                - quant["memory"]["inference_reserved_bytes"]
                / fp["memory"]["inference_reserved_bytes"],
                generated_token_ids_match=fp["generated_token_ids"]
                == quant["generated_token_ids"],
            )
        )
    result = dict(
        gpu=fp["gpu"],
        model=fp["model"],
        calibration_manifest_sha256=quant.get("calibration_manifest_sha256"),
        caveat=(
            "Train-calibrated W4A4; see ppl_summary.json for full quality evaluation."
            if quant.get("calibration_manifest_sha256")
            else "Uncalibrated W4A4 prototype. Matching output on two repeated-text prompts is not a model-quality evaluation."
        ),
        results=rows,
    )
    (root / "model_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument("--prompts", type=int, nargs="+", default=[128, 2048])
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory, args.prompts), indent=2))
