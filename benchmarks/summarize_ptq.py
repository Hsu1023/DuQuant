"""Collect the predeclared symmetric-W clipping/GPTQ ablation."""
import hashlib
import json
from pathlib import Path
import torch


def main():
    root = Path("benchmarks/ptq_llama2_final")
    rows = []
    baseline = {str(bits): json.loads(Path(f"benchmarks/kv_modes/llama2/kv{bits}/ppl_cuda.json").read_text())
                for bits in (4, 16)}
    for variant in ("sym_fixed", "sym_wsearch", "sym_search", "sym_gptq"):
        value = json.loads((root/variant/"summary.json").read_text())
        assert value["status"] == "complete", variant
        for path, digest in value["source_sha256"].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
        calibration = Path("benchmarks/cache/ptq_llama2_final")/variant/"manifest.json"
        assert hashlib.sha256(calibration.read_bytes()).hexdigest() == value["calibration_manifest_sha256"]
        for bits, result in value["results"].items():
            a, b = result["cuda"], result["integer_reference"]
            assert a["status"] == b["status"] == "complete"
            assert a["scored_tokens"] == baseline[bits]["scored_tokens"] == 339802
            assert a["dataset"]["token_ids_sha256"] == baseline[bits]["dataset"]["token_ids_sha256"]
            assert a["quantization"]["rotations_and_permutations_sha256"] == baseline[bits]["quantization"]["rotations_and_permutations_sha256"]
            check = result["integer_comparison"]
            assert check["predictions_equal"] and check["sampled_logits_equal"] and check["max_block_nll_delta"] == 0
            if "decode" in result:
                assert result["decode"]["status"] == "complete"
                assert result["decode"]["packed_attention_oracle"]["passed"]
        rows.append(value)
        assert value["results"]["4"]["cuda"]["quantization"]["dequantized_fp16_weights_sha256"] == value["results"]["16"]["cuda"]["quantization"]["dequantized_fp16_weights_sha256"]
    quantizers = {r["variant"]: r["results"]["4"]["cuda"]["quantization"] for r in rows}
    assert quantizers["sym_wsearch"]["dequantized_fp16_weights_sha256"] == quantizers["sym_search"]["dequantized_fp16_weights_sha256"]
    assert quantizers["sym_gptq"]["dequantized_fp16_weights_sha256"] != quantizers["sym_search"]["dequantized_fp16_weights_sha256"]
    for key in ("activation_clips", "kv_clips"):
        assert quantizers["sym_gptq"][key] == quantizers["sym_search"][key]
    for layer in range(32):
        pair = [torch.load(Path("benchmarks/cache/ptq_llama2_final")/variant/f"layer_{layer:02d}.pt",
                           mmap=True, map_location="cpu", weights_only=True)["quantization"]
                for variant in ("sym_search", "sym_gptq")]
        for name in ("qkv", "o", "gate_up", "down"):
            for key in ("scale", "zero"):
                assert torch.equal(pair[0]["weights"][name][key], pair[1]["weights"][name][key])
    performance = json.loads((root/"performance/summary.json").read_text())
    assert performance["status"] == "complete"
    for path, digest in performance["source_sha256"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    for row in rows:
        if row["variant"] in ("sym_search", "sym_gptq"):
            for bits in (4, 16):
                assert performance["results"][f"{row['variant']}_kv{bits}"]["calibration_manifest_sha256"] == row["calibration_manifest_sha256"]
    search = json.loads((root/"sym_search_calibration.json").read_text())
    gptq = json.loads((root/"sym_gptq_calibration.json").read_text())
    stats = {}
    for name in ("qkv", "o", "gate_up", "down"):
        details = [row["details"][name] for row in search]
        stats[name] = dict(weight_clip_min=min(x["weight"]["clip_min"] for x in details),
                           weight_clip_mean=sum(x["weight"]["clip_mean"] for x in details)/32,
                           weight_clip_max=max(x["weight"]["clip_max"] for x in details),
                           activation_clips=sorted({x["activation"]["chosen"] for x in details}),
                           mean_gptq_to_rtn_hessian_error=sum(x["details"][name]["gptq_hessian_error"]/
                               x["details"][name]["rtn_hessian_error"] for x in gptq)/32)
    for name in ("k", "v"):
        stats[name] = dict(clips=sorted({row["details"][name]["chosen"] for row in search}))
    result = dict(status="complete", model=rows[0]["model"], variants=rows, baseline=baseline,
                  performance=performance, rtn_gptq_weight_scales_and_zeros_identical=True,
                  clipping=stats, protocol="Frozen existing DuQuant rotations, permutation and smoothing. No LWC. Train-only 128x2048 samples; independent full WikiText-2 test. Each variant evaluated at both KV bitwidths without recalibrating between them.",
                  search="W per-channel reconstruction MSE, 0.20..1.00 step0.01; A/K/V deployed QDQ MSE, 0.40..1.00 step0.025. 64 evenly spaced activation rows per train sequence per projection/cache kind, 8192 rows per layer-kind.",
                  gptq="Same RTN-searched scales/clips; block128, damp1%, no act-order; true-sequential qkv/o/gate-up/down, own quantized prefix and A4 Hessians; no test-set tuning.",
                  cached_decode_caveat="KV4 common-input attention oracle passes, but full cached decode is not bitwise equivalent to dense attention; limited 256-token diagnostics are recorded separately from prefill PPL.")
    (root/"summary.json").write_text(json.dumps(result, indent=2)+"\n")
    for row in rows:
        print(row["variant"], {bits: r["cuda"]["ppl"] for bits, r in row["results"].items()})


if __name__ == "__main__":
    main()
