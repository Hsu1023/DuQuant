"""Require complete end-to-end evidence for every requested model family."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path("benchmarks/families")
    rows = []
    for name in ("llama1", "llama2", "llama3", "vicuna", "mistral"):
        p = root / name
        run = json.loads((p / "run.json").read_text())
        assert run["status"] == "complete"
        for source, expected_hash in run["source_sha256"].items():
            assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == expected_hash, source
        quality = json.loads((p / "ppl_summary.json").read_text())
        speed = json.loads((p / "model_summary.json").read_text())
        hf = json.loads((p / "hf_comparison.json").read_text())
        manifest = json.loads((p / "calibration_manifest.json").read_text())
        assert manifest["status"] == "complete" and manifest["lwc_epochs"] == 0
        assert hf["passed"]
        assert speed["calibration_manifest_sha256"] == quality["quantization"]["calibration_manifest_sha256"]
        diagnostic = quality["integer_reference_diagnostic"]
        assert diagnostic["block_count"] == quality["block_count"]
        assert diagnostic["scored_tokens"] == quality["scored_tokens"]
        assert manifest["nsamples"] == 128 and manifest["seqlen"] == 2048
        assert manifest["seed"] == 2 and manifest["alpha"] == 0.6
        assert manifest["rotation_steps"] == 256 and manifest["block_size"] == 128
        assert not manifest["let"]
        integer = quality["integer_reference_diagnostic"]["comparisons"]["cuda"]
        assert integer["top1_agreement_with_integer_reference"] == 1
        assert integer["max_per_block_nll_delta"] == 0 and integer["sampled_logits_bitwise_equal"]
        config = json.loads((Path(run["model"]) / "config.json").read_text())
        short_hf = json.loads((p / "model_fp16_p128_g32.json").read_text())["fp16_hf_validation"]
        # Earlier runs predate the explicit `passed` field. Recheck their
        # actual cases rather than requiring the newer serialization schema.
        assert short_hf.get("passed", True)
        assert len(short_hf["cases"]) == 12
        assert {(c["prompt_tokens"], c["decode_step"]) for c in short_hf["cases"]} == {
            (length, step) for length in (1, 17, 128) for step in range(4)
        }
        for case in short_hf["cases"]:
            assert case["logit_relative_l2"] < 0.02
            if not case["top1_agrees"]:
                assert case["accepted_exact_hf_tie"] and case["hf_top_margin"] == 0
                assert case["candidate_gap_on_hf"] <= case["hf_top_fp16_ulp"]
        rows.append(dict(
            family=name, model=run["model"], model_type=config["model_type"],
            q_heads=config["num_attention_heads"],
            kv_heads=config.get("num_key_value_heads", config["num_attention_heads"]),
            block_count=quality["block_count"], scored_tokens=quality["scored_tokens"],
            tokenizer=quality["dataset"]["tokenizer_class"],
            rotations_and_permutations_sha256=quality["quantization"]["rotations_and_permutations_sha256"],
            ppl={mode: value["ppl"] for mode, value in quality["metrics"].items()},
            hf=hf, short_hf=short_hf, cuda_vs_fake=quality["comparisons"]["cuda_vs_fake"],
            integer_reference=integer, generation=speed["results"],
        ))
    result = dict(status="complete", families=rows, protocol=dict(
        calibration="WikiText-2 train, 128 samples x 2048 tokens, seed 2, alpha .6, 256 rotation steps, block 128; no LWC/LET",
        ppl="All complete disjoint 2048-token WikiText-2 test blocks; tokenization is model-specific",
        generation="Single GPU, batch 1, prompt 128/1024, generate 32 tokens, 5 repetitions median; fixed-shape CUDA Graphs; real KV updates and greedy feedback",
        validation="HF FP16 full-test comparison plus short prefill/decode checks; independent integer-reference full-test predictions and NLL; sampled full-vocabulary logits",
        scope="One representative checkpoint per family; does not validate every model size, scaled RoPE, sliding-window Mistral, multi-GPU inference or W6A6",
    ))
    (root / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    for row in rows:
        print(row["family"], row["ppl"], [(g["prompt_tokens"], g["end_to_end_speedup"]) for g in row["generation"]])


if __name__ == "__main__":
    main()
