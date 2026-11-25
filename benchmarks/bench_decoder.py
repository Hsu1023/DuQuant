"""Full decoder block timing, actual LLaMA weights and token embeddings.

Includes norms, all seven projections, RoPE, Q/K/V quantization in W4A4,
cache writes, FP16 Flash SDPA, SwiGLU and residuals. Rotation parameters
are synthetic, so this is not a calibrated-model quality evaluation.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from duquant_cuda.decoder import LlamaDecoderBlock, load_layer_weights, random_rotations
from bench_single_layer import timed, timed_cold


def embedded_prompt(path, length, batch):
    from transformers import AutoTokenizer
    from safetensors import safe_open

    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    text = "The purpose of this experiment is to measure complete transformer decoder latency. Quantization changes the representation of weights and activations while preserving the structure of attention and feed forward computation. "
    ids = tok.encode(text, add_special_tokens=False)
    ids = torch.tensor(([tok.bos_token_id] + ids * ((length // len(ids)) + 1))[:length])
    index = json.loads((Path(path) / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    key = "model.embed_tokens.weight"
    with safe_open(str(Path(path) / index[key]), framework="pt", device="cpu") as f:
        embedded = torch.nn.functional.embedding(ids, f.get_tensor(key))
    return (
        embedded[None]
        .expand(batch, -1, -1)
        .contiguous()
        .to(device="cuda", dtype=torch.float16)
    )


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/mnt/public_models/llama2-7b-hf")
    p.add_argument("--prefill", default="128,512,2048")
    p.add_argument("--decode-context", default="128,2048,4095")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--output", default="benchmarks/decoder_results.json")
    p.add_argument("--cold", action="store_true")
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    torch.manual_seed(42)
    weights, cfg = load_layer_weights(args.model)
    rotations = random_rotations(cfg["hidden_size"], cfg["intermediate_size"], "cuda")
    models = {
        mode: LlamaDecoderBlock(
            weights,
            rotations,
            mode,
            eps=cfg["rms_norm_eps"],
            max_position=cfg["max_position_embeddings"],
        )
        for mode in ("fp16", "cuda", "reference", "integer_reference")
    }
    scenarios = [("prefill", int(v), 0) for v in args.prefill.split(",") if v] + [
        ("decode", 1, int(v)) for v in args.decode_context.split(",") if v
    ]
    tokens = embedded_prompt(
        args.model, max(t + c for _, t, c in scenarios), args.batch
    )
    metadata = dict(
        gpu=torch.cuda.get_device_name(),
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        torch=torch.__version__,
        model=args.model,
        layer=0,
        batch=args.batch,
        attention="forced FP16 Flash SDPA for all modes",
        kv_storage="FP16, quantized-dequantized K/V in W4A4",
        input="actual token embeddings of repeated fixed prose",
        rotation="fixed random orthogonal matrices and permutations, shared by QKV and gate/up, not calibrated",
        timing="median of repeated CUDA Graph measurements, full block; fixed position for each decode scenario, historical cache prepared outside timing",
        results=[],
    )
    Path(args.output).write_text(json.dumps(metadata, indent=2) + "\n")
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for phase, t, context in scenarios:
            caches = {
                mode: model.cache(args.batch, t + context)
                for mode, model in models.items()
            }
            if context:
                for mode, model in models.items():
                    model(tokens[:, :context].contiguous(), caches[mode])
            x = tokens[:, context : context + t].contiguous()
            outputs = {
                mode: model(x, caches[mode], context) for mode, model in models.items()
            }
            torch.cuda.synchronize()
            ref = outputs["integer_reference"].float()
            fake = outputs["reference"].float()
            fast = outputs["cuda"].float()
            fp = outputs["fp16"].float()
            impl_rel = ((fast - ref).norm() / ref.norm()).item()
            # Quantization boundaries can amplify small FP16 differences through
            # the block; report actual errors and reject large implementation drift.
            if not torch.isfinite(fast).all() or impl_rel > 0.03:
                raise AssertionError(f"Full-block reference mismatch: {impl_rel}")
            measurements = {mode: [] for mode in ("fp16", "cuda")}
            for rep in range(args.repeats):
                for mode in ("fp16", "cuda") if rep % 2 == 0 else ("cuda", "fp16"):
                    measurements[mode].append(
                        timed(lambda: models[mode](x, caches[mode], context))
                    )
            fp_us = statistics.median(measurements["fp16"])
            q_us = statistics.median(measurements["cuda"])
            row = dict(
                phase=phase,
                batch=args.batch,
                tokens=t,
                past_tokens=context,
                fp16_us=fp_us,
                duquant_us=q_us,
                speedup=fp_us / q_us,
                samples_us=measurements,
                implementation_relative_l2=impl_rel,
                implementation_max_abs=(fast - ref).abs().max().item(),
                legacy_fake_relative_l2=((fast - fake).norm() / fake.norm()).item(),
                quantization_relative_l2_vs_fp16=((ref - fp).norm() / fp.norm()).item(),
            )
            if args.cold:
                a = timed_cold(lambda: models["fp16"](x, caches["fp16"], context))
                b = timed_cold(lambda: models["cuda"](x, caches["cuda"], context))
                row.update(cold_fp16_us=a, cold_duquant_us=b, cold_speedup=a / b)
            metadata["results"].append(row)
            Path(args.output).write_text(json.dumps(metadata, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
