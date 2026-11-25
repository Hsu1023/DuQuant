"""Train-only calibration: FP16 activation maxima, folded smoothing, DuQuant search."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from datasets import Dataset
from transformers import AutoTokenizer
from safetensors import safe_open
from duquant_cuda.decoder import load_layer_weights, LlamaDecoderBlock
from duquant_cuda.calibration import (
    apply_smoothing,
    smoothing_scales,
    install_rotation_base,
    CalibratingLinear,
)

TRAIN_ARROW = "/mnt/public_datasets/delta_skip/hf_cache/Salesforce___wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-train.arrow"


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/mnt/public_models/llama2-7b-hf")
    p.add_argument("--output", default="benchmarks/cache/calibrated_w4a4")
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--rotation-steps", type=int, default=256)
    p.add_argument("--alpha", type=float, default=0.6)
    args = p.parse_args()
    out = Path(args.output)
    if (out / "manifest.json").exists():
        raise FileExistsError(
            "Choose a new output directory; existing calibration is preserved"
        )
    out.mkdir(parents=True, exist_ok=True)
    if not (
        args.nsamples > 0
        and 2 <= args.seqlen <= 4096
        and 0 <= args.alpha <= 1
        and args.rotation_steps > 0
    ):
        p.error("Invalid calibration configuration")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = Dataset.from_file(TRAIN_ARROW)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=False, local_files_only=True
    )
    ids = tokenizer("\n\n".join(data["text"]), return_tensors="pt").input_ids
    starts = [
        random.randint(0, ids.numel() - args.seqlen - 1) for _ in range(args.nsamples)
    ]
    tokens = torch.cat([ids[:, start : start + args.seqlen] for start in starts])
    path = Path(args.model)
    cfg = json.loads((path / "config.json").read_text())
    index = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    with safe_open(str(path / index["model.embed_tokens.weight"]), framework="pt") as f:
        embedding = f.get_tensor("model.embed_tokens.weight").half().cuda()
    hidden = F.embedding(tokens.cuda(), embedding)
    del embedding
    rotate_input = hidden.mean(0, keepdim=True)
    manifest = dict(
        format_version=1,
        status="running",
        model=str(path.resolve()),
        config_sha256=hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
        dataset="Salesforce/wikitext",
        dataset_config="wikitext-2-raw-v1",
        split="train",
        source_arrow=TRAIN_ARROW,
        source_arrow_sha256=hashlib.sha256(Path(TRAIN_ARROW).read_bytes()).hexdigest(),
        token_ids_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
        tokenizer_class=type(tokenizer).__name__,
        tokenizer_use_fast=bool(tokenizer.is_fast),
        sample_starts=starts,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        seed=args.seed,
        alpha=args.alpha,
        rotation_steps=args.rotation_steps,
        block_size=128,
        algorithm="Repository UniformAffineQuantizer.online_duquant_cali; 2 rotations + zigzag; FP16 maxima smoothing folded as smooth_and_let_inplace; rotation representative is propagated mean calibration input",
        attention="Backend Flash SDPA; QK Hadamard + QK/V qdq",
        lwc_epochs=0,
        let=False,
        layers=[],
    )
    install_rotation_base()
    start_time = time.perf_counter()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for layer in range(cfg["num_hidden_layers"]):
            started = time.perf_counter()
            weights, _ = load_layer_weights(path, layer)
            fp = LlamaDecoderBlock(
                weights,
                {},
                "fp16",
                eps=cfg["rms_norm_eps"],
                max_position=cfg["max_position_embeddings"],
                rope_theta=cfg.get("rope_theta", 10000.0),
            )
            maxima = {}

            def record(name, x, table=maxima):
                value = x.reshape(-1, x.shape[-1]).abs().amax(0)
                table[name] = (
                    value if name not in table else torch.maximum(table[name], value)
                )

            handles = [
                getattr(fp, name).register_forward_pre_hook(
                    lambda m, inp, name=name: record(name, inp[0])
                )
                for name in ("qkv", "o", "gate_up", "down")
            ]
            cache = fp.cache(1, args.seqlen)
            for sample in range(args.nsamples):
                hidden[sample : sample + 1] = fp(hidden[sample : sample + 1], cache)
            for handle in handles:
                handle.remove()
            del fp, cache
            scales = smoothing_scales(weights, maxima, args.alpha)
            apply_smoothing(weights, scales)
            block = LlamaDecoderBlock(
                weights,
                {},
                "fp16",
                eps=cfg["rms_norm_eps"],
                max_position=cfg["max_position_embeddings"],
                rope_theta=cfg.get("rope_theta", 10000.0),
            )
            block.mode = "reference"
            for name in ("qkv", "o", "gate_up", "down"):
                linear = getattr(block, name)
                setattr(
                    block, name, CalibratingLinear(linear.weight, args.rotation_steps)
                )
            rotate_input = block(rotate_input, block.cache(1, args.seqlen))
            rotations = {
                name: tuple(t.cpu() for t in getattr(block, name).exported_rotations)
                for name in ("qkv", "o", "gate_up", "down")
            }
            payload = dict(
                scales={name: x.cpu() for name, x in scales.items()},
                rotations=rotations,
            )
            target = out / f"layer_{layer:02d}.pt"
            torch.save(payload, target)
            manifest["layers"].append(
                dict(
                    index=layer,
                    file=target.name,
                    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    seconds=time.perf_counter() - started,
                )
            )
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(
                f"Calibrated {layer + 1}/{cfg['num_hidden_layers']}: {time.perf_counter() - started:.1f}s",
                flush=True,
            )
            del weights, block, maxima, scales, linear
            torch.cuda.empty_cache()
    manifest.update(status="complete", elapsed_seconds=time.perf_counter() - start_time)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {k: manifest[k] for k in ("status", "elapsed_seconds", "nsamples", "seed")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
