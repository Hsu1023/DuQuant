"""Layerwise block reconstruction on train only; freeze the existing calibration."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from datasets import Dataset
from transformers import AutoTokenizer
from safetensors import safe_open
from duquant_cuda.calibration import apply_smoothing
from duquant_cuda.decoder import LlamaDecoderBlock, load_layer_weights
from duquant_cuda.lwc import LWCDecoder


def train_step(block, inputs, target, optimizer, scaler):
    # Retry the same sample after AMP scale reduction; never count a skipped
    # overflow update as a completed training step.
    for attempt in range(16):
        optimizer.zero_grad(set_to_none=True)
        actual = block(inputs)
        loss = F.mse_loss(actual.float(), target.float())
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite LWC loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if any(param.grad is None for param in block.parameters()):
            raise RuntimeError("Disconnected LWC parameter gradient")
        finite = all(torch.isfinite(param.grad).all() for param in block.parameters())
        scaler.step(optimizer)
        scaler.update()
        if finite:
            return loss.detach(), attempt
    raise FloatingPointError("LWC gradients remain nonfinite after AMP retries")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calibration", default="benchmarks/cache/calibrated_w4a4")
    p.add_argument("--output", default="benchmarks/cache/lwc_w4a4")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument(
        "--fused-weight-quant",
        action="store_true",
        help="Use the numerically matched fused weight STE",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Replay and preserve completed layers, then continue",
    )
    p.add_argument(
        "--nsamples", type=int, default=0, help="0 uses every exported training sample"
    )
    p.add_argument(
        "--max-layers", type=int, default=0, help="Nonzero is a diagnostic partial run"
    )
    args = p.parse_args()
    src, out = Path(args.calibration), Path(args.output)
    base_bytes = (src / "manifest.json").read_bytes()
    base = json.loads(base_bytes)
    if (
        base["status"] != "complete"
        or base["split"] != "train"
        or base.get("lwc_epochs", 0)
    ):
        raise ValueError("Expected completed, train-only calibration without LWC")
    if args.epochs < 1 or args.lr <= 0 or args.nsamples < 0 or args.max_layers < 0:
        p.error("Invalid training configuration")
    if (out / "manifest.json").exists() and not args.resume:
        raise FileExistsError("Use a new output directory; existing run is preserved")
    out.mkdir(parents=True, exist_ok=True)
    path = Path(base["model"])
    cfg = json.loads((path / "config.json").read_text())
    nsamples = min(args.nsamples or base["nsamples"], base["nsamples"])
    num_layers = min(
        args.max_layers or cfg["num_hidden_layers"], cfg["num_hidden_layers"]
    )
    t = base["seqlen"]
    torch.manual_seed(base["seed"])
    if (
        hashlib.sha256(Path(base["source_arrow"]).read_bytes()).hexdigest()
        != base["source_arrow_sha256"]
    ):
        raise ValueError("Training data changed")
    data = Dataset.from_file(base["source_arrow"])
    tok = AutoTokenizer.from_pretrained(path, use_fast=False, local_files_only=True)
    ids = tok("\n\n".join(data["text"]), return_tensors="pt").input_ids
    all_tokens = torch.cat([ids[:, s : s + t] for s in base["sample_starts"]])
    if (
        hashlib.sha256(all_tokens.numpy().tobytes()).hexdigest()
        != base["token_ids_sha256"]
    ):
        raise ValueError("Training tokens changed")
    index = json.loads((path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    with safe_open(str(path / index["model.embed_tokens.weight"]), framework="pt") as f:
        embedding = f.get_tensor("model.embed_tokens.weight").half().cuda()
    with torch.no_grad():
        teacher_hidden = F.embedding(all_tokens[:nsamples].cuda(), embedding)
        quant_hidden = teacher_hidden.clone()
    del embedding
    manifest = copy.deepcopy(base)
    manifest.update(
        status="running",
        layers=[],
        lwc_epochs=args.epochs,
        lwc_lr=args.lr,
        lwc_nsamples=nsamples,
        lwc_init_logit=4.0,
        lwc_parent_manifest_sha256=hashlib.sha256(base_bytes).hexdigest(),
        lwc_objective="FP32 block-output MSE against original FP16 teacher; preceding trained fake blocks supply quantized inputs",
        lwc_parameterization="Independent per-output-channel lower/upper sigmoid factors; FP16 forward quantizer arithmetic; STE rounding; frozen rotation/smoothing; no LET",
        lwc_fixed_smoothing_alpha=base["alpha"],
    )
    logs = []
    completed = 0
    if args.resume:
        saved = json.loads((out / "manifest.json").read_text())
        for key in (
            "lwc_epochs",
            "lwc_lr",
            "lwc_nsamples",
            "lwc_parent_manifest_sha256",
        ):
            if saved[key] != manifest[key]:
                raise ValueError(f"Resume settings differ: {key}")
        manifest = saved
        logs = json.loads((out / "training.json").read_text())
        completed = len(manifest["layers"])
        if len(logs) != completed:
            raise ValueError("Incomplete checkpoint metadata")
        if completed > num_layers:
            raise ValueError("Resume cannot truncate completed layers")
    started = time.perf_counter()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for layer in range(num_layers):
            layer_start = time.perf_counter()
            weights, _ = load_layer_weights(path, layer)
            fp = LlamaDecoderBlock(
                weights,
                {},
                "fp16",
                eps=cfg["rms_norm_eps"],
                max_position=cfg["max_position_embeddings"],
                rope_theta=cfg.get("rope_theta", 10000.0),
            )
            cache = fp.cache(1, t)
            with torch.no_grad():
                for j in range(nsamples):
                    teacher_hidden[j : j + 1] = fp(teacher_hidden[j : j + 1], cache)
            del fp, cache
            entry = base["layers"][layer]
            filename = src / entry["file"]
            if hashlib.sha256(filename.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError("Calibration layer hash mismatch")
            payload = torch.load(filename, map_location="cuda", weights_only=True)
            apply_smoothing(weights, payload["scales"])
            if layer < completed:
                saved_entry = manifest["layers"][layer]
                trained_file = out / saved_entry["file"]
                if (
                    saved_entry["index"] != layer
                    or hashlib.sha256(trained_file.read_bytes()).hexdigest()
                    != saved_entry["sha256"]
                ):
                    raise ValueError("Trained layer integrity check failed")
                trained = torch.load(
                    trained_file, map_location="cuda", weights_only=True
                )
                deployed = LlamaDecoderBlock(
                    weights,
                    payload["rotations"],
                    "reference",
                    eps=cfg["rms_norm_eps"],
                    max_position=cfg["max_position_embeddings"],
                    rope_theta=cfg.get("rope_theta", 10000.0),
                    weight_clips=trained["weight_clips"],
                )
                cache = deployed.cache(1, t)
                with torch.no_grad():
                    for j in range(nsamples):
                        quant_hidden[j : j + 1] = deployed(
                            quant_hidden[j : j + 1], cache
                        )
                print(
                    f"Replayed completed LWC layer {layer + 1}/{num_layers}", flush=True
                )
                del weights, payload, trained, deployed, cache
                torch.cuda.empty_cache()
                continue
            block = LWCDecoder(weights, payload["rotations"], cfg)
            for name in ("qkv", "o", "gate_up", "down"):
                getattr(block, name).fused_weight_quant = args.fused_weight_quant
            optimizer = torch.optim.AdamW(
                block.parameters(), lr=args.lr, weight_decay=0
            )
            scaler = torch.amp.GradScaler("cuda", init_scale=1024)
            history = []
            overflow_retries = 0
            print(
                f"LWC layer {layer + 1}/{num_layers}: {nsamples} samples x {args.epochs} epochs",
                flush=True,
            )
            for epoch in range(args.epochs):
                losses = []
                for j in range(nsamples):
                    loss, retries = train_step(
                        block,
                        quant_hidden[j : j + 1],
                        teacher_hidden[j : j + 1],
                        optimizer,
                        scaler,
                    )
                    overflow_retries += retries
                    losses.append(loss)
                mean_loss = torch.stack(losses).mean().item()
                history.append(mean_loss)
                print(
                    f"layer {layer + 1} epoch {epoch + 1}/{args.epochs}: MSE={mean_loss:.7g}, elapsed={time.perf_counter() - layer_start:.1f}s",
                    flush=True,
                )
            bounds = block.export_bounds()
            # Validate training forward against the exact fake inference export before propagation.
            deployed = LlamaDecoderBlock(
                weights,
                payload["rotations"],
                "reference",
                eps=cfg["rms_norm_eps"],
                max_position=cfg["max_position_embeddings"],
                rope_theta=cfg.get("rope_theta", 10000.0),
                weight_clips={k: tuple(x.cuda() for x in v) for k, v in bounds.items()},
            )
            cache = deployed.cache(1, t)
            with torch.no_grad():
                expected = block(quant_hidden[:1])
                actual = deployed(quant_hidden[:1], cache)
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                for j in range(nsamples):
                    quant_hidden[j : j + 1] = deployed(quant_hidden[j : j + 1], cache)
            exported = dict(
                scales={k: v.cpu() for k, v in payload["scales"].items()},
                rotations={
                    k: tuple(x.cpu() for x in v)
                    for k, v in payload["rotations"].items()
                },
                weight_clips=bounds,
                lwc_logits={k: v.detach().cpu() for k, v in block.named_parameters()},
            )
            target = out / f"layer_{layer:02d}.pt"
            torch.save(exported, target)
            manifest["layers"].append(
                dict(
                    index=layer,
                    file=target.name,
                    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                )
            )
            logs.append(
                dict(
                    layer=layer,
                    epoch_train_mse=history,
                    seconds=time.perf_counter() - layer_start,
                    export_fake_bitwise_equal=True,
                    overflow_retries=overflow_retries,
                    fused_weight_quant=args.fused_weight_quant,
                    bounds={
                        k: {
                            "lower_min": v[0].min().item(),
                            "lower_mean": v[0].float().mean().item(),
                            "upper_min": v[1].min().item(),
                            "upper_mean": v[1].float().mean().item(),
                        }
                        for k, v in bounds.items()
                    },
                )
            )
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            (out / "training.json").write_text(json.dumps(logs, indent=2) + "\n")
            del (
                weights,
                block,
                deployed,
                optimizer,
                scaler,
                payload,
                cache,
                expected,
                actual,
                losses,
            )
            torch.cuda.empty_cache()
    manifest.update(
        status="complete" if num_layers == cfg["num_hidden_layers"] else "partial",
        lwc_elapsed_seconds=(
            sum(row["seconds"] for row in logs)
            if args.resume
            else time.perf_counter() - started
        ),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {k: manifest[k] for k in ("status", "lwc_epochs", "lwc_elapsed_seconds")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
