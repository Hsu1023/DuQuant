"""Freeze DuQuant rotations/smoothing; search clipping and compare RTN/GPTQ.

All fitting uses the original 128 x 2048 train samples. RTN emits three
ablations sharing the same searched quantizers. GPTQ reuses those quantizers,
but collects its own sequential A4 Hessians and propagates its own outputs.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from datasets import Dataset
from transformers import AutoTokenizer
from safetensors import safe_open
from duquant_cuda.backend import DuQuantLinearCUDA, transform
from duquant_cuda.calibration import apply_smoothing
from duquant_cuda.decoder import LlamaDecoderBlock, load_layer_weights, qdq
from duquant_cuda.ptq import symmetric_rtn, search_activation, gptq

NAMES = ("qkv", "o", "gate_up", "down")
W_RATIOS = [round(1.-i*.01, 4) for i in range(81)]
A_RATIOS = [round(1.-i*.025, 4) for i in range(25)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cpu_tree(x):
    if isinstance(x, torch.Tensor):
        return x.cpu()
    if isinstance(x, dict):
        return {k: cpu_tree(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return type(x)(cpu_tree(v) for v in x)
    return x


def sample_rows(x, count=64):
    x = x.reshape(-1, x.shape[-1])
    idx = torch.linspace(0, len(x)-1, min(count, len(x)), device=x.device).long()
    return x[idx].clone()


class CollectLinear(torch.nn.Module):
    def __init__(self, weight, rotations, act_clip=None, hessian=False):
        super().__init__()
        self.rotations = rotations
        self.weight = transform(weight, *rotations)
        self.act_clip = act_clip
        self.samples = []
        self.hessian = torch.zeros((weight.shape[1], weight.shape[1]), device=weight.device) if hessian else None
        self.tokens = 0

    def forward(self, x):
        z = transform(x, *self.rotations)
        if self.act_clip is not None:
            z = qdq(z, self.act_clip)
        if self.hessian is None:
            self.samples.append(sample_rows(z))
        else:
            flat = z.reshape(-1, z.shape[-1]).float()
            self.hessian.addmm_(flat.T, flat)
            self.tokens += len(flat)
        return F.linear(z, self.weight)


def groups(weights):
    return dict(qkv=torch.cat([weights[f"self_attn.{p}_proj.weight"] for p in ("q", "k", "v")]),
                o=weights["self_attn.o_proj.weight"],
                gate_up=torch.cat([weights[f"mlp.{p}_proj.weight"] for p in ("gate", "up")]),
                down=weights["mlp.down_proj.weight"])


def block_for(weights, rotations, cfg, plan=None, mode="cuda"):
    return LlamaDecoderBlock(weights, rotations, mode,
        eps=cfg["rms_norm_eps"], max_position=cfg["max_position_embeddings"],
        rope_theta=cfg.get("rope_theta", 10000.), kv_bits=4, quantization=plan)


def collect_kv(self, packed, cache, pos):
    b, t, _ = packed.shape
    _, k, v = packed.split((self.h, self.kv_h, self.kv_h), -1)
    k = k.reshape(b, t, self.kv_heads, 128).transpose(1, 2)
    co, si = self.cos[pos:pos+t], self.sin[pos:pos+t]
    k = (k*co+torch.cat((-k[..., 64:], k[..., :64]), -1)*si) @ self.hadamard
    self.k_samples.append(sample_rows(k))
    self.v_samples.append(sample_rows(v.reshape(-1, 128)))
    return LlamaDecoderBlock._qkv_reference(self, packed, cache, pos)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calibration", default="benchmarks/cache/calibrated_w4a4")
    p.add_argument("--output", default="benchmarks/cache/ptq_llama2_final")
    p.add_argument("--report", default="benchmarks/ptq_llama2_final")
    p.add_argument("--method", choices=["rtn", "gptq"], required=True)
    p.add_argument("--nsamples", type=int, default=128)
    p.add_argument("--max-layers", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    src, root, report = Path(args.calibration), Path(args.output), Path(args.report)
    base = json.loads((src/"manifest.json").read_text())
    if base["status"] != "complete" or base["split"] != "train" or base.get("lwc_epochs", 0):
        raise ValueError("Expected completed train-only calibration without LWC")
    if not 1 <= args.nsamples <= base["nsamples"]:
        raise ValueError("Invalid sample count")
    path = Path(base["model"])
    cfg = json.loads((path/"config.json").read_text())
    layers = min(args.max_layers or cfg["num_hidden_layers"], cfg["num_hidden_layers"])
    if sha(base["source_arrow"]) != base["source_arrow_sha256"]:
        raise ValueError("Training data hash mismatch")
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=False)
    data = Dataset.from_file(base["source_arrow"])
    ids = tok("\n\n".join(data["text"]), return_tensors="pt").input_ids
    t = base["seqlen"]
    all_tokens = torch.cat([ids[:, start:start+t] for start in base["sample_starts"]])
    if hashlib.sha256(all_tokens.numpy().tobytes()).hexdigest() != base["token_ids_sha256"]:
        raise ValueError("Training token hash mismatch")
    index = json.loads((path/"model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(str(path/index["model.embed_tokens.weight"]), framework="pt") as f:
        embedding = f.get_tensor("model.embed_tokens.weight").half().cuda()
    hidden = F.embedding(all_tokens[:args.nsamples].cuda(), embedding)
    del embedding, data, ids, all_tokens
    variants = ("sym_fixed", "sym_wsearch", "sym_search") if args.method == "rtn" else ("sym_gptq",)
    manifest, logs = {}, {}
    source_hashes = {name: sha(name) for name in ("benchmarks/search_ptq.py", "duquant_cuda/ptq.py",
        "duquant_cuda/backend.py", "duquant_cuda/decoder.py", "duquant_cuda/kv4.py")}
    for variant in variants:
        out = root/variant
        out.mkdir(parents=True, exist_ok=True)
        report.mkdir(parents=True, exist_ok=True)
        meta = copy.deepcopy(base)
        meta.update(status="running", layers=[], ptq=dict(
            variant=variant, method=args.method, weight_symmetric=True, weight_range=[-8, 7],
            activation_symmetric=False, kv_symmetric=False, weight_groupsize=-1,
            activation_groupsize=-1, kv_groupsize=128, nsamples=args.nsamples,
            weight_clip_grid=W_RATIOS, activation_kv_clip_grid=A_RATIOS,
            objective="W per-output-channel FP32 reconstruction MSE; A/K/V per-layer-kind FP16 QDQ MSE",
            activation_search_rows_per_sample=64, parent_manifest_sha256=sha(src/"manifest.json"),
            source_sha256=source_hashes, frozen_rotations=True, frozen_smoothing=True,
            gptq="block128, damp0.01, no act-order, fixed RTN-searched scales/clips; true sequential within block, own propagated A4 inputs" if args.method=="gptq" else None,
        ))
        if (out/"manifest.json").exists():
            if not args.resume:
                raise FileExistsError("Existing run: explicitly use --resume")
            saved = json.loads((out/"manifest.json").read_text())
            if saved["ptq"] != meta["ptq"]:
                raise ValueError("Resume settings/source mismatch")
            meta = saved
            logs[variant] = json.loads((report/f"{variant}_calibration.json").read_text())
        else:
            logs[variant] = []
        manifest[variant] = meta
    completed = len(manifest[variants[0]]["layers"])
    if any(len(manifest[v]["layers"]) != completed for v in variants):
        raise ValueError("Partial multi-variant checkpoint; preserve files and use a fresh output")
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for layer in range(layers):
            started = time.perf_counter()
            entry = base["layers"][layer]
            if sha(src/entry["file"]) != entry["sha256"]:
                raise ValueError("Parent layer integrity mismatch")
            payload = torch.load(src/entry["file"], map_location="cuda", weights_only=True)
            weights, _ = load_layer_weights(path, layer)
            apply_smoothing(weights, payload["scales"])
            rotations = payload["rotations"]
            ws = groups(weights)
            plans, details = {}, {}
            if layer < completed:
                variant = variants[-1]
                saved_entry = manifest[variant]["layers"][layer]
                saved_path = root/variant/saved_entry["file"]
                if sha(saved_path) != saved_entry["sha256"]:
                    raise ValueError("Resume layer integrity mismatch")
                saved = torch.load(saved_path, map_location="cuda", weights_only=True)
                block = block_for(weights, rotations, cfg, saved["quantization"])
                del saved
            elif args.method == "rtn":
                print(f"RTN layer {layer+1}/{layers}: collecting {args.nsamples} train sequences", flush=True)
                block = block_for(weights, rotations, cfg, mode="reference")
                for name in NAMES:
                    setattr(block, name, CollectLinear(ws[name], rotations[name]))
                block.k_samples, block.v_samples = [], []
                block._qkv_reference = types.MethodType(collect_kv, block)
                cache = block.cache(1, t)
                for j in range(args.nsamples):
                    block(hidden[j:j+1], cache)
                act_clips, kv_clips = {}, []
                searched, fixed = {}, {}
                for name in NAMES:
                    linear = getattr(block, name)
                    searched[name], wr = symmetric_rtn(linear.weight, W_RATIOS)
                    fixed[name], fr = symmetric_rtn(linear.weight, [.8])
                    act_clips[name], ar = search_activation(torch.cat(linear.samples), A_RATIOS)
                    details[name] = dict(weight={k: v for k, v in wr.items() if k!="clips"},
                                         fixed_weight_mse=fr["weight_mse"], activation=ar)
                for name, samples in (("k", block.k_samples), ("v", block.v_samples)):
                    clip, kr = search_activation(torch.cat(samples), A_RATIOS)
                    kv_clips.append(clip)
                    details[name] = kr
                plans = dict(
                    sym_fixed=dict(weights=fixed, act_clips={n:.9 for n in NAMES}, kv_clips=(.9, 1.)),
                    sym_wsearch=dict(weights=searched, act_clips={n:.9 for n in NAMES}, kv_clips=(.9, 1.)),
                    sym_search=dict(weights=searched, act_clips=act_clips, kv_clips=tuple(kv_clips)),
                )
                del block, cache
                block = block_for(weights, rotations, cfg, plans["sym_search"])
            else:
                parent = root/"sym_search"
                parent_meta = json.loads((parent/"manifest.json").read_text())
                if parent_meta["status"] != "complete" or parent_meta["ptq"]["nsamples"] != args.nsamples:
                    raise ValueError("GPTQ requires completed matched RTN search")
                pe = parent_meta["layers"][layer]
                if sha(parent/pe["file"]) != pe["sha256"]:
                    raise ValueError("RTN quantizer integrity mismatch")
                state = torch.load(parent/pe["file"], map_location="cuda", weights_only=True)
                plan = state["quantization"]
                block = block_for(weights, rotations, cfg, plan)
                cache = block.cache(1, t)
                for name in NAMES:
                    print(f"GPTQ layer {layer+1}/{layers} {name}: true-sequential A4 Hessian", flush=True)
                    collector = CollectLinear(ws[name], rotations[name], plan["act_clips"][name], hessian=True)
                    setattr(block, name, collector)
                    for j in range(args.nsamples):
                        block(hidden[j:j+1], cache)
                    collector.hessian.mul_(2./collector.tokens)
                    quant, detail = gptq(collector.weight, collector.hessian, plan["weights"][name]["scale"])
                    # Compare both weights against the exact same local Hessian.
                    def objective(q):
                        error = (q["codes"].float()-8)*q["scale"][:, None]-collector.weight.float()
                        return ((error @ collector.hessian)*error).sum().item()/error.shape[0]
                    detail.update(rtn_hessian_error=objective(plan["weights"][name]),
                                  gptq_hessian_error=objective(quant))
                    details[name] = detail
                    plan["weights"][name] = quant
                    setattr(block, name, DuQuantLinearCUDA(ws[name], *rotations[name],
                        quantized_weight=quant, act_clip=plan["act_clips"][name]))
                    del collector
                plans["sym_gptq"] = plan
                del state, cache
            cache = block.cache(1, t)
            for j in range(args.nsamples):
                hidden[j:j+1] = block(hidden[j:j+1], cache)
            if layer >= completed:
                elapsed = time.perf_counter()-started
                for variant in variants:
                    filename = root/variant/f"layer_{layer:02d}.pt"
                    torch.save(cpu_tree(dict(scales=payload["scales"], rotations=rotations,
                                             quantization=plans[variant])), filename)
                    manifest[variant]["layers"].append(dict(index=layer, file=filename.name, sha256=sha(filename)))
                    logs[variant].append(dict(layer=layer, seconds=elapsed, details=details))
                    (root/variant/"manifest.json").write_text(json.dumps(manifest[variant], indent=2)+"\n")
                    (report/f"{variant}_calibration.json").write_text(json.dumps(logs[variant], indent=2)+"\n")
                print(f"Saved layer {layer+1}/{layers}, {elapsed:.1f}s, A={plans[variants[-1]]['act_clips']}, KV={plans[variants[-1]]['kv_clips']}", flush=True)
            else:
                print(f"Replayed layer {layer+1}/{layers}", flush=True)
            del weights, ws, rotations, payload, plans, block, cache
            torch.cuda.empty_cache()
    for variant in variants:
        manifest[variant]["status"] = "complete" if layers == cfg["num_hidden_layers"] else "partial"
        manifest[variant]["ptq_elapsed_seconds"] = sum(r["seconds"] for r in logs[variant])
        (root/variant/"manifest.json").write_text(json.dumps(manifest[variant], indent=2)+"\n")
    print("Completed " + args.method, flush=True)


if __name__ == "__main__":
    main()
