"""Same-input stage diagnostics on the first training-text block (no test data)."""

import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from duquant_cuda.decoder import load_layer_weights, random_rotations, LlamaDecoderBlock
from duquant_cuda.backend import affine_quant, transform
from transformers import AutoTokenizer
from datasets import Dataset
from safetensors import safe_open
from torch.nn.attention import sdpa_kernel, SDPBackend

p = Path("/mnt/public_models/llama2-7b-hf")
d = Dataset.from_file(
    "/mnt/public_datasets/delta_skip/hf_cache/Salesforce___wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-train.arrow"
)
tok = (
    AutoTokenizer.from_pretrained(p, use_fast=False, local_files_only=True)(
        "\n\n".join(d["text"][:100]), return_tensors="pt"
    )
    .input_ids[:, :2048]
    .cuda()
)
idx = json.loads((p / "model.safetensors.index.json").read_text())["weight_map"]
with safe_open(str(p / idx["model.embed_tokens.weight"]), framework="pt") as f:
    emb = f.get_tensor("model.embed_tokens.weight").half().cuda()
x = F.embedding(tok, emb)
del emb
torch.manual_seed(42)
rows = []


def stats(a, b):
    a, b = a.float(), b.float()
    return {
        "rel_l2": ((a - b).norm() / b.norm()).item(),
        "different_fraction": (a != b).float().mean().item(),
        "max_abs": (a - b).abs().max().item(),
    }


with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    for i in range(4):
        w, cfg = load_layer_weights(p, i)
        r = random_rotations(4096, 11008, "cuda")
        a = LlamaDecoderBlock(w, r, "cuda")
        b = LlamaDecoderBlock(w, r, "integer_reference")
        aa, bb = {}, {}
        for model, dst in [(a, aa), (b, bb)]:
            for name in ["qkv", "o", "gate_up", "down"]:
                getattr(model, name).register_forward_hook(
                    lambda m, inp, out, n=name, dst=dst: dst.update(
                        {n + "_in": inp[0].clone(), n + "_out": out.clone()}
                    )
                )
        ya = a(x, a.cache(1, 2048))
        yb = b(x, b.cache(1, 2048))
        row = {
            "layer": i,
            "block_same_input": stats(ya, yb),
            "stages": {n: stats(aa[n], bb[n]) for n in aa},
        }
        for name in ["qkv", "o", "gate_up", "down"]:
            la, lb = getattr(a, name), getattr(b, name)
            z = bb[name + "_in"].contiguous()
            outa = la(z)
            outb = lb(z)
            packed, sc, zero, sums = la.activation(z)
            codes = (
                torch.stack((packed & 15, packed >> 4), -1).flatten(-2).reshape(z.shape)
            )
            qc, s, zp = affine_quant(transform(z, lb.r1, lb.perm, lb.r2), 0.9)
            row["stages"][name + "_isolated"] = {
                "output": stats(outa, outb),
                "codes": stats(codes, qc),
                "scales": stats(sc.flatten(), s.flatten()),
            }
        rows.append(row)
        print(json.dumps(row), flush=True)
        x = yb
        del a, b, w, r, ya, yb, aa, bb, la, lb, outa, outb
Path("benchmarks/numeric_probe_train.json").write_text(json.dumps(rows, indent=2))
