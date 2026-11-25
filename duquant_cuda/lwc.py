"""Differentiable per-channel LWC for the existing W4A4 inference semantics.

Only lower/upper clipping logits are parameters. Norms, original weights,
rotations, and permutations stay frozen. FP16 quantizer arithmetic matches the
export; STE supplies gradients through rounding, as in the upstream quantizer.
"""

import torch
import torch.nn.functional as F
from .backend import transform
from .decoder import LlamaDecoderBlock


def ste_round(x):
    return (x.round() - x).detach() + x


def qdq_ste(x, low, high=None):
    high = low if high is None else high
    lo = x.amin(-1, keepdim=True) * low
    hi = x.amax(-1, keepdim=True) * high
    scale = ((hi - lo) / 15).clamp(1.0e-5, 1.0e4)
    zero = (-lo / scale).clamp(-1.0e4, 1.0e4).round()
    codes = (ste_round(x.float() / scale).half() + zero).clamp(0, 15)
    return (codes - zero) * scale


class LWCLinear(torch.nn.Module):
    def __init__(self, w, rotations, init=4.0):
        super().__init__()
        with torch.no_grad():
            self.register_buffer("weight", transform(w, *rotations))
            self.register_buffer("xmin", self.weight.amin(-1, keepdim=True))
            self.register_buffer("xmax", self.weight.amax(-1, keepdim=True))
        self.fused_weight_quant = False
        for name, value in zip(("r1", "perm", "r2"), rotations):
            self.register_buffer(name, value)
        self.lower = torch.nn.Parameter(
            torch.full((w.shape[0], 1), init, device=w.device)
        )
        self.upper = torch.nn.Parameter(
            torch.full((w.shape[0], 1), init, device=w.device)
        )

    def bounds(self):
        return self.lower.sigmoid().half(), self.upper.sigmoid().half()

    def forward(self, x):
        x = qdq_ste(transform(x, self.r1, self.perm, self.r2), 0.9)
        if self.fused_weight_quant:
            from .lwc_quant import weight_qdq_ste

            weight = weight_qdq_ste(self.weight, self.xmin, self.xmax, *self.bounds())
        else:
            weight = qdq_ste(self.weight, *self.bounds())
        return F.linear(x, weight)


class LWCDecoder(torch.nn.Module):
    def __init__(self, weights, rotations, cfg):
        super().__init__()
        # Use the inference constructor only to share grouping and RoPE constants.
        block = LlamaDecoderBlock(
            weights,
            {},
            "fp16",
            eps=cfg["rms_norm_eps"],
            max_position=cfg["max_position_embeddings"],
            rope_theta=cfg.get("rope_theta", 10000.0),
        )
        self.h, self.i, self.heads, self.eps = block.h, block.i, block.heads, block.eps
        self.kv_h, self.kv_heads = block.kv_h, block.kv_heads
        for name in ("norm1", "norm2", "cos", "sin", "hadamard"):
            self.register_buffer(name, getattr(block, name))
        for name in ("qkv", "o", "gate_up", "down"):
            setattr(self, name, LWCLinear(getattr(block, name).weight, rotations[name]))

    def norm(self, x, w):
        return (
            x.float()
            * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        ).half() * w

    def forward(self, x):
        b, t, h = x.shape
        q, k, v = [
            z.reshape(b, t, heads, 128).transpose(1, 2)
            for z, heads in zip(
                self.qkv(self.norm(x, self.norm1)).split((h, self.kv_h, self.kv_h), -1),
                (self.heads, self.kv_heads, self.kv_heads),
            )
        ]

        def rope(z):
            return (
                z * self.cos[:t]
                + torch.cat((-z[..., 64:], z[..., :64]), -1) * self.sin[:t]
            )

        q = qdq_ste(rope(q) @ self.hadamard, 0.9)
        k = qdq_ste(rope(k) @ self.hadamard, 0.9)
        v = qdq_ste(v, 1.0)
        attn = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=self.heads != self.kv_heads
        )
        attn = attn.transpose(1, 2).contiguous().view(b, t, h)
        residual = x + self.o(attn)
        gu = self.gate_up(self.norm(residual, self.norm2))
        activated = F.silu(gu[..., : self.i]) * gu[..., self.i :]
        return residual + self.down(activated)

    @torch.no_grad()
    def export_bounds(self):
        return {
            name: tuple(x.detach().cpu() for x in getattr(self, name).bounds())
            for name in ("qkv", "o", "gate_up", "down")
        }
