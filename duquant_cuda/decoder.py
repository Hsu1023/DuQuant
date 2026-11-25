"""Complete LLaMA/Mistral decoder block with optional packed KV4 storage.

The default W4A4 path preserves legacy Q/K/V fake quantization in FP16 caches.
Explicit kv_bits=16 disables that extra quantization; kv_bits=4 keeps Q in
FP16 and uses compressed K/V with a fused, tiled decode implementation.
Train-only calibration and full-model accuracy are evaluated in CALIBRATED.md.
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .backend import DuQuantLinearCUDA, affine_quant, transform


def rms_reduction_width(n, h):
    # ATen Reduce.cuh: vectorized FP32 input (four scalars per lane), 512 threads.
    if h < 256 or h > 4096 or h & (h - 1):
        return 0
    height = min(16, 1 << (n.bit_length() - 1))
    return min(h // 4, 512 // height)


@triton.jit
def _rms(
    X,
    W,
    Y,
    R,
    RESIDUAL,
    N: tl.constexpr,
    H: tl.constexpr,
    EPS: tl.constexpr,
    ADD: tl.constexpr,
    B: tl.constexpr,
    WIDTH: tl.constexpr = 0,
):
    row = tl.program_id(0)
    j = tl.arange(0, B)
    x = tl.load(X + row * H + j, j < H, 0).to(tl.float32)
    if ADD:
        x = (
            (x + tl.load(RESIDUAL + row * H + j, j < H, 0).to(tl.float32))
            .to(tl.float16)
            .to(tl.float32)
        )
        tl.store(R + row * H + j, x, j < H)
    if WIDTH:
        # Match torch.square().mean(-1)'s sequential four-accumulator loop and
        # ascending warp shuffle order. A tree sum changes half-way roundings,
        # which can cross later INT4 bin boundaries despite tiny norm errors.
        lane = tl.arange(0, WIDTH)
        column = tl.arange(0, 4)
        accum = tl.full((WIDTH, 4), 0, tl.float32)
        for chunk in tl.static_range(H // (WIDTH * 4)):
            indices = (lane + chunk * WIDTH)[:, None] * 4 + column[None, :]
            value = tl.load(X + row * H + indices).to(tl.float32)
            if ADD:
                value = (
                    (value + tl.load(RESIDUAL + row * H + indices).to(tl.float32))
                    .to(tl.float16)
                    .to(tl.float32)
                )
            accum = accum + value * value
        even, odd = tl.split(accum.reshape((WIDTH, 2, 2)))
        v0, v2 = tl.split(even)
        v1, v3 = tl.split(odd)
        partial = ((v0 + v1) + v2) + v3
        for level in tl.static_range(0, 4):
            if (WIDTH >> (level + 1)) >= 32:
                partial = partial + tl.gather(
                    partial, (lane + (WIDTH >> (level + 1))) % WIDTH, 0
                )
        for offset in tl.static_range(0, 5):
            partial = partial + tl.gather(partial, (lane + (1 << offset)) % WIDTH, 0)
        var = tl.sum(tl.where(lane == 0, partial, 0), 0) / H
    else:
        var = tl.sum(x * x, 0) / H
    norm = (x * tl.rsqrt(var + EPS)).to(tl.float16).to(tl.float32)
    y = norm * tl.load(W + j, j < H, 0).to(tl.float32)
    tl.store(Y + row * H + j, y, j < H)


@triton.jit
def _silu_mul(X, Y, INTERMEDIATE: tl.constexpr, M: tl.constexpr, B: tl.constexpr):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    row = idx // INTERMEDIATE
    j = idx % INTERMEDIATE
    gate = tl.load(X + row * 2 * INTERMEDIATE + j, row < M, 0).to(tl.float32)
    up = tl.load(X + row * 2 * INTERMEDIATE + INTERMEDIATE + j, row < M, 0).to(
        tl.float32
    )
    silu = (
        tl.div_rn(gate, (1.0 + tl.extra.cuda.libdevice.exp(-gate)))
        .to(tl.float16)
        .to(tl.float32)
    )
    tl.store(Y + idx, silu * up, row < M)


@triton.jit
def _fake_quant_rows(x, CLIP: tl.constexpr):
    lo = (tl.min(x, 1) * CLIP).to(tl.float16).to(tl.float32)
    hi = (tl.max(x, 1) * CLIP).to(tl.float16).to(tl.float32)
    span = (hi - lo).to(tl.float16).to(tl.float32)
    s = (
        tl.minimum(tl.maximum((span / 15).to(tl.float16).to(tl.float32), 1.0e-5), 1.0e4)
        .to(tl.float16)
        .to(tl.float32)
    )
    z = tl.extra.cuda.libdevice.nearbyint(
        tl.minimum(
            tl.maximum(tl.div_rn(-lo, s).to(tl.float16).to(tl.float32), -1.0e4), 1.0e4
        )
    )
    q = (
        tl.extra.cuda.libdevice.nearbyint(tl.div_rn(x, s[:, None]))
        .to(tl.float16)
        .to(tl.float32)
    )
    q = tl.minimum(tl.maximum((q + z[:, None]).to(tl.float16).to(tl.float32), 0), 15)
    return (
        ((q - z[:, None]).to(tl.float16).to(tl.float32) * s[:, None])
        .to(tl.float16)
        .to(tl.float32)
    )


@triton.jit
def _hadamard(x, BH: tl.constexpr):
    d = tl.arange(0, 128)
    # Match the dense reference's Tensor Core accumulation order. Butterfly
    # halfway roundings can cross subsequent quantization boundaries.
    parity = tl.extra.cuda.libdevice.popc(d[:, None] & d[None, :]) & 1
    h = tl.where(parity == 0, 0.08837890625, -0.08837890625).to(tl.float16)
    return tl.dot(x.to(tl.float16), h).to(tl.float16).to(tl.float32)


@triton.jit
def _rope_cache(
    X,
    COS,
    SIN,
    Q,
    KC,
    VC,
    T: tl.constexpr,
    H: tl.constexpr,
    HEADS: tl.constexpr,
    BATCH: tl.constexpr,
    CAP: tl.constexpr,
    POS: tl.constexpr,
    QUANT: tl.constexpr,
    BH: tl.constexpr,
    KV_HEADS: tl.constexpr = 0,
):
    kv_heads: tl.constexpr = KV_HEADS if KV_HEADS else HEADS
    kv_h: tl.constexpr = kv_heads * 128
    row = tl.program_id(0) * BH + tl.arange(0, BH)
    b = row // (T * HEADS)
    t = row // HEADS % T
    head = row % HEADS
    d = tl.arange(0, 128)
    src = (b * T + t)[:, None] * (H + 2 * kv_h) + head[:, None] * 128 + d[None, :]
    # LLaMA's split-half RoPE convention.
    pair = (d + 64) % 128
    src_pair = (b * T + t)[:, None] * (H + 2 * kv_h) + head[:, None] * 128 + pair[None, :]
    kv_mask = (b[:, None] < BATCH) & (head[:, None] < kv_heads)
    q = tl.load(X + src, b[:, None] < BATCH, 0).to(tl.float32)
    k = tl.load(X + src + H, kv_mask, 0).to(tl.float32)
    qp = tl.load(X + src_pair, b[:, None] < BATCH, 0).to(tl.float32) * tl.where(
        d[None, :] < 64, -1.0, 1.0
    )
    kp = tl.load(X + src_pair + H, kv_mask, 0).to(tl.float32) * tl.where(
        d[None, :] < 64, -1.0, 1.0
    )
    co = tl.load(COS + (POS + t[:, None]) * 128 + d[None, :], b[:, None] < BATCH, 0).to(
        tl.float32
    )
    si = tl.load(SIN + (POS + t[:, None]) * 128 + d[None, :], b[:, None] < BATCH, 0).to(
        tl.float32
    )
    q = (
        (
            (q * co).to(tl.float16).to(tl.float32)
            + (qp * si).to(tl.float16).to(tl.float32)
        )
        .to(tl.float16)
        .to(tl.float32)
    )
    k = (
        (
            (k * co).to(tl.float16).to(tl.float32)
            + (kp * si).to(tl.float16).to(tl.float32)
        )
        .to(tl.float16)
        .to(tl.float32)
    )
    v = tl.load(X + src + H + kv_h, kv_mask, 0).to(tl.float32)
    if QUANT:
        q = _fake_quant_rows(_hadamard(q, BH), 0.9)
        k = _fake_quant_rows(_hadamard(k, BH), 0.9)
        v = _fake_quant_rows(v, 1.0)
    qdst = ((b * HEADS + head) * T + t)[:, None] * 128 + d[None, :]
    dst = ((b * kv_heads + head) * CAP + POS + t)[:, None] * 128 + d[None, :]
    tl.store(Q + qdst, q, b[:, None] < BATCH)
    tl.store(KC + dst, k, kv_mask)
    tl.store(VC + dst, v, kv_mask)


def hadamard_matrix(device):
    h = torch.ones(1, 1, device=device)
    for _ in range(7):
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return (h / (128**0.5)).half()


def qdq(x, clip):
    q, s, z = affine_quant(x, clip)
    return (q.half() - z[..., None]) * s[..., None]


class FakeLinear(torch.nn.Module):
    def __init__(self, w, rotations, weight_clip=0.8, act_clip=0.9, quantized_weight=None):
        super().__init__()
        r1, perm, r2 = rotations
        for name, value in zip(("r1", "perm", "r2"), rotations):
            self.register_buffer(name, value)
        self.act_clip = act_clip
        if quantized_weight is None:
            value = qdq(transform(w, r1, perm, r2), weight_clip)
        else:
            from .ptq import validate_weight
            q, s, z = validate_weight(quantized_weight, w)
            value = (q.half() - z[:, None]) * s[:, None]
        self.register_buffer("weight", value)

    def forward(self, x):
        return F.linear(
            qdq(transform(x, self.r1, self.perm, self.r2), self.act_clip), self.weight
        )


class IntegerReferenceLinear(torch.nn.Module):
    """Independent dense reference for affine INT4 arithmetic.

    Integer codes are represented exactly in FP32. For the tested K<=14336,
    unsigned dot products are <=225*K<2**24, so FP32 sums are exact integers.
    Scaling occurs after the dot, as required by real integer inference.
    """

    def __init__(self, w, rotations, weight_clip=0.8, act_clip=0.9, quantized_weight=None):
        super().__init__()
        for name, value in zip(("r1", "perm", "r2"), rotations):
            self.register_buffer(name, value)
        self.act_clip = act_clip
        if quantized_weight is None:
            q, s, z = affine_quant(transform(w, *rotations), weight_clip)
        else:
            from .ptq import validate_weight
            q, s, z = validate_weight(quantized_weight, w)
        self.register_buffer("codes", q.float())
        self.register_buffer("scale", s.float())
        self.register_buffer("zero", z.float())
        self.register_buffer("correction", w.shape[1] * z.float() - q.float().sum(-1))

    def forward(self, x):
        q, s, z = affine_quant(transform(x, self.r1, self.perm, self.r2), self.act_clip)
        c = F.linear(q.float(), self.codes)
        c = (
            c
            + z.float()[..., None] * self.correction
            - self.zero * q.float().sum(-1, keepdim=True)
        )
        return (c * s.float()[..., None] * self.scale).half()


class FloatLinear(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.register_buffer("weight", w)

    def forward(self, x):
        return F.linear(x, self.weight)


class KVCache:
    def __init__(self, batch, heads, capacity, device):
        self.k = torch.zeros(
            batch, heads, capacity, 128, device=device, dtype=torch.float16
        )
        self.v = torch.zeros_like(self.k)
        self.capacity = capacity

    @property
    def storage_bytes(self):
        return self.k.numel()*self.k.element_size()+self.v.numel()*self.v.element_size()

    workspace_bytes = 0


class LlamaDecoderBlock(torch.nn.Module):
    """Batch-first LLaMA/Mistral MHA or GQA block with static KV caches.

    Modes: fp16, cuda (W4A4), reference (PyTorch fake W4A4).
    Chunked prefill with a nonempty cache is intentionally unsupported. A graph
    captures one fixed cache position, not an entire growing generation loop.
    """

    def __init__(
        self,
        weights,
        rotations,
        mode="cuda",
        eps=1.0e-5,
        max_position=4096,
        rope_theta=10000.0,
        weight_clips=None,
        kv_bits=None,
        quantization=None,
    ):
        super().__init__()
        if mode not in ("cuda", "fp16", "reference", "integer_reference"):
            raise ValueError(mode)
        self.mode = mode
        if kv_bits not in (None, 4, 16) or (mode == "fp16" and kv_bits == 4):
            raise ValueError("kv_bits must be None (legacy), 4, or 16; FP16 baseline requires KV16")
        self.kv_bits = kv_bits
        if quantization is not None and (kv_bits not in (4, 16) or mode == "fp16"):
            raise ValueError("PTQ exports require explicit KV4/KV16 and a quantized mode")
        self.k_clip, self.v_clip = (quantization["kv_clips"] if quantization else (0.9, 1.0))
        if not all(0 < c <= 1 for c in (self.k_clip, self.v_clip)):
            raise ValueError("Invalid KV clipping")
        self.h = weights["self_attn.q_proj.weight"].shape[1]
        self.heads = self.h // 128
        self.i = weights["mlp.up_proj.weight"].shape[0]
        self.eps = eps
        if self.h % 128:
            raise ValueError("Only head_dim=128 is supported")
        self.kv_h = weights["self_attn.k_proj.weight"].shape[0]
        self.kv_heads = self.kv_h // 128
        if (self.kv_h % 128 or not self.kv_heads or self.heads % self.kv_heads
                or weights["self_attn.v_proj.weight"].shape[0] != self.kv_h
                or weights["self_attn.q_proj.weight"].shape[0] != self.h):
            raise ValueError("Invalid MHA/GQA projection dimensions")
        groups = {
            "qkv": torch.cat(
                [weights[f"self_attn.{p}_proj.weight"] for p in ("q", "k", "v")]
            ),
            "o": weights["self_attn.o_proj.weight"],
            "gate_up": torch.cat(
                [weights[f"mlp.{p}_proj.weight"] for p in ("gate", "up")]
            ),
            "down": weights["mlp.down_proj.weight"],
        }
        for name, w in groups.items():
            clip = weight_clips[name] if weight_clips is not None else 0.8
            options = {} if quantization is None else dict(
                quantized_weight=quantization["weights"][name],
                act_clip=quantization["act_clips"][name],
            )
            if mode == "cuda":
                linear = DuQuantLinearCUDA(w, *rotations[name], weight_clip=clip, **options)
            elif mode == "integer_reference":
                linear = IntegerReferenceLinear(w, rotations[name], weight_clip=clip, **options)
            elif mode == "reference":
                linear = FakeLinear(w, rotations[name], weight_clip=clip, **options)
            else:
                linear = FloatLinear(w)
            setattr(self, name, linear)
        self.register_buffer("norm1", weights["input_layernorm.weight"])
        self.register_buffer("norm2", weights["post_attention_layernorm.weight"])
        device = self.norm1.device
        inv = 1.0 / (
            rope_theta
            ** (torch.arange(0, 128, 2, device=device, dtype=torch.float32) / 128)
        )
        freq = (
            torch.arange(max_position, device=device, dtype=torch.float32)[:, None]
            * inv[None, :]
        )
        emb = torch.cat((freq, freq), -1)
        self.register_buffer("cos", emb.cos().half())
        self.register_buffer("sin", emb.sin().half())
        self.register_buffer("hadamard", hadamard_matrix(device))

    def cache(self, batch, capacity):
        if self.kv_bits == 4 and self.mode == "cuda":
            from .kv4 import PackedKVCache
            return PackedKVCache(batch, self.kv_heads, self.heads, capacity, self.norm1.device,
                                 k_clip=self.k_clip, v_clip=self.v_clip)
        return KVCache(batch, self.kv_heads, capacity, self.norm1.device)

    def _norm(self, x, w, residual=None):
        if self.mode in ("reference", "integer_reference"):
            if residual is not None:
                x = x + residual
            y = (
                x.float()
                * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
            ).half() * w
            return y, x
        y = torch.empty_like(x)
        r = torch.empty_like(x) if residual is not None else x
        _rms[(x.numel() // self.h,)](
            x,
            w,
            y,
            r,
            residual,
            x.numel() // self.h,
            self.h,
            self.eps,
            residual is not None,
            triton.next_power_of_2(self.h),
            rms_reduction_width(x.numel() // self.h, self.h),
            enable_fp_fusion=False,
        )
        return y, r

    def _qkv_reference(self, packed, cache, pos):
        b, t, _ = packed.shape
        q, k, v = [
            z.reshape(b, t, heads, 128).transpose(1, 2)
            for z, heads in zip(
                packed.split((self.h, self.kv_h, self.kv_h), -1),
                (self.heads, self.kv_heads, self.kv_heads),
            )
        ]
        co = self.cos[pos : pos + t]
        si = self.sin[pos : pos + t]

        def rope(z):
            return z * co + torch.cat((-z[..., 64:], z[..., :64]), -1) * si

        q, k = rope(q), rope(k)
        if self.kv_bits != 16:
            q, k = q @ self.hadamard, k @ self.hadamard
            if self.kv_bits is None:
                q = qdq(q, 0.9)
            k = qdq(k, self.k_clip)
            v = qdq(v, self.v_clip)
        cache.k[:, :, pos : pos + t].copy_(k)
        cache.v[:, :, pos : pos + t].copy_(v)
        return q

    def forward(self, x, cache, position=0):
        b, t, h = x.shape
        if x.dtype != torch.float16 or h != self.h or not x.is_contiguous():
            raise ValueError("Expected contiguous FP16 [B,T,H]")
        if cache.k.device != x.device or cache.k.shape[:2] != (b, self.kv_heads):
            raise ValueError("KV cache device/batch/head mismatch")
        if (
            position < 0
            or position + t > cache.capacity
            or position + t > len(self.cos)
        ):
            raise ValueError("Cache or RoPE position out of bounds")
        if position and t != 1:
            raise ValueError("Only prefill from zero or single-token decode supported")
        normalized, _ = self._norm(x, self.norm1)
        packed = self.qkv(normalized)
        compressed = self.kv_bits == 4 and self.mode == "cuda"
        if compressed:
            q, keys, values = cache.prepare(packed, self.cos, self.sin, position)
        elif self.mode in ("reference", "integer_reference"):
            q = self._qkv_reference(packed, cache, position)
        else:
            q = torch.empty((b, self.heads, t, 128), device=x.device, dtype=x.dtype)
            _rope_cache[(triton.cdiv(b * t * self.heads, 16),)](
                packed,
                self.cos,
                self.sin,
                q,
                cache.k,
                cache.v,
                t,
                h,
                self.heads,
                b,
                cache.capacity,
                position,
                self.mode == "cuda" and self.kv_bits is None,
                16,
                KV_HEADS=self.kv_heads,
                enable_fp_fusion=False,
            )
        # For single-token decode, every stored key is visible; top-left causal
        # SDPA would incorrectly mask all but the first key when q_len=1.
        if compressed and t == 1:
            attn = cache.decode(q, position+1)
        elif self.kv_bits == 4 and t == 1 and self.mode in ("reference", "integer_reference"):
            # Independent dense reference for KV4 decode's FP32 softmax and
            # accumulation. FlashAttention rounds intermediate probabilities
            # to FP16, so it is not a bitwise reference for this algorithm.
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.MATH):
                attn = F.scaled_dot_product_attention(
                    q.float(), cache.k[:, :, :position+t].float(),
                    cache.v[:, :, :position+t].float(),
                    enable_gqa=self.heads != self.kv_heads,
                ).to(x.dtype)
        else:
            attn = F.scaled_dot_product_attention(
                q,
                keys if compressed else cache.k[:, :, : position + t],
                values if compressed else cache.v[:, :, : position + t],
                is_causal=position == 0 and t > 1,
                enable_gqa=self.heads != self.kv_heads,
            )
        attn = attn.transpose(1, 2).contiguous().view(b, t, h)
        projected = self.o(attn)
        normalized, residual = self._norm(projected, self.norm2, x)
        gu = self.gate_up(normalized)
        if self.mode in ("reference", "integer_reference"):
            activated = F.silu(gu[..., : self.i]) * gu[..., self.i :]
        else:
            activated = torch.empty((b, t, self.i), device=x.device, dtype=x.dtype)
            _silu_mul[(triton.cdiv(b * t * self.i, 1024),)](
                gu, activated, self.i, b * t, 1024
            )
        return residual + self.down(activated)


def load_layer_weights(model_path, layer=0, device="cuda"):
    from safetensors import safe_open

    model_path = Path(model_path)
    index = json.loads((model_path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = f"model.layers.{layer}."
    weights = {}
    for shard in set(v for k, v in index.items() if k.startswith(prefix)):
        with safe_open(str(model_path / shard), framework="pt", device="cpu") as f:
            for k, v in index.items():
                if k.startswith(prefix) and v == shard:
                    weights[k[len(prefix) :]] = f.get_tensor(k).to(
                        device=device, dtype=torch.float16
                    )
    return weights, json.loads((model_path / "config.json").read_text())


def random_rotations(h, i, device):
    result = {}
    for name, k in [("qkv", h), ("o", h), ("gate_up", h), ("down", i)]:
        r = (
            torch.linalg.qr(torch.randn(2, 128, 128, device=device))[0]
            .half()
            .contiguous()
        )
        result[name] = (r[0], torch.randperm(k, device=device), r[1])
    return result
