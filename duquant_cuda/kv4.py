"""Packed asymmetric KV4 and split-K decode attention (head dimension 128).

Two codes per byte; FP16 scale/zero per token and KV head. Decode dequantizes
only tiles in registers, never materializing a full floating-point cache.
"""

import torch
import triton
import triton.language as tl

from .decoder import _hadamard


@triton.jit
def _quant(x, CLIP: tl.constexpr):
    lo = (tl.min(x, 1) * CLIP).to(tl.float16).to(tl.float32)
    hi = (tl.max(x, 1) * CLIP).to(tl.float16).to(tl.float32)
    span = (hi - lo).to(tl.float16).to(tl.float32)
    scale = tl.minimum(tl.maximum((span / 15).to(tl.float16).to(tl.float32), 1.e-5), 1.e4).to(tl.float16).to(tl.float32)
    zero = tl.extra.cuda.libdevice.nearbyint(tl.minimum(tl.maximum(
        tl.div_rn(-lo, scale).to(tl.float16).to(tl.float32), -1.e4), 1.e4))
    code = tl.extra.cuda.libdevice.nearbyint(tl.div_rn(x, scale[:, None])).to(tl.float16).to(tl.float32)
    code = tl.minimum(tl.maximum((code + zero[:, None]).to(tl.float16).to(tl.float32), 0), 15)
    dequant = ((code-zero[:, None]).to(tl.float16).to(tl.float32)*scale[:, None]).to(tl.float16)
    return code.to(tl.uint8), scale, zero, dequant


@triton.jit
def _rope_pack(X, COS, SIN, Q, KC, VC, KP, VP, KT, VT,
               POS, T: tl.constexpr, H: tl.constexpr, HEADS: tl.constexpr,
               KV_HEADS: tl.constexpr, BATCH: tl.constexpr, CAP: tl.constexpr,
               PREFILL: tl.constexpr, BH: tl.constexpr, K_CLIP: tl.constexpr, V_CLIP: tl.constexpr):
    row = tl.program_id(0)*BH + tl.arange(0, BH)
    b, t, head = row // (T*HEADS), row // HEADS % T, row % HEADS
    d = tl.arange(0, 128)
    base = (b*T+t)[:, None]*(H+2*KV_HEADS*128)+head[:, None]*128
    kv_mask = (b[:, None] < BATCH) & (head[:, None] < KV_HEADS)
    q = tl.load(X+base+d[None, :], b[:, None]<BATCH, 0).to(tl.float32)
    k = tl.load(X+base+H+d[None, :], kv_mask, 0).to(tl.float32)
    pair = (d+64)%128
    sign = tl.where(d<64, -1., 1.)
    qp = tl.load(X+base+pair[None, :], b[:, None]<BATCH, 0).to(tl.float32)*sign[None, :]
    kp = tl.load(X+base+H+pair[None, :], kv_mask, 0).to(tl.float32)*sign[None, :]
    co = tl.load(COS+(POS+t[:, None])*128+d[None, :], b[:, None]<BATCH, 0).to(tl.float32)
    si = tl.load(SIN+(POS+t[:, None])*128+d[None, :], b[:, None]<BATCH, 0).to(tl.float32)
    q = ((q*co).to(tl.float16).to(tl.float32)+(qp*si).to(tl.float16).to(tl.float32)).to(tl.float16).to(tl.float32)
    k = ((k*co).to(tl.float16).to(tl.float32)+(kp*si).to(tl.float16).to(tl.float32)).to(tl.float16).to(tl.float32)
    q, k = _hadamard(q, BH), _hadamard(k, BH)
    v = tl.load(X+base+H+KV_HEADS*128+d[None, :], kv_mask, 0).to(tl.float32)
    ki, ks, kz, kd = _quant(k, K_CLIP)
    vi, vs, vz, vd = _quant(v, V_CLIP)
    klo, khi = tl.split(ki.reshape((BH, 64, 2)))
    vlo, vhi = tl.split(vi.reshape((BH, 64, 2)))
    dst = (b*KV_HEADS+head)*CAP+POS+t
    mask = (b<BATCH)&(head<KV_HEADS)
    dd = tl.arange(0, 64)
    tl.store(KC+dst[:, None]*64+dd[None, :], klo | (khi << 4), mask[:, None])
    tl.store(VC+dst[:, None]*64+dd[None, :], vlo | (vhi << 4), mask[:, None])
    tl.store(KP+dst*2, ks, mask); tl.store(KP+dst*2+1, kz, mask)
    tl.store(VP+dst*2, vs, mask); tl.store(VP+dst*2+1, vz, mask)
    tl.store(Q+((b*HEADS+head)*T+t)[:, None]*128+d[None, :], q, b[:, None]<BATCH)
    if PREFILL:
        tmp = ((b*KV_HEADS+head)*T+t)[:, None]*128+d[None, :]
        tl.store(KT+tmp, kd, mask[:, None])
        tl.store(VT+tmp, vd, mask[:, None])


@triton.jit
def _decode_partial(Q, KC, VC, KP, VP, PART, LENGTH,
                    HEADS: tl.constexpr, KV_HEADS: tl.constexpr, CAP: tl.constexpr,
                    SPLITS: tl.constexpr, BN: tl.constexpr):
    bh, split = tl.program_id(0), tl.program_id(1)
    batch, head = bh//HEADS, bh%HEADS
    kvhead = head//(HEADS//KV_HEADS)
    t = split*BN+tl.arange(0, BN)
    d = tl.arange(0, 128)
    row = (batch*KV_HEADS+kvhead)*CAP+t
    mask = (t<LENGTH)&(t<CAP)
    kcode = tl.load(KC+row[:, None]*64+d[None, :]//2, mask[:, None], 0).to(tl.int32)
    vcode = tl.load(VC+row[:, None]*64+d[None, :]//2, mask[:, None], 0).to(tl.int32)
    kcode = (kcode >> ((d[None, :]%2)*4)) & 15
    vcode = (vcode >> ((d[None, :]%2)*4)) & 15
    ks = tl.load(KP+row*2, mask, 0).to(tl.float32)
    kz = tl.load(KP+row*2+1, mask, 0).to(tl.float32)
    vs = tl.load(VP+row*2, mask, 0).to(tl.float32)
    vz = tl.load(VP+row*2+1, mask, 0).to(tl.float32)
    k = ((kcode-kz[:, None]).to(tl.float16).to(tl.float32)*ks[:, None]).to(tl.float16).to(tl.float32)
    v = ((vcode-vz[:, None]).to(tl.float16).to(tl.float32)*vs[:, None]).to(tl.float16).to(tl.float32)
    q = tl.load(Q+bh*128+d).to(tl.float32)
    score = tl.sum(k*q[None, :], 1)*0.08838834764831845
    score = tl.where(mask, score, -1.e30)
    maximum = tl.max(score, 0)
    p = tl.where(mask, tl.exp(score-maximum), 0.)
    denom = tl.sum(p, 0)
    value = tl.sum(p[:, None]*v, 0)
    dst = (bh*SPLITS+split)*130
    tl.store(PART+dst+d, value)
    tl.store(PART+dst+128, maximum)
    tl.store(PART+dst+129, denom)


@triton.jit
def _decode_grouped(Q, KC, VC, KP, VP, PART, LENGTH,
                    HEADS: tl.constexpr, KV_HEADS: tl.constexpr, CAP: tl.constexpr,
                    SPLITS: tl.constexpr, BN: tl.constexpr, M: tl.constexpr):
    # A whole query-head group reuses each unpacked K/V tile. Two FP16
    # components of the softmax weights retain more precision than a single
    # FP16 cast while allowing Tensor Core PV products.
    bh, split = tl.program_id(0), tl.program_id(1)
    batch, kvhead = bh//KV_HEADS, bh%KV_HEADS
    GROUP: tl.constexpr = HEADS//KV_HEADS
    m, d = tl.arange(0, M), tl.arange(0, 128)
    t = split*BN+tl.arange(0, BN)
    row = (batch*KV_HEADS+kvhead)*CAP+t
    valid = (t<LENGTH)&(t<CAP)
    kc = tl.load(KC+row[:, None]*64+d[None, :]//2, valid[:, None], 0).to(tl.int32)
    vc = tl.load(VC+row[:, None]*64+d[None, :]//2, valid[:, None], 0).to(tl.int32)
    kc = (kc >> ((d[None, :]%2)*4)) & 15
    vc = (vc >> ((d[None, :]%2)*4)) & 15
    ks = tl.load(KP+row*2, valid, 0).to(tl.float32)
    kz = tl.load(KP+row*2+1, valid, 0).to(tl.float32)
    vs = tl.load(VP+row*2, valid, 0).to(tl.float32)
    vz = tl.load(VP+row*2+1, valid, 0).to(tl.float32)
    k = ((kc-kz[:, None]).to(tl.float16).to(tl.float32)*ks[:, None]).to(tl.float16)
    v = ((vc-vz[:, None]).to(tl.float16).to(tl.float32)*vs[:, None]).to(tl.float16)
    h = batch*HEADS+kvhead*GROUP+m
    q = tl.load(Q+h[:, None]*128+d[None, :], m[:, None]<GROUP, 0)
    score = tl.dot(q, tl.trans(k))*0.08838834764831845
    mask = (m[:, None]<GROUP)&valid[None, :]
    score = tl.where(mask, score, -1.e30)
    maximum = tl.max(score, 1)
    p = tl.where(mask, tl.exp(score-maximum[:, None]), 0.)
    denom = tl.sum(p, 1)
    hi = p.to(tl.float16)
    lo = (p-hi.to(tl.float32)).to(tl.float16)
    value = tl.dot(hi, v)+tl.dot(lo, v)
    dst = (h*SPLITS+split)*130
    tl.store(PART+dst[:, None]+d[None, :], value, m[:, None]<GROUP)
    tl.store(PART+dst+128, maximum, m<GROUP)
    tl.store(PART+dst+129, denom, m<GROUP)


@triton.jit
def _decode_merge(PART, OUT, SPLITS: tl.constexpr, BS: tl.constexpr):
    bh = tl.program_id(0)
    s = tl.arange(0, BS)
    d = tl.arange(0, 128)
    ptr = (bh*SPLITS+s)*130
    m = tl.load(PART+ptr+128, s<SPLITS, -1.e30)
    l = tl.load(PART+ptr+129, s<SPLITS, 0.)
    factor = tl.exp(m-tl.max(m, 0))
    denom = tl.sum(factor*l, 0)
    partial = tl.load(PART+ptr[:, None]+d[None, :], s[:, None]<SPLITS, 0.)
    result = tl.sum(partial*factor[:, None], 0)/denom
    tl.store(OUT+bh*128+d, result)


class PackedKVCache:
    def __init__(self, batch, heads, q_heads, capacity, device, k_clip=0.9, v_clip=1.0):
        if not all(0 < c <= 1 for c in (k_clip, v_clip)):
            raise ValueError("Invalid KV clipping")
        self.k_clip, self.v_clip = k_clip, v_clip
        self.k = torch.zeros((batch, heads, capacity, 64), device=device, dtype=torch.uint8)
        self.v = torch.zeros_like(self.k)
        self.k_params = torch.zeros((batch, heads, capacity, 2), device=device, dtype=torch.float16)
        self.v_params = torch.zeros_like(self.k_params)
        self.capacity, self.q_heads = capacity, q_heads
        self.tile = (32 if capacity<=256 else 64 if capacity<=2048 else 128) if q_heads!=heads else 128
        self.splits = triton.cdiv(capacity, self.tile)
        self.partial = torch.empty((batch, q_heads, self.splits, 130), device=device, dtype=torch.float32)

    @property
    def storage_bytes(self):
        return sum(x.numel()*x.element_size() for x in (self.k, self.v, self.k_params, self.v_params))

    @property
    def workspace_bytes(self):
        return self.partial.numel()*self.partial.element_size()

    def unpack(self, length):
        """Independent PyTorch diagnostic only; never used in CUDA inference."""
        result = []
        for codes, params in ((self.k, self.k_params), (self.v, self.v_params)):
            codes, params = codes[:, :, :length], params[:, :, :length]
            x = torch.stack((codes & 15, codes >> 4), -1).flatten(-2).half()
            result.append((x-params[..., 1:2])*params[..., 0:1])
        return tuple(result)

    def prepare(self, packed, cos, sin, position):
        b, t, width = packed.shape
        kv_heads = self.k.shape[1]
        h = width-2*kv_heads*128
        q = torch.empty((b, self.q_heads, t, 128), device=packed.device, dtype=packed.dtype)
        prefill = t>1
        kt = torch.empty((b, kv_heads, t, 128), device=packed.device, dtype=packed.dtype) if prefill else None
        vt = torch.empty_like(kt) if prefill else None
        _rope_pack[(triton.cdiv(b*t*self.q_heads, 16),)](
            packed, cos, sin, q, self.k, self.v, self.k_params, self.v_params, kt, vt,
            position, t, h, self.q_heads, kv_heads, b, self.capacity, prefill, 16,
            self.k_clip, self.v_clip,
            enable_fp_fusion=False,
        )
        return q, kt, vt

    def decode(self, q, length):
        out = torch.empty_like(q)
        if self.q_heads != self.k.shape[1]:
            groups = self.q_heads//self.k.shape[1]
            _decode_grouped[(q.shape[0]*self.k.shape[1], self.splits)](
                q, self.k, self.v, self.k_params, self.v_params, self.partial, length,
                self.q_heads, self.k.shape[1], self.capacity, self.splits, self.tile,
                max(16, triton.next_power_of_2(groups)),
                enable_fp_fusion=False, num_warps=4, num_stages=1,
            )
        else:
            _decode_partial[(q.shape[0]*self.q_heads, self.splits)](
                q, self.k, self.v, self.k_params, self.v_params, self.partial, length,
                self.q_heads, self.k.shape[1], self.capacity, self.splits, 128,
                enable_fp_fusion=False, num_warps=4,
            )
        _decode_merge[(q.shape[0]*self.q_heads,)](
            self.partial, out, self.splits, triton.next_power_of_2(self.splits),
            enable_fp_fusion=False, num_warps=4,
        )
        return out
