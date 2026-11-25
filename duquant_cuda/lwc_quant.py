"""Memory-efficient LWC weight STE, preserving FP16 forward/backward roundings.

The weight is frozen. Only two per-row bound gradients are returned. Keep ATen
sum reductions so the gradient reduction order matches the unfused reference.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _weight_forward(X, S, Z, Y, K: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    x = tl.load(X + row * K + col, col < K, 0).to(tl.float32)
    s = tl.load(S + row).to(tl.float32)
    z = tl.load(Z + row).to(tl.float32)
    q = tl.extra.cuda.libdevice.nearbyint(tl.div_rn(x, s)).to(tl.float16).to(tl.float32)
    q = (q + z).to(tl.float16).to(tl.float32)
    centered = (tl.minimum(tl.maximum(q, 0), 15) - z).to(tl.float16).to(tl.float32)
    tl.store(Y + row * K + col, centered * s, col < K)


@triton.jit
def _weight_backward(X, G, S, Z, A, D, K: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    x = tl.load(X + row * K + col, col < K, 0).to(tl.float32)
    g = tl.load(G + row * K + col, col < K, 0).to(tl.float32)
    s = tl.load(S + row).to(tl.float32)
    z = tl.load(Z + row).to(tl.float32)
    ratio = tl.div_rn(x, s)
    q = tl.extra.cuda.libdevice.nearbyint(ratio).to(tl.float16).to(tl.float32)
    raw = (q + z).to(tl.float16).to(tl.float32)
    centered = (tl.minimum(tl.maximum(raw, 0), 15) - z).to(tl.float16).to(tl.float32)
    direct = (g * centered).to(tl.float16)
    dcode = tl.where((raw >= 0) & (raw <= 15), (g * s).to(tl.float16).to(tl.float32), 0)
    indirect = -dcode * tl.div_rn(ratio, s)
    tl.store(A + row * K + col, direct, col < K)
    tl.store(D + row * K + col, indirect, col < K)


class WeightSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, xmin, xmax, low, high):
        lo, hi = xmin * low, xmax * high
        pre_scale = (hi - lo) / 15
        scale = pre_scale.clamp(1.0e-5, 1.0e4)
        zero = (-lo / scale).clamp(-1.0e4, 1.0e4).round()
        y = torch.empty_like(x)
        _weight_forward[(x.shape[0], triton.cdiv(x.shape[1], 1024))](
            x, scale, zero, y, x.shape[1], 1024, enable_fp_fusion=False
        )
        ctx.save_for_backward(x, xmin, xmax, scale, zero, pre_scale, lo)
        return y

    @staticmethod
    def backward(ctx, grad):
        x, xmin, xmax, scale, zero, pre_scale, lo = ctx.saved_tensors
        grad = grad.contiguous()
        a = torch.empty_like(x)
        d = torch.empty(x.shape, device=x.device, dtype=torch.float32)
        _weight_backward[(x.shape[0], triton.cdiv(x.shape[1], 1024))](
            x, grad, scale, zero, a, d, x.shape[1], 1024, enable_fp_fusion=False
        )
        gscale = a.sum(-1, keepdim=True) + d.sum(-1, keepdim=True).half()
        # Round has zero derivative, but ATen still evaluates this FP16 div
        # backward. Preserve its nonfinite behavior too (0 * inf on degenerate
        # ranges), rather than silently changing AMP's overflow decisions.
        gscale = gscale - torch.zeros_like(scale) * ((-lo / scale) / scale)
        gscale = torch.where((pre_scale >= 1.0e-5) & (pre_scale <= 1.0e4), gscale, 0)
        gspan = gscale / 15
        return None, None, None, -gspan * xmin, gspan * xmax


def weight_qdq_ste(weight, xmin, xmax, low, high):
    return WeightSTE.apply(weight, xmin, xmax, low, high)
