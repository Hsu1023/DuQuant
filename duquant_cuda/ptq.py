"""Train-only clipping search and symmetric INT4 RTN/GPTQ export.

Signed codes [-8,7] are stored as uint4 with zero=8; the existing affine
integer kernels therefore execute symmetric weights without requantization.
GPTQ uses fixed per-output-channel FP16 scales, block=128, no act-order.
"""
import torch
import triton
import triton.language as tl

from .backend import affine_quant


def validate_weight(state, weight):
    q, s, z = (state[k] for k in ("codes", "scale", "zero"))
    n, k = weight.shape
    if (q.shape != (n, k) or q.dtype != torch.uint8 or s.shape != (n,)
            or z.shape != (n,) or s.dtype != torch.float16 or z.dtype != torch.float16
            or any(t.device != weight.device for t in (q, s, z))):
        raise ValueError("Invalid exported weight shape, dtype or device")
    if not (bool((q <= 15).all()) and bool(torch.isfinite(s).all())
            and bool((s > 0).all()) and bool((z == 8).all())):
        raise ValueError("Invalid symmetric INT4 codes/parameters")
    return q, s, z


@torch.no_grad()
def symmetric_rtn(weight, ratios):
    """Choose clipping per output row by weight reconstruction MSE."""
    codes = torch.empty_like(weight, dtype=torch.uint8)
    scales = torch.empty(weight.shape[0], device=weight.device, dtype=torch.float16)
    clips = torch.empty(weight.shape[0], device=weight.device)
    errors = torch.empty_like(clips)
    for start in range(0, len(weight), 256):
        w = weight[start:start+256].float()
        bound = w.abs().amax(1)
        best = torch.full_like(bound, float("inf"))
        for ratio in ratios:
            scale = (bound * ratio / 7).clamp(1.e-5, 1.e4).half()
            q = (w / scale[:, None]).round().clamp(-8, 7)
            error = (w-q*scale[:, None]).square().mean(1)
            take = error < best
            best = torch.minimum(best, error)
            scales[start:start+len(w)] = torch.where(take, scale, scales[start:start+len(w)])
            clips[start:start+len(w)] = torch.where(take, ratio, clips[start:start+len(w)])
        scale = scales[start:start+len(w)]
        codes[start:start+len(w)] = ((w/scale[:, None]).round().clamp(-8, 7)+8).byte()
        errors[start:start+len(w)] = best
    return dict(codes=codes, scale=scales, zero=torch.full_like(scales, 8)), dict(
        clip_min=clips.min().item(), clip_mean=clips.mean().item(), clip_max=clips.max().item(),
        weight_mse=errors.mean().item(), clips=clips.cpu(),
    )


@torch.no_grad()
def search_activation(x, ratios):
    """One clip per projection/cache kind; FP16 deployed affine QDQ MSE."""
    errors = []
    for ratio in ratios:
        total = 0.
        for chunk in x.split(512):
            q, s, z = affine_quant(chunk, ratio)
            y = (q.half()-z[:, None])*s[:, None]
            total += (y.float()-chunk.float()).square().sum().item()
        errors.append(total / x.numel())
    index = min(range(len(errors)), key=errors.__getitem__)
    return ratios[index], dict(ratios=ratios, mse=errors, chosen=ratios[index], rows=len(x))


@triton.jit
def _gptq_block(W, HI, SCALE, CODES, ERROR, N: tl.constexpr, K: tl.constexpr,
                START: tl.constexpr, B: tl.constexpr, R: tl.constexpr):
    row = tl.program_id(0)*R+tl.arange(0, R)
    col = tl.arange(0, B)
    w = tl.load(W+row[:, None]*K+START+col[None, :], row[:, None]<N, 0)
    s = tl.load(SCALE+row, row<N, 1).to(tl.float32)
    for i in range(B):
        wi = tl.sum(tl.where(col[None, :]==i, w, 0.), 1)
        code = tl.minimum(tl.maximum(tl.extra.cuda.libdevice.nearbyint(
            tl.div_rn(wi, s)), -8.), 7.)
        diagonal = tl.load(HI+(START+i)*K+START+i)
        error = tl.div_rn(wi-code*s, diagonal)
        tl.store(CODES+row*K+START+i, (code+8).to(tl.uint8), row<N)
        tl.store(ERROR+row*B+i, error, row<N)
        h = tl.load(HI+(START+i)*K+START+col)
        w = w-tl.where(col[None, :]>=i, error[:, None]*h[None, :], 0.)


@torch.no_grad()
def gptq(weight, hessian, scale, damp=0.01, block=128):
    """Standard GPTQ error feedback with a preselected fixed quantizer."""
    w = weight.float().clone()
    h = hessian.float().clone()
    n, k = w.shape
    if k % block:
        raise ValueError("GPTQ input width must be divisible by block size")
    dead = h.diag() == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    diagonal = torch.arange(k, device=w.device)
    damping = damp*h.diag().mean()
    h[diagonal, diagonal] += damping
    inv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    inv = inv.contiguous()
    codes = torch.empty_like(weight, dtype=torch.uint8)
    error = torch.empty((n, block), device=w.device)
    for start in range(0, k, block):
        _gptq_block[(triton.cdiv(n, 16),)](
            w, inv, scale, codes, error, n, k, start, block, 16,
            enable_fp_fusion=False, num_warps=4,
        )
        if start+block < k:
            w[:, start+block:] -= error @ inv[start:start+block, start+block:]
    return dict(codes=codes, scale=scale, zero=torch.full_like(scale, 8)), dict(
        damping=damping.item(), dead_columns=dead.sum().item(), blocksize=block, act_order=False,
    )
