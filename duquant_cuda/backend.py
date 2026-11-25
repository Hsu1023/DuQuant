"""Experimental single-layer DuQuant W4A4 inference (FP16, sm_89 / sm_90).

CUTLASS is an external build dependency. Set CUTLASS_PATH to its checkout.
Weights are transformed and packed once; every forward transforms and quantizes
its actual input. No output/activation caching is used.
"""

import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import torch
import triton
import triton.language as tl

_LIB = None
_LIB90 = None


def _arch(device=None):
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


def _nvcc():
    cuda = os.environ.get("CUDA_HOME")
    if cuda:
        return str(Path(cuda) / "bin/nvcc")
    found = shutil.which("nvcc")
    return found or "/usr/local/cuda-12.4/bin/nvcc"


def _build(src, so, arch, deps=()):
    root = src.parent
    build = root / "build"
    build.mkdir(exist_ok=True)
    if so.exists() and so.stat().st_mtime >= max(
        p.stat().st_mtime for p in (src, *deps)
    ):
        return
    cutlass = os.environ.get("CUTLASS_PATH")
    if not cutlass:
        raise RuntimeError("Set CUTLASS_PATH to a CUTLASS checkout (tested with 3.x).")
    log = build / (so.stem + "_compile.log")
    result = subprocess.run(
        [
            _nvcc(),
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-DNDEBUG",
            f"-arch={arch}",
            "--shared",
            "-Xcompiler=-fPIC",
            "-I" + str(Path(cutlass) / "include"),
            "-I" + str(Path(cutlass) / "tools/util/include"),
            str(src),
            "-o",
            str(so) + ".tmp",
        ],
        capture_output=True,
        text=True,
    )
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"CUDA compilation failed; see {log}")
    os.replace(str(so) + ".tmp", so)


def load_gemm():
    """Ada/Ampere path: CUTLASS Sm80 u4 x u4 IMMA (packed activations)."""
    global _LIB
    if _LIB is not None:
        return _LIB
    root = Path(__file__).resolve().parent
    so = root / "build" / "gemm.so"
    _build(root / "gemm.cu", so, "sm_89", (root / "fused_gemm.cuh",))
    lib = ctypes.CDLL(str(so))
    lib.duquant_gemm.argtypes = (
        [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p, ctypes.c_int]
    )
    lib.duquant_gemm.restype = ctypes.c_int
    lib.duquant_gemm_fused.argtypes = lib.duquant_gemm.argtypes + [ctypes.c_void_p] * 7
    lib.duquant_gemm_fused.restype = ctypes.c_int
    _LIB = lib
    return lib


def load_gemm_sm90():
    """Hopper path: packed u4 weights widened in registers, s8 activations, INT8 wgmma."""
    global _LIB90
    if _LIB90 is not None:
        return _LIB90
    root = Path(__file__).resolve().parent
    so = root / "build" / "gemm_sm90.so"
    _build(root / "gemm_sm90.cu", so, "sm_90a", (root / "small_m.cuh", root / "linear_small.cuh"))
    lib = ctypes.CDLL(str(so))
    lib.duquant_sm90_gemm.argtypes = (
        [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p, ctypes.c_int]
    )
    lib.duquant_sm90_gemm.restype = ctypes.c_int
    lib.duquant_sm90_gemm_fused.argtypes = (
        lib.duquant_sm90_gemm.argtypes + [ctypes.c_void_p] * 7
    )
    lib.duquant_sm90_gemm_fused.restype = ctypes.c_int
    lib.duquant_gemv_u4.argtypes = lib.duquant_sm90_gemm_fused.argtypes
    lib.duquant_gemv_u4.restype = ctypes.c_int
    lib.duquant_linear_small.argtypes = (
        [ctypes.c_void_p] * 6
        + [ctypes.c_int] * 3
        + [ctypes.c_float, ctypes.c_void_p, ctypes.c_int]
        + [ctypes.c_void_p] * 11
        + [ctypes.c_int, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_void_p]
        + [ctypes.c_void_p] * 3
        + [ctypes.c_int, ctypes.c_void_p, ctypes.c_longlong]
    )
    lib.duquant_linear_small.restype = ctypes.c_int
    _LIB90 = lib
    return lib


@triton.jit
def _rotate(
    X,
    R,
    P,
    Y,
    M: tl.constexpr,
    K: tl.constexpr,
    PERM: tl.constexpr,
    BM: tl.constexpr,
    TRANS: tl.constexpr = False,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    block = tl.program_id(1)
    kk = tl.arange(0, 128)
    src = block * 128 + kk
    if PERM:
        src = tl.load(P + src)
    if TRANS and PERM:
        x = tl.load(X + rows[:, None] + src[None, :] * M, rows[:, None] < M, 0)
    else:
        x = tl.load(X + rows[:, None] * K + src[None, :], rows[:, None] < M, 0)
    r = tl.load(R + kk[:, None] * 128 + kk[None, :])
    y = tl.dot(x, r).to(tl.float16)
    if TRANS and not PERM:
        tl.store(
            Y + rows[:, None] + (block * 128 + kk[None, :]) * M, y, rows[:, None] < M
        )
    else:
        tl.store(
            Y + rows[:, None] * K + block * 128 + kk[None, :], y, rows[:, None] < M
        )


@triton.jit
def _quant(
    X,
    Q,
    S,
    Z,
    SUM,
    M: tl.constexpr,
    K: tl.constexpr,
    CLIP: tl.constexpr,
    B: tl.constexpr,
    UNPACKED: tl.constexpr = False,
):
    row = tl.program_id(0)
    j = tl.arange(0, B)
    x = tl.load(X + row * K + j, j < K, 0).to(tl.float32)
    lo = (
        (tl.min(tl.where(j < K, x, float("inf")), 0) * CLIP)
        .to(tl.float16)
        .to(tl.float32)
    )
    hi = (
        (tl.max(tl.where(j < K, x, -float("inf")), 0) * CLIP)
        .to(tl.float16)
        .to(tl.float32)
    )
    span = (hi - lo).to(tl.float16).to(tl.float32)
    scale = (
        tl.minimum(tl.maximum((span / 15).to(tl.float16).to(tl.float32), 1.0e-5), 1.0e4)
        .to(tl.float16)
        .to(tl.float32)
    )
    zp = tl.extra.cuda.libdevice.nearbyint(
        tl.minimum(
            tl.maximum(tl.div_rn(-lo, scale).to(tl.float16).to(tl.float32), -1.0e4),
            1.0e4,
        )
    )
    q = (
        tl.extra.cuda.libdevice.nearbyint(tl.div_rn(x, scale))
        .to(tl.float16)
        .to(tl.float32)
    )
    q = tl.minimum(tl.maximum((q + zp).to(tl.float16).to(tl.float32), 0), 15).to(
        tl.int32
    )
    q = tl.where(j < K, q, 0)
    if UNPACKED:
        # s8 codes in [0,15], one byte per element (Hopper INT8 wgmma operand).
        tl.store(Q + row * K + j, q.to(tl.int8), j < K)
    else:
        # Interleave adjacent low/high nibbles.
        pairs = tl.reshape(q, (B // 2, 2))
        packed = tl.sum(
            pairs * tl.full((1, 2), 1, tl.int32) * (1 << (tl.arange(0, 2)[None, :] * 4)),
            1,
        )
        tl.store(
            Q + row * (K // 2) + tl.arange(0, B // 2),
            packed.to(tl.uint8),
            tl.arange(0, B // 2) < K // 2,
        )
    tl.store(S + row, scale)
    tl.store(Z + row, zp)
    tl.store(SUM + row, tl.sum(q, 0))


@triton.jit
def _epilogue(
    C,
    AS,
    AZ,
    AX,
    WS,
    WZ,
    WX,
    BIAS,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    row = i // N
    col = i % N
    c = tl.load(C + i, i < M * N, 0).to(tl.float32)
    az = tl.load(AZ + row, row < M, 0).to(tl.float32)
    wz = tl.load(WZ + col).to(tl.float32)
    ax = tl.load(AX + row, row < M, 0).to(tl.float32)
    wx = tl.load(WX + col).to(tl.float32)
    y = (
        (c - az * wx - wz * ax + K * az * wz)
        * tl.load(AS + row, row < M, 0).to(tl.float32)
        * tl.load(WS + col).to(tl.float32)
    )
    if HAS_BIAS:
        y += tl.load(BIAS + col).to(tl.float32)
    tl.store(Y + i, y, i < M * N)


@triton.jit
def _gemv(
    A,
    AS,
    AZ,
    W,
    WS,
    WZ,
    WX,
    AX,
    BIAS,
    Y,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BN, BK), 0, tl.int32)
    for start in range(tl.cdiv(K // 2, BK)):
        idx = start * BK + k
        a = tl.load(A + idx, idx < K // 2, 0).to(tl.int32)
        w = tl.load(
            W + n[:, None] * (K // 2) + idx[None, :],
            (n[:, None] < N) & (idx[None, :] < K // 2),
            0,
        ).to(tl.int32)
        acc += (w & 15) * (a[None, :] & 15) + (w >> 4) * (a[None, :] >> 4)
    c = tl.sum(acc, 1).to(tl.float32)
    az = tl.load(AZ).to(tl.float32)
    ax = tl.load(AX).to(tl.float32)
    wz = tl.load(WZ + n, n < N, 0).to(tl.float32)
    wx = tl.load(WX + n, n < N, 0).to(tl.float32)
    out = (
        (c - az * wx - wz * ax + K * az * wz)
        * tl.load(AS).to(tl.float32)
        * tl.load(WS + n, n < N, 0).to(tl.float32)
    )
    if HAS_BIAS:
        out += tl.load(BIAS + n, n < N, 0).to(tl.float32)
    tl.store(Y + n, out, n < N)


@torch.no_grad()
def affine_quant(x, clip):
    low, high = clip if isinstance(clip, tuple) else (clip, clip)
    lo = x.amin(-1, keepdim=True) * low
    hi = x.amax(-1, keepdim=True) * high
    scale = ((hi - lo) / 15).clamp(1.0e-5, 1.0e4)
    zp = (-lo / scale).clamp(-1.0e4, 1.0e4).round()
    q = ((x.float() / scale).round().half() + zp).clamp(0, 15)
    return q.to(torch.uint8), scale.squeeze(-1), zp.squeeze(-1)


def pack(q):
    return (q[..., ::2] | (q[..., 1::2] << 4)).contiguous()


def transform(x, r1, perm, r2):
    shape = x.shape
    z = (x.reshape(-1, 128) @ r1).reshape(-1, shape[-1])
    return (z[:, perm].reshape(-1, 128) @ r2).reshape(shape)


class DuQuantLinearCUDA(torch.nn.Module):
    """Standalone layer: ungrouped asymmetric W4A4, two 128x128 rotations.

    Input: contiguous FP16 CUDA [..., K]; output: FP16 [..., N].
    Smooth/LET transforms must be folded by the caller before construction.
    """

    @torch.no_grad()
    def __init__(self, weight, r1, perm, r2, bias=None, act_clip=0.9, weight_clip=0.8,
                 quantized_weight=None):
        super().__init__()
        if weight.dtype != torch.float16 or not weight.is_cuda:
            raise ValueError("weight must be FP16 CUDA")
        self.n, self.k = weight.shape
        if self.k % 128 or self.n % 32:
            raise ValueError("K must be divisible by 128 and N by 32")
        for r in (r1, r2):
            if (
                r.shape != (128, 128)
                or r.dtype != weight.dtype
                or r.device != weight.device
            ):
                raise ValueError("rotation must be FP16 [128,128] on weight device")
        if (
            perm.shape != (self.k,)
            or perm.dtype not in (torch.int32, torch.int64)
            or perm.device != weight.device
        ):
            raise ValueError("permutation must be integer [K] on weight device")
        if not torch.equal(
            perm.sort().values, torch.arange(self.k, device=weight.device)
        ):
            raise ValueError("invalid permutation")
        if not (0 < act_clip <= 1):
            raise ValueError("clipping ratios must be in (0,1]")
        for bound in weight_clip if isinstance(weight_clip, tuple) else (weight_clip,):
            if isinstance(bound, torch.Tensor):
                if (
                    bound.shape != (self.n, 1)
                    or bound.dtype != weight.dtype
                    or bound.device != weight.device
                ):
                    raise ValueError(
                        "Learned clipping bounds must be FP16 [N,1] on the weight device"
                    )
                valid = bool(((bound > 0) & (bound <= 1)).all())
            else:
                valid = 0 < bound <= 1
            if not valid:
                raise ValueError("clipping ratios must be in (0,1]")
        if bias is not None and (
            bias.shape != (self.n,)
            or bias.device != weight.device
            or bias.dtype != weight.dtype
        ):
            raise ValueError("bias must be FP16 [N] on weight device")
        for name, t in [("r1", r1), ("r2", r2), ("perm", perm), ("bias", bias)]:
            self.register_buffer(name, t.contiguous() if t is not None else None)
        self.act_clip = act_clip
        if quantized_weight is None:
            qw, sw, zw = affine_quant(transform(weight, r1, perm, r2), weight_clip)
        else:
            from .ptq import validate_weight
            qw, sw, zw = validate_weight(quantized_weight, weight)
        self.register_buffer("weight", pack(qw))
        self.register_buffer("weight_scale", sw.contiguous())
        self.register_buffer("weight_zero", zw.contiguous())
        self.register_buffer("weight_sum", qw.int().sum(-1).to(torch.int32))
        self.register_buffer(
            "weight_correction", self.k * zw.to(torch.int32) - self.weight_sum
        )
        self.register_buffer(
            "epilogue_bias",
            bias.contiguous()
            if bias is not None
            else torch.zeros(self.n, device=weight.device, dtype=weight.dtype),
        )
        # Hopper has no INT4 MMA: use the s8-activation / u4-weight wgmma kernel.
        self.hopper = _arch(weight.device) == 90
        self.gemm = load_gemm_sm90() if self.hopper else load_gemm()
        self.gemm_config = None
        # Hopper decode path (see use_small): the cooperative CUDA kernel does the
        # whole layer (transform, quantization, dp4a GEMV); larger M uses the s8 wgmma GEMM.
        self.small_m = 4 if self.hopper else 0
        self.small_gemv_config = None
        self.prefetch_bytes = 16 << 20  # L2 warm-up overlapping the activation phases
        self.small_trace = None  # optional int64 [grid, 8] globaltimer trace (debug)
        # Next DuQuantLinearCUDA in execution order (set by the model); its
        # rotations and leading weight bytes are prefetched into L2 during decode.
        self.prefetch_next = None
        self.next_weight_prefetch = 8 << 20
        if self.hopper:
            self.register_buffer("perm32", self.perm.to(torch.int32).contiguous())
            # R1^T and R2^T padded to 132 halves per row (Tensor Core B operands,
            # bank spreading in the kernel).
            r1t = torch.zeros((128, 132), device=weight.device, dtype=torch.float16)
            r1t[:, :128] = self.r1.t()
            self.register_buffer("r1t", r1t, persistent=False)
            r2t = torch.zeros((128, 132), device=weight.device, dtype=torch.float16)
            r2t[:, :128] = self.r2.t()
            self.register_buffer("r2t", r2t, persistent=False)
            # Barrier counters of the cooperative small-M kernel: one slot of 4
            # per (row bucket, config, with/without GEMV) because the epoch is
            # derived from the grid size.
            self.register_buffer(
                "small_counters",
                torch.zeros((3, 3, 2, 4), device=weight.device, dtype=torch.int64),
                persistent=False,
            )
        self.fused_epilogue = True
        self.gemv_bn = 16
        self.gemv_bk = 512
        self.rotation_bm = None
        self.rotation_warps = None
        self.transposed_rotation = True

    def sm90_config(self, m):
        """Shape-only sm90 GEMM dispatch, from a GH200 sweep of the LLaMA shapes.

        Configs 0-4: data-parallel token tiles 16/32/64/128/256; 5-8: Stream-K
        16/32/64/128. Stream-K pays off when there are few output tiles and a
        long K (down projection: N=4096, K=11008).
        """
        n, k = self.n, self.k
        sk = k >= 8192 and n <= 8192
        if m <= 16:
            return 5 if sk else 0
        if m <= 64:
            return 6 if sk else (0 if n <= 4096 else 1 if m <= 32 else 2)
        if m <= 128:
            return 7 if sk else (6 if n <= 4096 else 3)
        if m <= 256:
            return 8 if sk else (2 if n <= 4096 else 4 if n <= 16384 else 3)
        if m <= 512:
            return 3 if n <= 4096 else 4
        return 4

    def use_small(self, m):
        # Cooperative decode kernel: always for M <= 2; for 3-4 rows only when
        # N is small enough that the GEMM path is not already faster.
        return m <= self.small_m and (m <= 2 or self.n <= 8192)

    def launch_config(self, m):
        # Shape-only dispatch: no activation-dependent tuning in forward.
        bm = self.rotation_bm or (16 if m <= 32 else 32 if m <= 128 else 64)
        warps = self.rotation_warps or (4 if m == 1 else 8)
        gemm = self.gemm_config
        if gemm is None and self.hopper:
            gemm = self.sm90_config(m)
        elif gemm is None:
            gemm = 2 if m >= 256 or (m >= 64 and self.n >= 8192) else 1
        return bm, warps, gemm

    def _next_prefetch_args(self):
        nxt = self.prefetch_next
        if nxt is None or not getattr(nxt, "hopper", False):
            return (0, 0, 0, 0, 0, 0)
        w_bytes = min(nxt.weight.numel(), self.next_weight_prefetch)
        return (nxt.r1t.data_ptr(), nxt.r2t.data_ptr(), nxt.perm32.data_ptr(), nxt.k,
                nxt.weight.data_ptr(), w_bytes)

    def small_linear(self, x, y=None):
        """Hopper decode path (M <= 4): one cooperative kernel for the whole layer.

        With y=None only the activation transform/quantization runs and the
        packed codes, scales, zero points and code sums are returned.
        """
        m = x.numel() // self.k
        nb = self.k // 128
        dev = x.device
        part = torch.empty((3, m, nb), device=dev, dtype=torch.float32)
        a = torch.empty((m, self.k // 2), device=dev, dtype=torch.uint8)
        s = torch.empty(m, device=dev, dtype=torch.float16)
        zero = torch.empty_like(s)
        sums = torch.empty(m, device=dev, dtype=torch.int32)
        cfg = self.small_gemv_config
        if cfg is None:
            cfg = 1
        status = self.gemm.duquant_linear_small(
            x.data_ptr(), self.r1t.data_ptr(), self.r2t.data_ptr(), self.perm32.data_ptr(),
            self.weight.data_ptr(), y.data_ptr() if y is not None else 0,
            m, self.n, self.k, float(self.act_clip),
            torch.cuda.current_stream().cuda_stream, cfg,
            self.weight_scale.data_ptr(), self.weight_zero.data_ptr(),
            self.weight_correction.data_ptr(), self.epilogue_bias.data_ptr(),
            part[0].data_ptr(), part[1].data_ptr(), part[2].data_ptr(),
            a.data_ptr(), s.data_ptr(), zero.data_ptr(), sums.data_ptr(),
            int(y is not None), self.prefetch_bytes,
            self.small_counters[(max(m, 1) - 1).bit_length(), cfg, int(y is not None)].data_ptr(),
            self.small_trace.data_ptr() if self.small_trace is not None else 0,
            *self._next_prefetch_args(),
        )
        if status:
            raise RuntimeError(f"small-M DuQuant kernel failed: {status}")
        return a, s, zero, sums

    def activation(self, x, unpacked=None):
        m = x.numel() // self.k
        if unpacked is None and self.use_small(m):
            return self.small_linear(x)
        if unpacked is None:
            unpacked = self.hopper and m > 1
        t = torch.empty_like(x)
        z = torch.empty_like(x)
        bm, warps, _ = self.launch_config(m)
        grid = (triton.cdiv(m, bm), self.k // 128)
        _rotate[grid](
            x,
            self.r1,
            self.perm,
            t,
            m,
            self.k,
            False,
            bm,
            self.transposed_rotation and m > 1,
            num_warps=warps,
        )
        _rotate[grid](
            t,
            self.r2,
            self.perm,
            z,
            m,
            self.k,
            True,
            bm,
            self.transposed_rotation and m > 1,
            num_warps=warps,
        )
        a = torch.empty(
            (m, self.k) if unpacked else (m, self.k // 2),
            device=x.device,
            dtype=torch.int8 if unpacked else torch.uint8,
        )
        s = torch.empty(m, device=x.device, dtype=torch.float16)
        zero = torch.empty_like(s)
        sums = torch.empty(m, device=x.device, dtype=torch.int32)
        _quant[(m,)](
            z,
            a,
            s,
            zero,
            sums,
            m,
            self.k,
            self.act_clip,
            triton.next_power_of_2(self.k),
            unpacked,
        )
        return a, s, zero, sums

    def forward(self, x):
        if (
            x.dtype != torch.float16
            or x.device != self.weight.device
            or not x.is_contiguous()
            or x.shape[-1] != self.k
        ):
            raise ValueError("input must be contiguous FP16 [...,K] on weight device")
        m = x.numel() // self.k
        if m == 0:
            return torch.empty((*x.shape[:-1], self.n), device=x.device, dtype=x.dtype)
        _, _, gemm_config = self.launch_config(m)
        with torch.cuda.device(x.device):
            if self.use_small(m):
                y = torch.empty(
                    (*x.shape[:-1], self.n), device=x.device, dtype=torch.float16
                )
                self.small_linear(x, y)
                return y
            a, s, z, sums = self.activation(x)
            y = torch.empty(
                (*x.shape[:-1], self.n), device=x.device, dtype=torch.float16
            )
            if m == 1:
                _gemv[(triton.cdiv(self.n, self.gemv_bn),)](
                    a,
                    s,
                    z,
                    self.weight,
                    self.weight_scale,
                    self.weight_zero,
                    self.weight_sum,
                    sums,
                    self.bias,
                    y,
                    self.n,
                    self.k,
                    self.bias is not None,
                    self.gemv_bn,
                    self.gemv_bk,
                )
            elif self.fused_epilogue:
                fused = (
                    self.gemm.duquant_sm90_gemm_fused
                    if self.hopper
                    else self.gemm.duquant_gemm_fused
                )
                status = fused(
                    a.data_ptr(),
                    self.weight.data_ptr(),
                    y.data_ptr(),
                    m,
                    self.n,
                    self.k,
                    torch.cuda.current_stream().cuda_stream,
                    gemm_config,
                    s.data_ptr(),
                    z.data_ptr(),
                    sums.data_ptr(),
                    self.weight_scale.data_ptr(),
                    self.weight_zero.data_ptr(),
                    self.weight_correction.data_ptr(),
                    self.epilogue_bias.data_ptr(),
                )
                if status:
                    raise RuntimeError(f"CUTLASS fused GEMM failed: {status}")
            else:
                c = torch.empty((m, self.n), device=x.device, dtype=torch.int32)
                plain = (
                    self.gemm.duquant_sm90_gemm if self.hopper else self.gemm.duquant_gemm
                )
                status = plain(
                    a.data_ptr(),
                    self.weight.data_ptr(),
                    c.data_ptr(),
                    m,
                    self.n,
                    self.k,
                    torch.cuda.current_stream().cuda_stream,
                    gemm_config,
                )
                if status:
                    raise RuntimeError(f"CUTLASS GEMM failed: {status}")
                _epilogue[(triton.cdiv(m * self.n, 256),)](
                    c,
                    s,
                    z,
                    sums,
                    self.weight_scale,
                    self.weight_zero,
                    self.weight_sum,
                    self.bias,
                    y,
                    m,
                    self.n,
                    self.k,
                    self.bias is not None,
                    256,
                )
            return y
