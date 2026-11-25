"""Exportable train-only DuQuant rotation calibration and folded smoothing.

Uses the repository's UniformAffineQuantizer search unchanged. The inference
attention remains this backend's Flash SDPA implementation, not main.py's eager
attention. No LWC/LET optimization is performed.
"""

import torch


@torch.no_grad()
def apply_smoothing(weights, scales):
    """Fold the four positive scales as in smooth_and_let_inplace (zero shifts)."""
    q, u, o, d = (scales[k] for k in ("qkv", "gate_up", "o", "down"))
    weights["input_layernorm.weight"].div_(q)
    for name in ("q", "k", "v"):
        weights[f"self_attn.{name}_proj.weight"].mul_(q[None])
    weights["post_attention_layernorm.weight"].div_(u)
    for name in ("gate", "up"):
        weights[f"mlp.{name}_proj.weight"].mul_(u[None])
    weights["mlp.up_proj.weight"].div_(d[:, None])
    weights["mlp.down_proj.weight"].mul_(d[None])
    kv_h = weights["self_attn.v_proj.weight"].shape[0]
    groups = o.numel() // kv_h
    grouped_o = o.reshape(kv_h // 128, groups, 128)
    if not torch.equal(grouped_o, grouped_o[:, :1].expand_as(grouped_o)):
        raise ValueError("GQA output scales must be tied across repeated V heads")
    weights["self_attn.v_proj.weight"].div_(grouped_o[:, 0].reshape(-1, 1))
    weights["self_attn.o_proj.weight"].mul_(o[None])


def smoothing_scales(weights, maxima, alpha):
    sources = {
        "qkv": "self_attn.q_proj.weight",
        "gate_up": "mlp.up_proj.weight",
        "o": "self_attn.o_proj.weight",
        "down": "mlp.down_proj.weight",
    }
    result = {}
    for name, key in sources.items():
        act = maxima[name].half().clamp(min=1.0e-5)
        w = weights[key].abs().amax(0).clamp(min=1.0e-5)
        if name == "o":
            kv_heads = weights["self_attn.v_proj.weight"].shape[0] // 128
            groups = act.numel() // (kv_heads * 128)
            if groups > 1:
                # One V head supplies several Q heads. Their output scales
                # must agree so scaling can be folded into that single V.
                act = act.reshape(kv_heads, groups, 128).amax(1, keepdim=True).expand(-1, groups, -1).reshape(-1)
                w = w.reshape(kv_heads, groups, 128).amax(1, keepdim=True).expand(-1, groups, -1).reshape(-1)
        result[name] = (act.pow(alpha) / w.pow(1 - alpha)).clamp(min=0.01)
        if not torch.isfinite(result[name]).all():
            raise ValueError("Nonfinite smoothing scale")
    return result


def install_rotation_base(n=128):
    """Construct only the required Rot.pkl entry, using get_rot.py's Givens order."""
    import utils

    r = torch.eye(n, device="cuda")
    for i in range(n - 1):
        step = torch.eye(n, device="cuda")
        c, s = (i + 2) ** -0.5, ((i + 1) / (i + 2)) ** 0.5
        step[i, i] = step[i + 1, i + 1] = c
        step[i, i + 1], step[i + 1, i] = -s, s
        r = step @ r
    utils.Rot[n] = utils.exchange_row_col(r, 0, n - 1).cpu()


class CalibratingLinear(torch.nn.Module):
    def __init__(self, weight, steps):
        super().__init__()
        self.register_buffer("weight", weight)
        self.steps = steps
        self.exported_rotations = None

    def forward(self, x):
        from quantize.quantizer import UniformAffineQuantizer
        from .decoder import FakeLinear

        if self.exported_rotations is not None:
            raise RuntimeError("Rotation calibration must run only once per projection")
        q = UniformAffineQuantizer(
            n_bits=4,
            quant_method="duquant",
            block_size=128,
            max_rotation_step=self.steps,
            permutation_times=1,
            dynamic_method="per_token",
            lac=0.9,
        )
        q.online_duquant_cali(x.reshape(-1, x.shape[-1]))
        self.exported_rotations = (
            q.R[0].contiguous(),
            q.permutation_list[0].to(x.device),
            q.R[1].contiguous(),
        )
        fake = FakeLinear(self.weight, self.exported_rotations)
        return fake(x)
