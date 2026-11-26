"""GEMM tile sweep over the full-model LLaMA-2-7B linear shapes (complete layer forward).

Times DuQuantLinearCUDA(x) (activation transform + quantization + GEMM with fused
epilogue) for every GEMM config of the current device and records the shape-only
default chosen by launch_config. Used to derive the per-architecture dispatch
tables in duquant_cuda/backend.py. Run from the repository root.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import triton

from duquant_cuda.backend import DuQuantLinearCUDA

# qkv (fused), o, gate_up (fused), down
SHAPES = "4096,12288;4096,4096;4096,22016;11008,4096"


def timed(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench_cudagraph(fn, rep=50) * 1000


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default=SHAPES)
    p.add_argument("--tokens", default="3,5,8,16,32,64,128,256,512,1024,2048,4096")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.manual_seed(0)
    rows = []
    for pair in args.shapes.split(";"):
        k, n = map(int, pair.split(","))
        w = torch.randn(n, k, device="cuda", dtype=torch.float16) / k**0.5
        r = torch.linalg.qr(torch.randn(2, 128, 128, device="cuda"))[0].half()
        layer = DuQuantLinearCUDA(w, r[0], torch.randperm(k, device="cuda"), r[1])
        for m in map(int, args.tokens.split(",")):
            if layer.use_small(m):
                continue
            x = torch.randn(m, k, device="cuda", dtype=torch.float16)
            layer.gemm_config = None
            default = layer.launch_config(m)[2]
            ref = layer(x).float()
            times = {}
            for cfg in range(layer.num_gemm_configs):
                layer.gemm_config = cfg
                out = layer(x).float()
                assert torch.equal(out, ref) or (out - ref).norm() / ref.norm() < 1e-3, (k, n, m, cfg)
                times[cfg] = round(timed(lambda: layer(x)), 2)
            layer.gemm_config = None
            row = dict(
                K=k, N=n, M=m,
                fp16=round(timed(lambda: torch.nn.functional.linear(x, w)), 2),
                act=round(timed(lambda: layer.activation(x)), 2),
                best=min(times, key=times.get), default=default, times=times,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(dict(
        device=torch.cuda.get_device_name(), torch=torch.__version__,
        triton=triton.__version__, rows=rows), indent=1) + "\n")


if __name__ == "__main__":
    main()
