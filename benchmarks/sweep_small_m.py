"""Decode-kernel sweep: cooperative linear (M <= 4) rows-per-warp config and L2 prefetch.

For each LLaMA-2-7B full-model linear shape and M in 1..4 (where the cooperative
layout fits the device), times the complete layer forward with a cold L2 (256 MiB
flush outside the timed region, as bench_single_layer --cold) for every
small_gemv_config and prefetch budget, and the FP16 linear for reference.
Run from the repository root.
"""

import argparse
import ctypes
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from bench_single_layer import timed, timed_cold
from duquant_cuda.backend import DuQuantLinearCUDA

SHAPES = "4096,12288;4096,4096;4096,22016;11008,4096"


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default=SHAPES)
    p.add_argument("--tokens", default="1,2,3,4")
    p.add_argument("--prefetch-mib", default="0,4,16")
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
            x = torch.randn(m, k, device="cuda", dtype=torch.float16)
            row = dict(K=k, N=n, M=m, fp16_cold=round(timed_cold(lambda: torch.nn.functional.linear(x, w)), 2),
                       fp16_warm=round(timed(lambda: torch.nn.functional.linear(x, w)), 2))
            if not layer.small_fits(m):
                row["small"] = None
                rows.append(row)
                print(json.dumps(row), flush=True)
                continue
            info = (ctypes.c_int * 4)()
            layer.gemm.duquant_linear_small_plan(m, n, k, 1, 1, info)
            row["plan"] = dict(own=info[0], alias_r2=info[1], grid=info[2], smem=info[3])
            small_m = layer.small_m
            layer.small_m = 0
            row["gemm_cold"] = round(timed_cold(lambda: layer(x)), 2)
            layer.small_m = small_m
            times = {}
            for cfg in range(3):
                for mib in map(int, args.prefetch_mib.split(",")):
                    layer.small_gemv_config, layer.prefetch_bytes = cfg, mib << 20
                    times[f"{cfg}/{mib}"] = round(timed_cold(lambda: layer(x)), 2)
            layer.small_gemv_config, layer.prefetch_bytes = None, 16 << 20
            row["small_cold"] = times
            row["best"] = min(times, key=times.get)
            rows.append(row)
            print(json.dumps(row), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(dict(
        device=torch.cuda.get_device_name(), torch=torch.__version__, rows=rows), indent=1) + "\n")


if __name__ == "__main__":
    main()
