"""KV4 decode-attention tile/warp sweep (LLaMA-2-7B MHA: 32 heads, head dim 128).

Times PackedKVCache.decode (split partials + merge) at several context lengths
for each (tokens per tile, warps) pair; the cache capacity equals the prompt plus
32 generated tokens as in the end-to-end benchmark. Run from the repository root.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import triton

from duquant_cuda.kv4 import PackedKVCache


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lengths", default="128,512,1024,2048")
    p.add_argument("--heads", type=int, default=32)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rows = []
    for length in map(int, args.lengths.split(",")):
        cap = length + 32
        cache = PackedKVCache(1, args.heads, args.heads, cap, "cuda")
        cache.k.random_(0, 256)
        cache.v.random_(0, 256)
        cache.k_params[..., 0].uniform_(0.01, 0.1)
        cache.v_params[..., 0].uniform_(0.01, 0.1)
        cache.k_params[..., 1].uniform_(0, 15)
        cache.v_params[..., 1].uniform_(0, 15)
        q = torch.randn(1, args.heads, 1, 128, device="cuda", dtype=torch.float16)
        default = (cache.tile, cache.decode_warps)
        ref = cache.decode(q, length).float()
        times = {}
        for tile in (16, 32, 64, 128):
            for warps in (1, 2, 4, 8):
                cache.tile, cache.decode_warps = tile, warps
                cache.splits = triton.cdiv(cap, tile)
                cache.partial = torch.empty((1, args.heads, cache.splits, 130), device="cuda", dtype=torch.float32)
                out = cache.decode(q, length).float()
                assert (out - ref).abs().max() < 2e-2, (length, tile, warps)
                times[f"{tile}x{warps}"] = round(
                    triton.testing.do_bench_cudagraph(lambda: cache.decode(q, length), rep=50) * 1000, 2)
        row = dict(length=length, default=f"{default[0]}x{default[1]}",
                   best=min(times, key=times.get), times=times)
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(dict(
        device=torch.cuda.get_device_name(), rows=rows), indent=1) + "\n")


if __name__ == "__main__":
    main()
