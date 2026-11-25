"""Reproducible complete-path microbenchmark. Run from repository root."""

import argparse
import json
import ctypes
import sys
import statistics
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import triton
from duquant_cuda.backend import DuQuantLinearCUDA, affine_quant, transform, pack


def timed(fn):
    # CUDA Graph removes Python overhead equally for FP16 and W4A4.
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench_cudagraph(fn, rep=100) * 1000


def timed_cold(fn):
    # Flush L2 before each invocation, but exclude the flush from timing.
    flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    # PyTorch 2.7 lacks Event(external=True); use the CUDA runtime flag.
    cudart = ctypes.CDLL(
        str(
            Path(torch.__file__).parent.parent
            / "nvidia/cuda_runtime/lib/libcudart.so.12"
        )
    )
    record = cudart.cudaEventRecordWithFlags
    record.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]
    record.restype = ctypes.c_int
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        flush.fill_(1)
        if record(start.cuda_event, stream.cuda_stream, 1):
            raise RuntimeError("start event capture failed")
        fn()
        if record(end.cuda_event, stream.cuda_stream, 1):
            raise RuntimeError("end event capture failed")
    times = []
    for _ in range(30):
        graph.replay()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)
    return statistics.median(times)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shapes", default="4096,4096;4096,11008;11008,4096")
    p.add_argument("--tokens", default="1,16,128,512,2048")
    p.add_argument("--output", default="benchmarks/results.json")
    p.add_argument("--tune", action="store_true")
    p.add_argument("--cold", action="store_true")
    p.add_argument(
        "--model",
        help="Optional local LLaMA-2 model: use actual first-layer projection weights; activations/rotations remain synthetic",
    )
    args = p.parse_args()
    torch.manual_seed(42)
    results = []
    print("GPU", torch.cuda.get_device_name(), flush=True)
    for pair in args.shapes.split(";"):
        k, n = map(int, pair.split(","))
        if args.model:
            from safetensors import safe_open

            names = {
                (4096, 4096): "self_attn.q_proj",
                (4096, 11008): "mlp.up_proj",
                (11008, 4096): "mlp.down_proj",
            }
            key = "model.layers.0." + names[k, n] + ".weight"
            index = json.loads(
                (Path(args.model) / "model.safetensors.index.json").read_text()
            )
            with safe_open(
                str(Path(args.model) / index["weight_map"][key]),
                framework="pt",
                device="cpu",
            ) as f:
                w = f.get_tensor(key).to(device="cuda", dtype=torch.float16)
        else:
            w = torch.randn(n, k, device="cuda", dtype=torch.float16) / (k**0.5)
        r = torch.linalg.qr(torch.randn(2, 128, 128, device="cuda"))[0].half()
        perm = torch.randperm(k, device="cuda")
        layer = DuQuantLinearCUDA(w, r[0], perm, r[1])
        qw, sw, zw = affine_quant(transform(w, *[r[0], perm, r[1]]), 0.8)
        wd = (qw.half() - zw[:, None]) * sw[:, None]
        for m in map(int, args.tokens.split(",")):
            x = torch.randn(1, m, k, device="cuda", dtype=torch.float16)
            transformed = transform(x, r[0], perm, r[1])
            qa, sa, za = affine_quant(transformed, 0.9)
            ref = torch.nn.functional.linear(
                (qa.half() - za[..., None]) * sa[..., None], wd
            )
            if args.tune:
                if m > 1:
                    best = (float("inf"), None)
                    for bm in (16, 32, 64, 128):
                        for warps in (4, 8):
                            layer.rotation_bm, layer.rotation_warps = bm, warps
                            t = timed(lambda: layer.activation(x))
                            if t < best[0]:
                                best = (t, (bm, warps))
                    layer.rotation_bm, layer.rotation_warps = best[1]
                if m == 1:
                    best = (float("inf"), None)
                    for bn in (4, 8, 16):
                        for bk in (128, 256, 512):
                            layer.gemv_bn, layer.gemv_bk = bn, bk
                            t = timed(lambda: layer(x))
                            if t < best[0]:
                                best = (t, (bn, bk))
                    layer.gemv_bn, layer.gemv_bk = best[1]
                else:
                    best = (float("inf"), None)
                    for config in range(9 if layer.hopper else 3):
                        layer.gemm_config = config
                        t = timed(lambda: layer(x))
                        if t < best[0]:
                            best = (t, config)
                    layer.gemm_config = best[1]
            out = layer(x)
            torch.cuda.synchronize()
            rel = ((out.float() - ref.float()).norm() / ref.float().norm()).item()
            maxerr = (out - ref).abs().max().item()
            a, s, z, sums = layer.activation(x)
            codes = qa.to(torch.int8) if a.dtype == torch.int8 else pack(qa)
            mismatch = (a != codes.reshape_as(a)).float().mean().item()
            # Different accumulation order in rotations may change borderline
            # quantization bins. Large errors must never be timed as successes.
            if rel > 0.002:
                raise AssertionError(f"reference mismatch {k, n, m}: {rel}")
            fp = timed(lambda: torch.nn.functional.linear(x, w))
            quant = timed(lambda: layer(x))
            act = timed(lambda: layer.activation(x))
            row = dict(
                K=k,
                N=n,
                M=m,
                fp16_us=round(fp, 3),
                duquant_us=round(quant, 3),
                activation_us=round(act, 3),
                speedup=round(fp / quant, 3),
                relative_l2=rel,
                max_abs_error=maxerr,
                packed_byte_mismatch=mismatch,
                gemm_config=layer.launch_config(m)[2],
                gemv=[layer.gemv_bn, layer.gemv_bk],
                rotation=list(layer.launch_config(m)[:2]),
                transposed_rotation=layer.transposed_rotation,
                fused_epilogue=layer.fused_epilogue,
            )
            if args.cold:
                cold_fp = timed_cold(lambda: torch.nn.functional.linear(x, w))
                cold_q = timed_cold(lambda: layer(x))
                row.update(
                    cold_fp16_us=round(cold_fp, 3),
                    cold_duquant_us=round(cold_q, 3),
                    cold_speedup=round(cold_fp / cold_q, 3),
                )
            results.append(row)
            print(json.dumps(row), flush=True)
    Path(args.output).write_text(
        json.dumps(
            dict(
                device=torch.cuda.get_device_name(),
                torch=torch.__version__,
                triton=triton.__version__,
                model=args.model,
                method="CUDA Graph single layer, random activations and fixed rotations, includes activation transform/quant/pack and GEMM; optional cold timings flush 256 MiB before each invocation outside timing",
                results=results,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
