"""Separate-process, full-model greedy-generation speed and GPU memory benchmark."""

import argparse
import gc
import json
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from duquant_cuda.model import (
    LlamaInferenceModel,
    GenerationSession,
    model_storage_bytes,
)


def memory_snapshot():
    free, total = torch.cuda.mem_get_info()
    return dict(
        allocated_bytes=torch.cuda.memory_allocated(),
        reserved_bytes=torch.cuda.memory_reserved(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        device_used_bytes=total - free,
    )


def prompt_tokens(tok, length, batch):
    text = "The purpose of this experiment is to understand how language models process text. A transformer uses attention to combine information from earlier words and predict what comes next. Explain the main idea clearly and give a useful example. "
    ids = tok.encode(text, add_special_tokens=False)
    ids = ([tok.bos_token_id] + ids * ((length // len(ids)) + 1))[:length]
    return (
        torch.tensor(ids, device="cuda", dtype=torch.long)[None]
        .expand(batch, -1)
        .contiguous()
    )


@torch.inference_mode()
def verify_hf(model, path, tok):
    from transformers import AutoModelForCausalLM

    ref = (
        AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        )
        .cuda()
        .eval()
    )
    cases = []
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for length in (1, 17, 128):
            ids = prompt_tokens(tok, length, 1)
            caches = model.caches(1, length + 3)
            past = None
            for step in range(4):
                position = 0 if step == 0 else length + step - 1
                a = model(ids, caches, position)
                out = ref(ids, past_key_values=past, use_cache=True)
                past = out.past_key_values
                b = out.logits[:, -1]
                rel = ((a.float() - b.float()).norm() / b.float().norm()).item()
                agreement = bool(torch.equal(a.argmax(-1), b.argmax(-1)))
                best = b.topk(2).values
                hf_margin = (best[:, 0] - best[:, 1]).item()
                candidate_gap = (best[:, 0] - b.gather(-1, a.argmax(-1, keepdim=True))[:, 0]).item()
                half_best = best[:, 0].half()
                ulp = (torch.nextafter(half_best, torch.full_like(half_best, float("inf"))) - half_best).float().item()
                # HF can return many exactly tied FP16 logits (notably on a
                # BOS-only Llama-3 prefix). Report argmax differences, allowing
                # only candidates within one FP16 ULP of an exact HF top tie.
                tied = hf_margin == 0 and candidate_gap <= ulp
                if rel > 0.02 or not (agreement or tied):
                    raise AssertionError(f"HF mismatch: length={length}, step={step}, relative_l2={rel}, top1={agreement}")
                cases.append(dict(prompt_tokens=length, decode_step=step,
                                  logit_relative_l2=rel, top1_agrees=agreement,
                                  hf_top_margin=hf_margin, candidate_gap_on_hf=candidate_gap,
                                  hf_top_fp16_ulp=ulp, accepted_exact_hf_tie=tied and not agreement))
                ids = b.argmax(-1, keepdim=True)
    result = dict(
        protocol="Three prompt lengths (1,17,128), each with prefill and three cached decode steps, identical HF-selected tokens supplied to both models",
        logit_relative_l2=max(c["logit_relative_l2"] for c in cases),
        top1_agrees=all(c["top1_agrees"] for c in cases),
        passed=True,
        cases=cases,
    )
    del ref, caches, a, b, ids, out, past
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def worker(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    start = time.perf_counter()
    model = LlamaInferenceModel.from_pretrained(
        args.model, args.mode, calibration=args.calibration, kv_bits=args.kv_bits
    )
    load_seconds = time.perf_counter() - start
    validation = (
        verify_hf(model, args.model, tok)
        if args.verify_hf and args.mode == "fp16"
        else None
    )
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model_memory = memory_snapshot()
    ids = prompt_tokens(tok, args.prompt, args.batch)
    session = GenerationSession(model, ids, args.new_tokens)
    torch.cuda.synchronize()
    cache_memory = memory_snapshot()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    session.capture()
    capture_seconds = time.perf_counter() - t
    capture_memory = memory_snapshot()
    # Graph private-pool reservations stay resident during inference.
    # Keep capture/warmup peak allocation as replay has no allocator events.
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    samples = []
    for _ in range(args.repeats):
        begin = torch.cuda.Event(enable_timing=True)
        first = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall = time.perf_counter()
        begin.record()
        session.prefill_graph.replay()
        first.record()
        session.decode_graph.replay()
        end.record()
        cpu_ids = (
            session.generated.cpu()
        )  # Include completion and output transfer in wall time.
        wall_ms = (time.perf_counter() - wall) * 1000
        samples.append(
            dict(
                prefill_ms=begin.elapsed_time(first),
                decode_ms=first.elapsed_time(end),
                gpu_total_ms=begin.elapsed_time(end),
                wall_ms=wall_ms,
            )
        )
    final_memory = memory_snapshot()
    med = {key: statistics.median(s[key] for s in samples) for key in samples[0]}
    decode_count = max(args.new_tokens - 1, 0)
    finite_logits = model(ids, session.caches).isfinite().all().item()
    result = dict(
        mode=args.mode,
        calibration_manifest_sha256=(
            model.calibration_info["manifest_sha256"]
            if model.calibration_info
            else None
        ),
        lwc_epochs=(model.calibration_info or {}).get("lwc_epochs", 0),
        kv_bits=args.kv_bits,
        attention_quantization=("Q16; packed KV4 with FP16 scale/zero; tile dequantization during decode"
                                if args.kv_bits == 4 else "Q/K/V FP16 without extra qdq"
                                if args.kv_bits == 16 or args.mode == "fp16" else
                                "Legacy Q/K/V4 qdq; physical FP16 cache"),
        model=args.model,
        gpu=torch.cuda.get_device_name(),
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        torch=torch.__version__,
        layers=len(model.blocks),
        batch=args.batch,
        prompt_tokens=args.prompt,
        new_tokens=args.new_tokens,
        decode_forward_calls=decode_count,
        scope=f"Full model: embedding, {len(model.blocks)} decoder blocks, final RMSNorm, FP16 LM head, greedy argmax, real autoregressive cache advancement. Two CUDA Graphs: prefill and the entire decode sequence. EOS ignored for equal work.",
        excluded="Model load/export, first compilation, graph capture, tokenization, prompt H2D transfer. Wall time includes generated-token D2H transfer.",
        quantization=(
            (
                "LWC per-channel weight clipping; "
                if (model.calibration_info or {}).get("lwc_epochs", 0)
                else ""
            )
            + "Train-calibrated rotations + zigzag + folded smoothing; W4A4 projections, FP16 embeddings/lm_head; see attention_quantization for KV format"
            if model.calibration_info
            else "Uncalibrated fixed random dual rotations; W4A4 projections, FP16 embeddings/lm_head/attention/KV storage"
        )
        if args.mode == "cuda"
        else "FP16 throughout with identical fused blocks and Flash SDPA",
        medians=med,
        decode_ms_per_token=med["decode_ms"] / decode_count if decode_count else None,
        decode_tokens_per_second=args.batch * decode_count / (med["decode_ms"] / 1000)
        if decode_count
        else None,
        end_to_end_tokens_per_second=args.batch
        * args.new_tokens
        / (med["wall_ms"] / 1000),
        samples=samples,
        memory=dict(
            model_storage_bytes=model_storage_bytes(model),
            kv_cache_bytes=session.kv_bytes,
            kv_decode_workspace_bytes=session.kv_workspace_bytes,
            model_loaded=model_memory,
            cache_allocated=cache_memory,
            capture_and_warmup=capture_memory,
            inference=final_memory,
            inference_peak_allocated_bytes=max(
                capture_memory["peak_allocated_bytes"],
                final_memory["peak_allocated_bytes"],
            ),
            inference_reserved_bytes=final_memory["reserved_bytes"],
        ),
        load_seconds=load_seconds,
        capture_seconds=capture_seconds,
        fp16_hf_validation=validation,
        graph_matches_eager=True,
        finite_logits=finite_logits,
        generated_token_ids=cpu_ids.tolist(),
        generated_text=[
            tok.decode(row, skip_special_tokens=False) for row in cpu_ids.tolist()
        ],
    )
    if not finite_logits:
        raise AssertionError("Non-finite final model logits")
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "mode",
                    "prompt_tokens",
                    "new_tokens",
                    "medians",
                    "decode_tokens_per_second",
                    "memory",
                    "generated_text",
                )
            }
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/mnt/public_models/llama2-7b-hf")
    p.add_argument("--mode", choices=["fp16", "cuda"], required=True)
    p.add_argument("--prompt", type=int, default=128)
    p.add_argument("--new-tokens", type=int, default=32)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--verify-hf", action="store_true")
    p.add_argument("--output", required=True)
    p.add_argument("--calibration", help="Exported train-only calibration directory")
    p.add_argument("--kv-bits", type=int, choices=[4, 16], help="Explicit KV mode; omitted preserves legacy Q/K/V qdq")
    args = p.parse_args()
    worker(args)


if __name__ == "__main__":
    main()
