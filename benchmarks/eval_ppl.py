"""Controlled full WikiText-2 PPL comparison of FP16, fake W4A4 and real W4A4.

Defaults to seed-42 uncalibrated parameters; --calibration loads a train-only
export. Full blocks, no overlap, shifted FP32 CE, exact scored-token denominator.
"""

import os
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from duquant_cuda.model import LlamaInferenceModel

DEFAULT_ARROW = os.environ.get("DUQUANT_WIKITEXT_TEST", "/mnt/public_datasets/delta_skip/hf_cache/Salesforce___wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-test.arrow")


def score_next_tokens(logits, token_ids):
    """Position t predicts token t+1; neither BOS nor final logits are scored."""
    shifted = logits[:, :-1]
    target = token_ids[:, 1:]
    losses = F.cross_entropy(
        shifted.float().reshape(-1, shifted.shape[-1]),
        target.reshape(-1),
        reduction="none",
    )
    return losses, shifted.argmax(-1), target


def update_tensor_hash(digest, tensor):
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(str(tensor.dtype).encode())
    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())


@torch.inference_mode()
def quantization_fingerprint(model, mode):
    rotations = hashlib.sha256()
    weights = hashlib.sha256()
    if mode == "fp16":
        return None
    for index, block in enumerate(model.blocks):
        for name in ("qkv", "o", "gate_up", "down"):
            linear = getattr(block, name)
            tag = f"{index}:{name}".encode()
            rotations.update(tag)
            weights.update(tag)
            for key in ("r1", "perm", "r2"):
                update_tensor_hash(rotations, getattr(linear, key))
            if mode == "cuda":
                q = torch.stack(
                    (linear.weight & 15, linear.weight >> 4), dim=-1
                ).flatten(-2)
                w = (q.half() - linear.weight_zero[:, None]) * linear.weight_scale[
                    :, None
                ]
            elif mode == "integer_reference":
                w = (
                    linear.codes.half() - linear.zero.half()[:, None]
                ) * linear.scale.half()[:, None]
            else:
                w = linear.weight
            update_tensor_hash(weights, w)
            del w
            if mode == "cuda":
                del q
    gc.collect()
    torch.cuda.empty_cache()
    calibration = getattr(model, "calibration_info", None)
    result = dict(
        rotations_and_permutations_sha256=rotations.hexdigest(),
        dequantized_fp16_weights_sha256=weights.hexdigest(),
        weight_bits=4,
        activation_bits=4,
        weight_clip="learned_per_channel"
        if calibration and calibration.get("lwc_epochs", 0) > 0
        else 0.8,
        act_clip=0.9,
        qk_clip=0.9 if model.blocks[0].kv_bits != 16 else None,
        v_clip=1.0 if model.blocks[0].kv_bits != 16 else None,
        **({"kv_bits": model.blocks[0].kv_bits, "query_bits": 16}
           if model.blocks[0].kv_bits is not None else {}),
        seed=calibration["seed"] if calibration else 42,
        calibrated=calibration is not None,
        **(
            {"lwc_epochs": calibration["lwc_epochs"], "lwc_lr": calibration["lwc_lr"]}
            if calibration and calibration.get("lwc_epochs", 0)
            else {}
        ),
        **(
            {"calibration_manifest_sha256": calibration["manifest_sha256"]}
            if calibration
            else {}
        ),
    )
    if calibration and calibration.get("ptq"):
        result.update(
            ptq=calibration["ptq"], weight_symmetric=True, activation_symmetric=False,
            weight_clip="exported_per_channel", act_clip="exported_per_projection",
            qk_clip=None, v_clip=None,
            kv_clips=[(b.k_clip, b.v_clip) for b in model.blocks] if model.blocks[0].kv_bits == 4 else None,
            activation_clips=[{n: getattr(b, n).act_clip for n in ("qkv", "o", "gate_up", "down")}
                              for b in model.blocks],
        )
    return result


def load_tokens(args):
    from datasets import Dataset
    from transformers import AutoTokenizer

    data = Dataset.from_file(args.arrow)
    text = "\n\n".join(data["text"])
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, use_fast=False, local_files_only=True
    )
    tokens = tokenizer(text, return_tensors="pt").input_ids
    meta = dict(
        dataset="Salesforce/wikitext",
        config="wikitext-2-raw-v1",
        split="test",
        source_revision="b08601e04326c79dfdd32d625aee71d232d685c3",
        source_arrow=args.arrow,
        source_arrow_sha256=hashlib.sha256(Path(args.arrow).read_bytes()).hexdigest(),
        rows=len(data),
        join="two newlines",
        tokenizer_class=type(tokenizer).__name__,
        tokenizer_use_fast=bool(tokenizer.is_fast),
        add_special_tokens=True,
        token_count=tokens.numel(),
        token_ids_sha256=hashlib.sha256(tokens.numpy().tobytes()).hexdigest(),
    )
    return tokens, meta


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=os.environ.get("DUQUANT_LLAMA2", "/mnt/public_models/llama2-7b-hf"))
    p.add_argument(
        "--mode", choices=["fp16", "fake", "cuda", "integer_reference", "hf"], required=True
    )
    p.add_argument("--arrow", default=DEFAULT_ARROW)
    p.add_argument("--seqlen", type=int, default=2048)
    p.add_argument("--max-blocks", type=int, default=0, help="0 = every complete block")
    p.add_argument("--sample-blocks", type=int, default=4)
    p.add_argument("--output", required=True)
    p.add_argument("--calibration", help="Exported train-only calibration directory")
    p.add_argument("--kv-bits", type=int, choices=[4, 16], help="Omit for legacy Q/K/V qdq")
    args = p.parse_args()
    if args.seqlen < 2 or args.seqlen > 4096:
        p.error("seqlen must be in [2,4096]")
    if args.max_blocks < 0 or args.sample_blocks < 0:
        p.error("block counts must be nonnegative")
    tokens, data_meta = load_tokens(args)
    count = tokens.numel() // args.seqlen
    if args.max_blocks:
        count = min(count, args.max_blocks)
    if not count:
        raise ValueError("No complete evaluation blocks")
    print(
        f"{args.mode}: {count} blocks, {count * (args.seqlen - 1)} scored tokens",
        flush=True,
    )
    mode = "reference" if args.mode == "fake" else args.mode
    if mode == "hf":
        from transformers import AutoModelForCausalLM
        if args.calibration:
            p.error("HF reference uses the original FP16 checkpoint")
        if args.kv_bits == 4:
            p.error("Original HF reference does not implement KV4")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, torch_dtype=torch.float16,
            attn_implementation="sdpa",
        ).cuda().eval()
        params, caches = None, None
        vocab_size = model.config.vocab_size
    else:
        model = LlamaInferenceModel.from_pretrained(
            args.model, mode, seed=42, calibration=args.calibration, kv_bits=args.kv_bits
        )
        params = quantization_fingerprint(model, args.mode)
        caches = model.caches(1, args.seqlen)
        vocab_size = model.lm_head.shape[0]
    result = dict(
        mode=args.mode,
        model=args.model,
        dataset=data_meta,
        quantization=params,
        kv_bits=args.kv_bits,
        seqlen=args.seqlen,
        block_count=count,
        scored_tokens=count * (args.seqlen - 1),
        unused_tail_tokens=tokens.numel() % args.seqlen,
        limit_blocks=args.max_blocks,
        protocol="Disjoint full blocks; cache reset logically at position=0 for each block; predict tokens 1..seqlen-1; FP32 CE; all modes use Flash SDPA; no calibration or fitting on eval data",
        blocks=[],
        status="running",
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = output.parent / "cache"
    arrays.mkdir(exist_ok=True)
    predictions = []
    targets = []
    sample_logits = []
    sample_indices = []
    start = time.perf_counter()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for index in range(count):
            x = tokens[:, index * args.seqlen : (index + 1) * args.seqlen].cuda()
            logits = (model(x, use_cache=False).logits if mode == "hf"
                      else model(x, caches, last_token_only=False))
            losses, pred, target = score_next_tokens(logits, x)
            if not torch.isfinite(losses).all():
                raise AssertionError(f"Nonfinite losses in block {index}")
            nll = losses.double().sum().item()
            correct = (pred == target).sum().item()
            result["blocks"].append(
                dict(
                    index=index,
                    nll_sum=nll,
                    scored_tokens=target.numel(),
                    correct_next_tokens=correct,
                )
            )
            predictions.append(pred.cpu().numpy().astype(np.int32).ravel())
            targets.append(target.cpu().numpy().astype(np.int32).ravel())
            if index < args.sample_blocks:
                positions = torch.linspace(0, args.seqlen - 2, 64, device="cuda").long()
                sample_logits.append(logits[0, positions].cpu().numpy())
                sample_indices.extend(
                    (index * (args.seqlen - 1) + positions.cpu().numpy()).tolist()
                )
            if index % 10 == 0 or index == count - 1:
                total = math.fsum(b["nll_sum"] for b in result["blocks"])
                seen = sum(b["scored_tokens"] for b in result["blocks"])
                print(
                    f"{args.mode} {index + 1}/{count}: PPL={math.exp(total / seen):.6f}",
                    flush=True,
                )
                output.write_text(json.dumps(result, indent=2) + "\n")
            del logits, losses, x, target, pred
    total = math.fsum(b["nll_sum"] for b in result["blocks"])
    scored = sum(b["scored_tokens"] for b in result["blocks"])
    array_path = arrays / (output.stem + ".npz")
    np.savez_compressed(
        array_path,
        predictions=np.concatenate(predictions),
        targets=np.concatenate(targets),
        sample_logits=np.concatenate(sample_logits)
        if sample_logits
        else np.empty((0, vocab_size), dtype=np.float16),
        sample_indices=np.asarray(sample_indices, dtype=np.int64),
    )
    result.update(
        status="complete",
        nll_sum=total,
        mean_nll=total / scored,
        ppl=math.exp(total / scored),
        next_token_accuracy=sum(b["correct_next_tokens"] for b in result["blocks"])
        / scored,
        elapsed_seconds=time.perf_counter() - start,
        diagnostic_arrays=str(array_path),
        sampled_logit_rows=len(sample_indices),
    )
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "mode",
                    "ppl",
                    "mean_nll",
                    "next_token_accuracy",
                    "elapsed_seconds",
                    "quantization",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
