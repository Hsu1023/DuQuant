"""Packed-attention oracle checks and teacher-forced cached-decode diagnostics.

W4 rounding can amplify tiny floating-point attention differences. Report
end-to-end differences explicitly; validate the kernel on identical Q/K/V
inputs, and score actual held-out continuations instead of claiming bitwise
equivalence between different attention reduction implementations.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from datasets import Dataset
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoTokenizer
from duquant_cuda.model import LlamaInferenceModel
from bench_model import prompt_tokens
from eval_ppl import DEFAULT_ARROW
from duquant_cuda.kv4 import PackedKVCache


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--kv-bits", required=True, type=int, choices=[4, 16])
    p.add_argument("--output", required=True)
    args = p.parse_args()
    fast = LlamaInferenceModel.from_pretrained(args.model, "cuda", calibration=args.calibration, kv_bits=args.kv_bits)
    ref = LlamaInferenceModel.from_pretrained(args.model, "integer_reference", calibration=args.calibration, kv_bits=args.kv_bits)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    attention_errors = []
    original_decode = PackedKVCache.decode

    def checked_decode(cache, q, length):
        actual = original_decode(cache, q, length)
        k, v = cache.unpack(length)  # Diagnostic only, outside speed benchmarks.
        with sdpa_kernel(SDPBackend.MATH):
            expected = F.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                enable_gqa=q.shape[1]!=k.shape[1]).half()
        delta = (actual.float()-expected.float()).abs()
        attention_errors.append(torch.stack((
            delta.norm()/expected.float().norm().clamp_min(1.e-12),
            delta.max(), (delta/(0.001+0.002*expected.float().abs())).max())))
        return actual

    PackedKVCache.decode = checked_decode
    cases = []
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for length in (1, 129, 1024):
            a, b = fast.caches(1, length+7), ref.caches(1, length+7)
            ids = prompt_tokens(tok, length, 1)
            for step in range(8):
                position = 0 if step == 0 else length+step-1
                x, y = fast(ids, a, position), ref(ids, b, position)
                rel = ((x.float()-y.float()).norm()/y.float().norm()).item()
                cases.append(dict(prompt=length, step=step, relative_l2=rel,
                                  max_abs=(x-y).abs().max().item(),
                                  top1_agrees=bool(torch.equal(x.argmax(-1), y.argmax(-1))),
                                  bitwise_equal=bool(torch.equal(x, y))))
                ids = y.argmax(-1, keepdim=True)
            del a, b
    # Eight evenly spaced test windows. Each uses 128 prompt tokens followed
    # by 32 ground-truth tokens, with actual cache writes on every step.
    slow_tok = AutoTokenizer.from_pretrained(args.model, use_fast=False, local_files_only=True)
    data = Dataset.from_file(DEFAULT_ARROW)
    tokens = slow_tok("\n\n".join(data["text"]), return_tensors="pt").input_ids.cuda()
    block_indices = torch.linspace(0, tokens.numel()//2048-1, 8).long().tolist()
    losses_a, losses_b, preds_a, preds_b, targets = [], [], [], [], []
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for index in block_indices:
            start = index*2048
            a, b = fast.caches(1, 159), ref.caches(1, 159)
            ids = tokens[:, start:start+128]
            for step in range(32):
                position = 0 if step==0 else 128+step-1
                x, y = fast(ids, a, position), ref(ids, b, position)
                target = tokens[:, start+128+step]
                losses_a.append(F.cross_entropy(x.float(), target))
                losses_b.append(F.cross_entropy(y.float(), target))
                preds_a.append(x.argmax(-1)); preds_b.append(y.argmax(-1)); targets.append(target)
                ids = target[:, None]
            del a, b
    la, lb = torch.stack(losses_a).double(), torch.stack(losses_b).double()
    pa, pb, target = torch.cat(preds_a), torch.cat(preds_b), torch.cat(targets)
    cached = dict(scored_tokens=256, block_indices=block_indices, prompt_tokens=128,
                  continuation_tokens=32, dataset="WikiText-2 raw test",
                  cuda_ppl=la.mean().exp().item(), reference_ppl=lb.mean().exp().item(),
                  mean_nll_delta=(la-lb).mean().item(),
                  prediction_agreement=(pa==pb).double().mean().item(),
                  cuda_accuracy=(pa==target).double().mean().item(),
                  reference_accuracy=(pb==target).double().mean().item(),
                  cuda_token_nll=la.tolist(), reference_token_nll=lb.tolist(),
                  note="Small cached-decode diagnostic, not the full-test PPL protocol or an equivalence guarantee")
    PackedKVCache.decode = original_decode
    maximum = torch.stack(attention_errors).amax(0).tolist() if attention_errors else [0.,0.,0.]
    attention = dict(calls=len(attention_errors), max_relative_l2=maximum[0],
                     max_abs=maximum[1], max_tolerance_ratio=maximum[2],
                     tolerance="abs(error) <= .001 + .002*abs(FP32 dense oracle rounded to FP16)",
                     passed=maximum[2]<=1)
    result = dict(model=args.model, kv_bits=args.kv_bits, cases=cases,
                  status="complete" if attention["passed"] else "failed",
                  packed_attention_oracle=attention, cached_decode=cached,
                  max_relative_l2=max(c["relative_l2"] for c in cases),
                  top1_agreement=sum(c["top1_agrees"] for c in cases)/len(cases),
                  reference="Exact integer-code linear algebra; PyTorch norms/activations; FP32 dense SDPA for KV4 single-token decode, Flash SDPA for prefill/KV16",
                  protocol="Identical reference-selected tokens supplied to both models; 3 prompt lengths x 8 steps; includes split-boundary and long-cache cases")
    Path(args.output).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)
    if not attention["passed"]:
        raise AssertionError("Packed attention differs from shared-input dense oracle")
    if args.kv_bits==16 and (result["max_relative_l2"]!=0 or cached["mean_nll_delta"]!=0):
        raise AssertionError("KV16 should match the independent reference exactly")


if __name__ == "__main__":
    main()
