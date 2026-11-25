"""WikiText-2 PPL and integer-reference check of the LLaMA-2 reconstruction exports.

Uses the protocol of benchmarks/prepare_lambada.py (slow tokenizer, '\\n\\n' join,
special tokens, every full disjoint 2048 block, shifted FP32 CE) and verifies the
token hash against the tracked protocol before scoring. The integer check runs
the first --check-blocks blocks through the CUDA backend and the independent
integer reference and requires identical per-block NLL sums.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llama_reconstruct import *  # noqa: F401,F403  (ROOT, MODEL, BASE, load_trial, torch, F)
from datasets import Dataset
from transformers import AutoTokenizer
from torch.nn.attention import SDPBackend, sdpa_kernel


def wikitext_tokens():
    protocol = json.loads((ROOT / "benchmarks/lambada_compare/llama2_protocol.json").read_text())["wikitext"]
    arrow = os.environ.get("DUQUANT_WIKITEXT_TEST", protocol["source_arrow"])
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False, local_files_only=True)
    tokens = tok("\n\n".join(Dataset.from_file(arrow)["text"]), return_tensors="pt").input_ids
    digest = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
    if digest != protocol["token_ids_sha256"]:
        raise RuntimeError("WikiText token hash differs from the tracked protocol")
    return tokens, protocol


@torch.inference_mode()
def block_nll(model, tokens, blocks):
    caches = model.caches(1, 2048)
    out = []
    for i in range(blocks):
        ids = tokens[:, i * 2048 : (i + 1) * 2048].cuda()
        logits = model(ids, caches, last_token_only=False)
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                               ids[:, 1:].reshape(-1), reduction="none")
        value = loss.double().sum().item()
        assert math.isfinite(value)
        out.append(value)
    return out


def load(mode, variant):
    if variant == "fp16":
        return LlamaInferenceModel.from_pretrained(MODEL, "fp16", kv_bits=16)
    return load_trial(mode, variant)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default="fp16,rtn,gptq")
    p.add_argument("--check-blocks", type=int, default=8)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise FileExistsError("Choose a new output file")
    torch.backends.cuda.matmul.allow_tf32 = False
    tokens, protocol = wikitext_tokens()
    record = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, protocol=protocol, results={})
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for variant in args.variants.split(","):
            model = load("cuda", variant)
            start = time.perf_counter()
            nll = block_nll(model, tokens, protocol["blocks"])
            total = math.fsum(nll)
            res = dict(wikitext_ppl=math.exp(total / protocol["scored_tokens"]), nll_sum=total,
                       blocks=protocol["blocks"], seconds=time.perf_counter() - start)
            del model
            torch.cuda.empty_cache()
            if variant != "fp16" and args.check_blocks:
                ref = load("integer_reference", variant)
                ref_nll = block_nll(ref, tokens, args.check_blocks)
                del ref
                torch.cuda.empty_cache()
                res["integer_check"] = dict(
                    blocks=args.check_blocks,
                    exact=ref_nll == nll[: args.check_blocks],
                    max_abs_block_nll_difference=max(abs(a - b) for a, b in zip(ref_nll, nll)),
                )
            record["results"][variant] = res
            print(variant, json.dumps(res), flush=True)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
