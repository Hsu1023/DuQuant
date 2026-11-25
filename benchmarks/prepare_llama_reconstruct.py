"""Freeze the LLaMA-2 reconstruction training tokens (train.pt / train.json).

Rebuilds exactly the 128 x 2048 WikiText-2 train sequences recorded in the root
calibration manifest (same slow tokenizer, same sample starts) and checks the
token hash against that manifest before writing.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from datasets import Dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llama_reconstruct import BASE, MODEL, REPORT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--calibration", default="benchmarks/cache/calibrated_w4a4")
    args = p.parse_args()
    if (BASE / "train.pt").exists():
        raise FileExistsError("Existing training tokens are preserved")
    root = json.loads((Path(args.calibration) / "manifest.json").read_text())
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False, local_files_only=True)
    data = Dataset.from_file(root["source_arrow"])
    ids = tok("\n\n".join(data["text"]), return_tensors="pt").input_ids
    starts, seqlen = root["sample_starts"], root["seqlen"]
    tokens = torch.cat([ids[:, s : s + seqlen] for s in starts])
    digest = hashlib.sha256(tokens.numpy().tobytes()).hexdigest()
    if digest != root["token_ids_sha256"]:
        raise RuntimeError("Token hash differs from the calibration manifest")
    BASE.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    torch.save(tokens, BASE / "train.pt")
    train = dict(source_arrow=root["source_arrow"], split="train", nsamples=len(starts),
                 seqlen=seqlen, seed=root["seed"], sample_starts=starts, tokens_sha256=digest)
    (BASE / "train.json").write_text(json.dumps(train, indent=2) + "\n")
    print("Prepared", tuple(tokens.shape), digest, flush=True)


if __name__ == "__main__":
    main()
