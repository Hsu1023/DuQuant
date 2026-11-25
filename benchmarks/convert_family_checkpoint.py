"""Lossless bin-to-safetensors conversion; optionally repair legacy LLaMA-1 metadata."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path)
    p.add_argument("--legacy-llama1", action="store_true")
    args = p.parse_args()
    root = args.model
    index = json.loads((root / "pytorch_model.bin.index.json").read_text())
    report = dict(format="bin to safetensors; tensor values and dtypes unchanged", shards=[])
    mapping = {}
    for source in sorted(set(index["weight_map"].values())):
        target = source.replace("pytorch_model-", "model-").replace(".bin", ".safetensors")
        state = torch.load(root / source, map_location="cpu", weights_only=True, mmap=True)
        if not (root / target).exists():
            tmp = root / (target + ".tmp")
            save_file(state, tmp, metadata={"format": "pt"})
            tmp.replace(root / target)
        with safe_open(root / target, framework="pt") as f:
            assert set(f.keys()) == set(state)
            for key, value in state.items():
                actual = f.get_tensor(key)
                assert actual.dtype == value.dtype and torch.equal(actual, value), key
        report["shards"].append(dict(source=source, target=target, tensor_bitwise_equal=True,
            source_sha256=digest(root/source), target_sha256=digest(root/target)))
        mapping[source] = target
        del state
        print(f"Verified {target}", flush=True)
    index["weight_map"] = {key: mapping[value] for key, value in index["weight_map"].items()}
    (root / "model.safetensors.index.json").write_text(json.dumps(index, indent=2)+"\n")
    if args.legacy_llama1:
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=str(root / "tokenizer.model"))
        assert (sp.bos_id(), sp.eos_id(), sp.unk_id(), sp.vocab_size()) == (1, 2, 0, 32000)
        patches = {}
        for filename in ("config.json", "generation_config.json", "tokenizer_config.json", "special_tokens_map.json"):
            before = json.loads((root / filename).read_text())
            after = dict(before)
            if filename == "config.json":
                after.update(bos_token_id=1, eos_token_id=2, pad_token_id=0,
                             max_position_embeddings=before.get("max_sequence_length", 2048))
            elif filename == "generation_config.json":
                after.update(bos_token_id=1, eos_token_id=2, pad_token_id=0)
            else:
                after.update(bos_token="<s>", eos_token="</s>", unk_token="<unk>")
                if filename == "tokenizer_config.json":
                    after.update(tokenizer_class="LlamaTokenizer", add_bos_token=True,
                                 add_eos_token=False, model_max_length=2048, legacy=True)
            patches[filename] = dict(before=before, after=after)
            (root / filename).write_text(json.dumps(after, indent=2)+"\n")
        report["metadata_repair"] = dict(
            reason="Legacy conversion has empty special-token strings and incorrect BOS/EOS IDs; use IDs verified from the unchanged SentencePiece model; expose its original 2048 context limit",
            files=patches,
        )
    (root / "conversion.json").write_text(json.dumps(report, indent=2)+"\n")
    print("Conversion verified", flush=True)


if __name__ == "__main__":
    main()
