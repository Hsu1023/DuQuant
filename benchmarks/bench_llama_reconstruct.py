"""Use the existing generation benchmark with the reconstruction RTN export."""
import hashlib
import json
from pathlib import Path
import sys

from llama_reconstruct import BASE, load_trial
import bench_model


def load_export(cls, path, mode="cuda", **kwargs):
    assert mode == "cuda" and kwargs.get("kv_bits") == 4
    model = load_trial("cuda", "rtn")
    raw = (BASE / "rtn/manifest.json").read_bytes()
    model.calibration_info = dict(json.loads(raw), manifest_sha256=hashlib.sha256(raw).hexdigest())
    return model


if __name__ == "__main__":
    bench_model.LlamaInferenceModel.from_pretrained = classmethod(load_export)
    bench_model.main()
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    result = json.loads(output.read_text())
    result["quantization"] = "DuQuant with learned folded scales/clipping, RTN W4 symmetric, A4 asymmetric, Q16 and packed asymmetric KV4"
    result["source_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), Path("benchmarks/bench_model.py"),
                  Path("benchmarks/llama_reconstruct.py"),
                  Path("benchmarks/duquant_reconstruct.py"), *Path("duquant_cuda").glob("*.py")]
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
