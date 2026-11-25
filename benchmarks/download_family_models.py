"""Download public family checkpoints with verified SHA256 and resumable ranges."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import requests


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fetch(repo, name):
    root = Path("benchmarks/cache/family_models") / name
    root.mkdir(parents=True, exist_ok=True)
    response = requests.get(f"https://modelscope.cn/api/v1/models/{repo}/repo/files",
                            params={"Revision": "master", "Recursive": "true"}, timeout=30)
    response.raise_for_status()
    keep = {"config.json", "generation_config.json", "tokenizer.model",
            "tokenizer_config.json", "special_tokens_map.json", "pytorch_model.bin.index.json"}
    files = [f for f in response.json()["Data"]["Files"]
             if f["Name"] in keep or f["Name"].startswith("pytorch_model-") and f["Name"].endswith(".bin")]
    records = [{k: f[k] for k in ("Name", "Size", "Sha256")} for f in files]
    (root / "download_manifest.json").write_text(json.dumps(dict(repo=repo, files=records), indent=2)+"\n")
    print(f"Downloading {repo}: {sum(f['Size'] for f in files)/1e9:.2f} GB", flush=True)
    tasks, handles = [], {}
    chunk = 32 << 20
    for f in files:
        path = root / f["Name"]
        if path.exists() and path.stat().st_size == f["Size"] and sha256(path) == f["Sha256"]:
            continue
        partial = path.with_name(path.name + ".ranges")
        fd = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
        os.ftruncate(fd, f["Size"])
        handles[f["Name"]] = fd
        for start in range(0, f["Size"], chunk):
            tasks.append((f, start, min(start+chunk, f["Size"]), fd))
    lock = threading.Lock()
    progress = [0]
    started = time.perf_counter()

    def part(task):
        f, start, end, fd = task
        marker = root / f".{f['Name']}.{start}.done"
        if marker.exists() and marker.read_text() == f["Sha256"]:
            return
        url = f"https://modelscope.cn/models/{repo}/resolve/master/{f['Name']}"
        for attempt in range(6):
            try:
                with requests.get(url, headers={"Range": f"bytes={start}-{end-1}"},
                                  stream=True, timeout=(15, 45)) as r:
                    r.raise_for_status()
                    expected = f"bytes {start}-{end-1}/{f['Size']}"
                    if r.status_code != 206 or r.headers.get("Content-Range") != expected:
                        if not (r.status_code == 200 and start == 0 and end == f["Size"]):
                            raise ValueError("Server returned an incorrect byte range")
                    offset = start
                    for data in r.iter_content(1 << 20):
                        if offset + len(data) > end:
                            raise ValueError("Oversized range response")
                        view = memoryview(data)
                        while view:
                            written = os.pwrite(fd, view, offset)
                            offset += written
                            view = view[written:]
                    if offset != end:
                        raise ValueError("Truncated range response")
                marker.write_text(f["Sha256"])
                with lock:
                    progress[0] += end-start
                    if progress[0] // (256 << 20) != (progress[0]-(end-start)) // (256 << 20):
                        print(f"{name}: {progress[0]/1e9:.2f} GB, {progress[0]/1e6/(time.perf_counter()-started):.1f} MB/s", flush=True)
                return
            except (requests.RequestException, ValueError):
                if attempt == 5:
                    raise
                time.sleep(min(2**attempt, 15))

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for future in as_completed([pool.submit(part, t) for t in tasks]):
                future.result()
    finally:
        for fd in handles.values():
            os.close(fd)
    for f in files:
        path = root / f["Name"]
        partial = path.with_name(path.name + ".ranges")
        if f["Name"] in handles:
            if sha256(partial) != f["Sha256"]:
                raise ValueError(f"SHA256 mismatch: {f['Name']}")
            partial.replace(path)
        for marker in root.glob(f".{f['Name']}.*.done"):
            marker.unlink()
    (root / "source.json").write_text(json.dumps(dict(provider="ModelScope", repo_id=repo,
        revision="master; exact downloaded content pinned by download_manifest.json SHA256",
        verified_files=records), indent=2)+"\n")
    print(f"Completed and SHA256 verified: {repo}", flush=True)


if __name__ == "__main__":
    fetch("AI-ModelScope/vicuna-7b-v1.5", "vicuna-7b-v1.5")
    fetch("skyline2006/llama-7b", "llama1-7b-ms")
