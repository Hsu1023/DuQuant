"""Full-model inference harness for the experimental LLaMA-2 W4A4 backend.

Graphs capture a fixed prompt length and generation length. Each decode step
uses the previous generated token and advances the cache; EOS is deliberately
ignored for fixed-work benchmarking. This is not a serving engine.
"""

import gc
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .decoder import (
    LlamaDecoderBlock,
    _rms,
    load_layer_weights,
    random_rotations,
    rms_reduction_width,
)
import triton


class LlamaInferenceModel(torch.nn.Module):
    def __init__(self, blocks, embedding, final_norm, lm_head, eps):
        super().__init__()
        self.blocks = torch.nn.ModuleList(blocks)
        self.register_buffer("embedding", embedding)
        self.register_buffer("final_norm", final_norm)
        self.register_buffer("lm_head", lm_head)
        self.h = embedding.shape[1]
        self.eps = eps

    def caches(self, batch, capacity):
        return [block.cache(batch, capacity) for block in self.blocks]

    def forward(self, token_ids, caches, position=0, last_token_only=True):
        if len(caches) != len(self.blocks):
            raise ValueError("One KV cache per layer is required")
        x = F.embedding(token_ids, self.embedding)
        for block, cache in zip(self.blocks, caches):
            x = block(x, cache, position)
        # Generation needs only the last prediction; PPL scores every position.
        if last_token_only:
            x = x[:, -1:].contiguous()
        normalized = torch.empty_like(x)
        _rms[(x.numel() // self.h,)](
            x,
            self.final_norm,
            normalized,
            x,
            None,
            x.numel() // self.h,
            self.h,
            self.eps,
            False,
            triton.next_power_of_2(self.h),
            rms_reduction_width(x.numel() // self.h, self.h),
            enable_fp_fusion=False,
        )
        if last_token_only:
            return F.linear(normalized[:, 0], self.lm_head)
        return F.linear(normalized, self.lm_head)

    @classmethod
    @torch.inference_mode()
    def from_pretrained(cls, path, mode="cuda", seed=42, calibration=None, kv_bits=None):
        from safetensors import safe_open

        if mode not in ("cuda", "fp16", "reference", "integer_reference"):
            raise ValueError("Unsupported inference/reference mode")
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        calibration_info = None
        if calibration is not None:
            from .calibration import apply_smoothing

            calibration = Path(calibration)
            raw = (calibration / "manifest.json").read_bytes()
            calibration_info = json.loads(raw)
            if mode == "fp16":
                raise ValueError("Use the original FP16 checkpoint as baseline")
            if (
                calibration_info["status"] != "complete"
                or calibration_info["format_version"] != 1
                or len(calibration_info["layers"]) != cfg["num_hidden_layers"]
                or calibration_info["config_sha256"]
                != hashlib.sha256((path / "config.json").read_bytes()).hexdigest()
                or Path(calibration_info["model"]).resolve() != path.resolve()
            ):
                raise ValueError(
                    "Calibration is incomplete or belongs to a different model"
                )
            calibration_info["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        if cfg.get("model_type") not in ("llama", "mistral"):
            raise ValueError("Only LLaMA and Mistral decoder layouts are supported")
        if cfg["hidden_size"] // cfg["num_attention_heads"] != 128:
            raise ValueError("Only head_dim=128 is supported")
        if cfg.get("rope_scaling") or cfg.get("sliding_window"):
            raise ValueError("Scaled RoPE and sliding-window attention are not supported")
        if cfg.get("attention_bias", False) or cfg.get("mlp_bias", False):
            raise ValueError("Biased projections are not supported")
        torch.manual_seed(seed)
        blocks = []
        for idx in range(cfg["num_hidden_layers"]):
            weights, _ = load_layer_weights(path, idx)
            weight_clips = None
            quantization = None
            if calibration_info is not None:
                entry = calibration_info["layers"][idx]
                filename = calibration / entry["file"]
                if (
                    entry["index"] != idx
                    or hashlib.sha256(filename.read_bytes()).hexdigest()
                    != entry["sha256"]
                ):
                    raise ValueError("Calibration layer integrity check failed")
                state = torch.load(filename, map_location="cuda", weights_only=True)
                apply_smoothing(weights, state["scales"])
                rotations = state["rotations"]
                if calibration_info.get("lwc_epochs", 0) > 0:
                    weight_clips = state["weight_clips"]
                if calibration_info.get("ptq"):
                    quantization = state["quantization"]
                del state
            else:
                rotations = (
                    random_rotations(
                        cfg["hidden_size"], cfg["intermediate_size"], "cuda"
                    )
                    if mode != "fp16"
                    else {}
                )
            block = LlamaDecoderBlock(
                weights,
                rotations,
                mode,
                eps=cfg["rms_norm_eps"],
                max_position=cfg["max_position_embeddings"],
                rope_theta=cfg.get("rope_theta", 10000.0),
                weight_clips=weight_clips,
                kv_bits=kv_bits,
                quantization=quantization,
            )
            if block.kv_heads != cfg.get("num_key_value_heads", block.heads):
                raise ValueError("KV projection shape differs from model configuration")
            if blocks:
                # RoPE and the fixed attention transform are identical across layers.
                block.cos = blocks[0].cos
                block.sin = blocks[0].sin
                block.hadamard = blocks[0].hadamard
            blocks.append(block)
            del weights, rotations, block
            if idx % 8 == 7:
                print(
                    f"Loaded {idx + 1}/{cfg['num_hidden_layers']} layers ({mode})",
                    flush=True,
                )
        index = json.loads((path / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        extras = []
        for key in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
            with safe_open(str(path / index[key]), framework="pt", device="cpu") as f:
                extras.append(f.get_tensor(key).to(device="cuda", dtype=torch.float16))
        model = cls(blocks, *extras, cfg["rms_norm_eps"]).eval()
        model.calibration_info = calibration_info
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return model


class GenerationSession:
    def __init__(self, model, prompt_ids, new_tokens):
        if prompt_ids.ndim != 2 or prompt_ids.shape[1] < 1 or new_tokens < 1:
            raise ValueError("Nonempty prompt and generation required")
        self.model = model
        self.prompt = prompt_ids.clone()
        self.prompt_length = prompt_ids.shape[1]
        self.new_tokens = new_tokens
        capacity = self.prompt_length + new_tokens - 1
        if capacity > model.blocks[0].cos.shape[0]:
            raise ValueError("Request exceeds RoPE context limit")
        self.caches = model.caches(prompt_ids.shape[0], capacity)
        self.generated = torch.empty(
            (prompt_ids.shape[0], new_tokens),
            device=prompt_ids.device,
            dtype=torch.long,
        )
        self.prefill_graph = None
        self.decode_graph = None

    def prefill(self):
        logits = self.model(self.prompt, self.caches)
        self.generated[:, 0].copy_(logits.argmax(-1))
        return logits

    def decode(self):
        for step in range(1, self.new_tokens):
            logits = self.model(
                self.generated[:, step - 1 : step],
                self.caches,
                self.prompt_length + step - 1,
            )
            self.generated[:, step].copy_(logits.argmax(-1))

    @torch.inference_mode()
    def eager(self):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            self.prefill()
            self.decode()
        return self.generated

    @torch.inference_mode()
    def capture(self):
        # Warm every actual sequence length/position before graph capture.
        expected = self.eager().clone()
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        pool = torch.cuda.graph_pool_handle()
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            self.prefill_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.prefill_graph, pool=pool, stream=stream):
                self.prefill()
            self.decode_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.decode_graph, pool=pool, stream=stream):
                self.decode()
        torch.cuda.current_stream().wait_stream(stream)
        self.replay()
        torch.cuda.synchronize()
        if not torch.equal(expected, self.generated):
            raise AssertionError("Captured generation differs from eager generation")
        del expected
        gc.collect()
        torch.cuda.empty_cache()

    def replay(self):
        if self.prefill_graph is None:
            raise RuntimeError("Capture first")
        self.prefill_graph.replay()
        self.decode_graph.replay()
        return self.generated

    @property
    def kv_bytes(self):
        return sum(c.storage_bytes for c in self.caches)

    @property
    def kv_workspace_bytes(self):
        return sum(c.workspace_bytes for c in self.caches)


def model_storage_bytes(model):
    # Count unique backing storage, not duplicate views or shared RoPE buffers.
    seen = set()
    total = 0
    for tensor in list(model.buffers()) + list(model.parameters()):
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total
