"""FlatQuant's released non-orthogonal transforms with the common INT4 backend.

The learned clipping parameters are deliberately replaced by the comparison's
shared clipping search. Original comparison modules remain unchanged.
"""
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from comparison_models import ROOT, MODEL, BASE, NAMES, grouped_weights, identity_rotations
from duquant_cuda.backend import DuQuantLinearCUDA
from duquant_cuda.decoder import LlamaDecoderBlock, IntegerReferenceLinear, qdq
from duquant_cuda.kv4 import PackedKVCache, _quant
from duquant_cuda.model import LlamaInferenceModel

REPO = ROOT/'benchmarks/cache/FlatQuant'
CHECKPOINT = BASE/'flatquant_matrices.pth'
FOLDER = BASE/'flatquant'
PREFIXES = dict(qkv='self_attn.ln_trans', gate_up='mlp.up_gate_trans', down='mlp.down_trans')


def kron(x, left, right):
    shape = x.shape
    z = x.reshape(-1, left.shape[0], right.shape[0])
    return (left.T @ (z @ right)).reshape(shape)


def transforms(state):
    result = {}
    for name, prefix in PREFIXES.items():
        result[name] = dict(left=state[prefix+'.matrix_left'].half(),
                            right=state[prefix+'.matrix_right'].half())
    result['o'] = dict(left=state['self_attn.o_trans.matrix'].half())
    result['k'] = state['self_attn.kcache_trans.matrix'].half()
    result['q'] = state['self_attn.kcache_trans.matrix_inv_t'].half()
    return result


@torch.inference_mode()
def transform_weights(original, state):
    """Official FP64 inverse-transpose folding, excluding learned clipping."""
    w = {k: v.clone() for k, v in original.items()}
    for name, projections in [('qkv', ['self_attn.'+n+'_proj.weight' for n in ('q','k','v')]),
                               ('gate_up', ['mlp.'+n+'_proj.weight' for n in ('gate','up')]),
                               ('down', ['mlp.down_proj.weight'])]:
        p = PREFIXES[name]
        for key in projections:
            z = w[key].double()/state[p+'.diag_scale'].double()
            z = kron(z, state[p+'.matrix_left_inv'].double(), state[p+'.matrix_right_inv'].double())
            if key == 'self_attn.v_proj.weight':
                z = (z.T.reshape(-1, 128) @ state['self_attn.vcache_trans.matrix'].double()).reshape(z.T.shape).T
            w[key] = z.half().contiguous()
    w['self_attn.o_proj.weight'] = kron(w['self_attn.o_proj.weight'].double(),
        state['self_attn.o_trans.matrix_inv_t'].double(),
        state['self_attn.vcache_trans.matrix_inv_t'].double()).half()
    # Match official reparameterization: input diagonals fold into RMSNorm,
    # down-projection diagonal folds into up-projection output channels.
    for norm, prefix in [('input_layernorm.weight', PREFIXES['qkv']),
                         ('post_attention_layernorm.weight', PREFIXES['gate_up'])]:
        w[norm] = (w[norm].double()*state[prefix+'.diag_scale'].double()).half()
    w['mlp.up_proj.weight'] = (w['mlp.up_proj.weight'].double()*
                               state[PREFIXES['down']+'.diag_scale'].double()[:, None]).half()
    return w


class FlatLinear(torch.nn.Module):
    def __init__(self, w, trans, state=None, act_clip=None, mode='reference', collect=False, hessian=False):
        super().__init__()
        self.register_buffer('left', trans['left'])
        self.register_buffer('right', trans.get('right'))
        self.act_clip = act_clip
        self.samples = [] if collect else None
        self.hessian = torch.zeros((w.shape[1], w.shape[1]), device=w.device) if hessian else None
        self.tokens = 0
        self.register_buffer('weight', None)
        if state is not None and mode == 'cuda':
            self.inner = DuQuantLinearCUDA(w, *identity_rotations(w.shape[1], w.device),
                                          quantized_weight=state, act_clip=act_clip)
        elif state is not None and mode == 'integer_reference':
            self.inner = IntegerReferenceLinear(w, identity_rotations(w.shape[1], w.device),
                                                 quantized_weight=state, act_clip=act_clip)
        else:
            self.weight = w if state is None else (state['codes'].half()-state['zero'][:, None])*state['scale'][:, None]

    def basis(self, x):
        if self.right is not None:
            return kron(x, self.left, self.right)
        shape = x.shape
        return (self.left.T @ x.reshape(-1, self.left.shape[0], 128)).reshape(shape)

    def forward(self, x):
        z = self.basis(x).contiguous()
        if self.samples is not None:
            from search_ptq import sample_rows
            self.samples.append(sample_rows(z))
        if hasattr(self, 'inner'):
            return self.inner(z)
        if self.act_clip is not None:
            z = qdq(z, self.act_clip)
        if self.hessian is not None:
            zf = z.reshape(-1, z.shape[-1]).float()
            self.hessian.addmm_(zf.T, zf)
            self.tokens += len(zf)
        return F.linear(z, self.weight)


def prepare_qkv(packed, cos, sin, position, qmat, kmat):
    b, t, width = packed.shape
    h = width//3
    q, k, v = [z.reshape(b, t, h//128, 128).transpose(1, 2) for z in packed.split(h, -1)]
    co, si = cos[position:position+t], sin[position:position+t]
    def rope(z):
        return z*co + torch.cat((-z[..., 64:], z[..., :64]), -1)*si
    q = (rope(q) @ qmat).contiguous()
    k = (rope(k) @ kmat).contiguous()
    return q, k, v.contiguous()


@triton.jit
def _pack_transformed(K, V, KC, VC, KP, VP, KD, VD, POS,
                      T: tl.constexpr, ROWS: tl.constexpr, CAP: tl.constexpr,
                      KCLIP: tl.constexpr, VCLIP: tl.constexpr, BLOCK: tl.constexpr):
    r = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, 128)
    k = tl.load(K+r[:, None]*128+d[None, :], r[:, None]<ROWS, 0).to(tl.float32)
    v = tl.load(V+r[:, None]*128+d[None, :], r[:, None]<ROWS, 0).to(tl.float32)
    ki, ks, kz, kd = _quant(k, KCLIP)
    vi, vs, vz, vd = _quant(v, VCLIP)
    kl, kh = tl.split(ki.reshape(BLOCK, 64, 2))
    vl, vh = tl.split(vi.reshape(BLOCK, 64, 2))
    dst = (r//T)*CAP + POS + r%T
    dd = tl.arange(0, 64)
    tl.store(KC+dst[:, None]*64+dd[None, :], kl | (kh << 4), r[:, None]<ROWS)
    tl.store(VC+dst[:, None]*64+dd[None, :], vl | (vh << 4), r[:, None]<ROWS)
    tl.store(KP+dst*2, ks, r<ROWS); tl.store(KP+dst*2+1, kz, r<ROWS)
    tl.store(VP+dst*2, vs, r<ROWS); tl.store(VP+dst*2+1, vz, r<ROWS)
    tl.store(KD+r[:, None]*128+d[None, :], kd, r[:, None]<ROWS)
    tl.store(VD+r[:, None]*128+d[None, :], vd, r[:, None]<ROWS)


class FlatKVCache(PackedKVCache):
    def prepare(self, packed, cos, sin, position):
        q, k, v = prepare_qkv(packed, cos, sin, position, self.qmat, self.kmat)
        kd, vd = torch.empty_like(k), torch.empty_like(v)
        rows = k.numel()//128
        _pack_transformed[(triton.cdiv(rows, 16),)](
            k, v, self.k, self.v, self.k_params, self.v_params, kd, vd, position,
            k.shape[2], rows, self.capacity, self.k_clip, self.v_clip, 16,
            enable_fp_fusion=False)
        return q, kd, vd


class FlatBlock(LlamaDecoderBlock):
    def cache(self, batch, capacity):
        if self.mode == 'cuda' and self.kv_bits == 4:
            cache = FlatKVCache(batch, self.kv_heads, self.heads, capacity,
                                self.norm1.device, self.k_clip, self.v_clip)
            cache.qmat, cache.kmat = self.qmat, self.kmat
            return cache
        return super().cache(batch, capacity)

    def _qkv_reference(self, packed, cache, pos):
        q, k, v = prepare_qkv(packed, self.cos, self.sin, pos, self.qmat, self.kmat)
        if hasattr(self, 'k_samples'):
            from search_ptq import sample_rows
            self.k_samples.append(sample_rows(k))
            self.v_samples.append(sample_rows(v.transpose(1,2).contiguous()))
        if self.kv_bits == 4:
            k, v = qdq(k, self.k_clip), qdq(v, self.v_clip)
        cache.k[:, :, pos:pos+k.shape[2]].copy_(k)
        cache.v[:, :, pos:pos+v.shape[2]].copy_(v)
        return q


def make_block(w, trans, cfg, plan=None, mode='reference', quantize_kv=True):
    block = FlatBlock(w, {}, 'fp16', eps=cfg['rms_norm_eps'],
                      max_position=cfg['max_position_embeddings'], kv_bits=16,
                      rope_theta=cfg.get('rope_theta', 10000.))
    block.mode, block.kv_bits = mode, 4 if quantize_kv else 16
    block.k_clip, block.v_clip = plan['kv_clips'] if plan else (.9, 1.)
    block.register_buffer('qmat', trans['q'])
    block.register_buffer('kmat', trans['k'])
    for name, weight in grouped_weights(w).items():
        setattr(block, name, FlatLinear(weight, trans[name],
            state=plan['weights'][name] if plan else None,
            act_clip=plan['act_clips'][name] if plan else None, mode=mode))
    return block


def original_extras():
    index = json.loads((MODEL/'model.safetensors.index.json').read_text())['weight_map']
    extra = {}
    for dst, key in [('embedding','model.embed_tokens.weight'), ('final_norm','model.norm.weight'), ('lm_head','lm_head.weight')]:
        with safe_open(str(MODEL/index[key]), framework='pt') as f:
            extra[dst] = f.get_tensor(key).half()
    return extra


@torch.inference_mode()
def load_model(mode='cuda'):
    meta = json.loads((FOLDER/'manifest.json').read_text())
    assert meta['status'] == 'complete' and len(meta['layers']) == 32
    cfg = json.loads((MODEL/'config.json').read_text())
    blocks = []
    for i in range(32):
        state = torch.load(FOLDER/'rotated'/f'layer_{i:02d}.pt', map_location='cuda', weights_only=True)
        plan = torch.load(FOLDER/'gptq'/f'layer_{i:02d}.pt', map_location='cuda', weights_only=True)
        block = make_block(state['weights'], state['transforms'], cfg, plan, mode)
        if blocks:
            block.cos, block.sin, block.hadamard = blocks[0].cos, blocks[0].sin, blocks[0].hadamard
        blocks.append(block)
        del state, plan
    extra = torch.load(FOLDER/'extras.pt', map_location='cuda', weights_only=True)
    return LlamaInferenceModel(blocks, **extra, eps=cfg['rms_norm_eps']).eval()
