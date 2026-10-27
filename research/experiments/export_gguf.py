"""Export an OpenBonsai shared-sign ternary state to Prism's PTQ1_0 GGUF layout.

Target layout: prism-ml/Ternary-Bonsai-2-9B-gguf, Ternary-Bonsai-2-9B-PTQ1_0.gguf (rev 5b2abb0,
1,997,486,976 B), loadable by Prism's llama.cpp fork (commit c9c34bc). Same 49 KVs (order and
types), same 427 tensors (names, order, types, ne, offsets); only general.name, the Hadamard
sign vectors (ours) and, if the snapshot differs, tokenizer data change.

  python export_gguf.py --state ternary_state.pt --endpoints endpoint_state.pt [--e2e e2e_state.pt] \
      --snapshot <HF dir> --out model.gguf [--name OpenBonsai-Qwen3.5-9B-PTQ1_0] [--verify]
  python export_gguf.py --verify-only --out model.gguf --state ... --endpoints ... [--e2e ...] --snapshot ...
  python export_gguf.py --selftest

Conventions (see the export spec; all checked against the fork's source):
  * Folded weights: stored W_rot = W diag(s) blockdiag(H4096), H normalized Sylvester; one shared
    sign vector per input width (4096, 12288). The runtime computes W_rot @ H(s*x).
    token_embd stores E_rot = (E*s) H and the lookup restores e = s * (H z).
  * PTQ1_0 (ggml type 143): 128 weights -> 28 bytes {qs[24], qh[2], fp16 d}; base-3 digits
    (code+1), byte = ceil(q*256/243). Negative (sign-bit) scales are normalized by negating
    the group's codes and clearing the bit, and zero-scale groups get all-zero codes (same
    function; bytes as quantize_row_ptq1_0_ref writes them; the official file has neither).
  * Signs: the file carries one vector per input width; it must equal the stored "signs" of
    every module of that width AND signs_from_seed(seed) (what e2e_kd.py rebuilds), else refused.
  * e2e_state.pt norms that differ from the snapshot need --e2e-norms (export) or
    --ignore-e2e-norms (deliberately drop); they are never dropped silently.
  * GatedDeltaNet: V heads grouped -> tiled (tiled t = r*n_k + k takes grouped g = k*rep + r) on
    in_proj_qkv V rows, in_proj_z rows, conv1d V channels, in_proj_a/b rows, A_log, dt_bias.
    out_proj (ssm_out) columns stay grouped (prism.hadamard.gdn_v_grouped = true).
  * F32: zero-centered RMSNorm weights get +1 (input/post-attention/final norm, q_norm, k_norm),
    linear_attn.norm does not; ssm_a = -exp(A_log); conv1d squeezed. BF16 -> F32 is exact.

The writer is self-contained (struct + numpy), streams tensor data to disk and packs the
embedding/head in row chunks. --verify re-reads the file with an independent reader/decoder.
CPU only.
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only, before torch is imported

import argparse
import gzip
import hashlib
import json
import math
import re
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
DEFAULT_HEADER = HERE / "ref" / "bonsai2_9b_header.json"
DEFAULT_TOKENS = HERE / "ref" / "bonsai2_9b_tokens.json.gz"  # official tokenizer.ggml.tokens (sha256 == header's)
OFFICIAL_SIZE = 1_997_486_976
DEFAULT_NAME = "OpenBonsai-Qwen3.5-9B-PTQ1_0"

# ---------------------------------------------------------------------------------------------
# GGUF / ggml constants
# ---------------------------------------------------------------------------------------------
GGUF_VERSION = 3
ALIGN = 32
GGML_F32 = 0
GGML_PTQ1_0 = 143
FTYPE_MOSTLY_PTQ1_0 = 143  # include/llama.h; gguf-py's LlamaFileType says 129, the C++ side and the file use 143
QK = 128
BLOCK_BYTES = 28
HADAMARD_BLOCK = 4096
VT = {"u8": 0, "i8": 1, "u16": 2, "i16": 3, "u32": 4, "i32": 5, "f32": 6, "bool": 7, "str": 8, "arr": 9,
      "u64": 10, "i64": 11, "f64": 12}
VT_NAME = {v: k for k, v in VT.items()}
SCALAR_FMT = {"u8": "<B", "i8": "<b", "u16": "<H", "i16": "<h", "u32": "<I", "i32": "<i", "f32": "<f",
              "bool": "<?", "u64": "<Q", "i64": "<q", "f64": "<d"}
NP_ELEM = {"u8": "<u1", "i8": "<i1", "u16": "<u2", "i16": "<i2", "u32": "<u4", "i32": "<i4", "f32": "<f4",
           "bool": "<u1", "u64": "<u8", "i64": "<i8", "f64": "<f8"}
TOKEN_NORMAL, TOKEN_CONTROL, TOKEN_USER_DEFINED, TOKEN_UNUSED = 1, 3, 4, 5
HF = "model.language_model."

# Prism converter's get_vocab_base_pre probe text and the hashes it maps to "qwen35".
CHKTXT = ('\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n\U0001f680 (normal) \U0001f636\u200d\U0001f32b\ufe0f '
          '(multiple emojis concatenated) \u2705 \U0001f999\U0001f999 3 33 333 3333 33333 333333 3333333 33333333 '
          '3.3 3..3 3...3 \u1780\u17b6\u1793\u17cb\u178f\u17c2\u1796\u17b7\u179f\u17c1\u179f\u17a2\u17b6\u1785'
          '\U0001f601 ?\u6211\u60f3\u5728apple\u5de5\u4f5c1314151\u5929\uff5e ------======= '
          '\u043d\u0435\u0449\u043e \u043d\u0430 \u0411\u044a\u043b\u0433\u0430\u0440\u0441\u043a\u0438 '
          '\'\'\'\'\'\'```````""""......!!!!!!?????? I\'ve been \'told he\'s there, \'RE you sure? '
          '\'M not sure I\'ll make it, \'D you like some tea? We\'Ve a\'lL')
CHKTXT_SHA256 = "a36179c1e2b1cfcfbfad1dc779addad2ffc778201e4cf0d538bddeafa2c526da"
QWEN35_PRE_HASHES = {"d30d75d9059f1aa2c19359de71047b3ae408c70875e8a3ccf8c5fba56c9d8af4",   # Qwen/Qwen3.5-9B-Instruct
                     "1444df51289cfa8063b96f0e62b1125440111bc79a52003ea14b6eac7016fd5f"}   # openbmb/MiniCPM-V-4_6

# KVs that cannot be derived from the HF snapshot; values come from the official header.
COPY_FROM_HEADER = {"general.type", "general.size_label", "general.sampling.min_p", "general.sampling.temp",
                    "general.sampling.top_p", "general.sampling.top_k", "general.sampling.penalty_repeat"}
BY_DESIGN = {"general.name", "prism.hadamard.sign_values"}


def align_up(n, a=ALIGN):
    return (n + a - 1) // a * a


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------------------------
# PTQ1_0 codec (writer side)
# ---------------------------------------------------------------------------------------------
def pack_ptq1_0(codes: np.ndarray, sbits: np.ndarray) -> np.ndarray:
    """codes int8 [rows, in] in {-1,0,1}; sbits uint16 fp16 scale bits [rows, in/128] -> uint8 [rows, in/128*28].

    Arithmetic form of the spec's verified packer (selftest checks byte equality with it):
    qs[m]    <- weights m + 16n,      n = 0..4 (n = 0 most significant trit)
    qs[16+m] <- weights 80 + m + 8n,  n = 0..4
    qh[h]    <- weights 120 + h + 2m, m = 0..3, then *3
    byte = ceil(q * 256 / 243); d (fp16) in bytes 26-27 little-endian.
    """
    rows, width = codes.shape
    assert width % QK == 0 and sbits.shape == (rows, width // QK)
    nb = rows * (width // QK)
    d = (codes.reshape(nb, QK) + np.int8(1)).astype(np.uint16)
    out = np.empty((nb, BLOCK_BYTES), np.uint8)
    q = 81 * d[:, 0:16] + 27 * d[:, 16:32] + 9 * d[:, 32:48] + 3 * d[:, 48:64] + d[:, 64:80]
    out[:, 0:16] = (q * 256 + 242) // 243
    q = 81 * d[:, 80:88] + 27 * d[:, 88:96] + 9 * d[:, 96:104] + 3 * d[:, 104:112] + d[:, 112:120]
    out[:, 16:24] = (q * 256 + 242) // 243
    q = (27 * d[:, 120:122] + 9 * d[:, 122:124] + 3 * d[:, 124:126] + d[:, 126:128]) * 3
    out[:, 24:26] = (q * 256 + 242) // 243
    out[:, 26:28] = np.ascontiguousarray(sbits, dtype="<u2").reshape(nb, 1).view(np.uint8)
    return out.reshape(rows, (width // QK) * BLOCK_BYTES)


def normalize_scale_signs(codes: np.ndarray, sbits: np.ndarray):
    """Canonicalize groups in place (same function, bytes as quantize_row_ptq1_0_ref would write them):
    a scale with the fp16 sign bit set (incl. -0.0) -> negate the group's codes and clear the bit;
    then a zero scale -> zero the group's codes (they are dead, and the C quantizer writes d = 0 with
    all-zero codes). Returns (negative-scale groups, zero-scale groups whose nonzero codes were cleared)."""
    rows, g = sbits.shape
    cv = codes.reshape(rows, g, QK)
    assert np.shares_memory(cv, codes), "codes must be contiguous (in-place normalization)"
    neg = (sbits & 0x8000) != 0
    n_neg = int(neg.sum())
    if n_neg:
        cv[neg] = -cv[neg]
        sbits &= 0x7FFF
    dead = (sbits == 0) & cv.any(-1)
    n_dead = int(dead.sum())
    if n_dead:
        cv[dead] = 0
    return n_neg, n_dead


# ---------------------------------------------------------------------------------------------
# PTQ1_0 codec (verify side, independent: digit_n(b) = ((uint8)(b*3^n) * 3) >> 8)
# ---------------------------------------------------------------------------------------------
_DIGITS = np.array([[(((b * 3 ** n) & 0xFF) * 3) >> 8 for n in range(5)] for b in range(256)], np.int8)


def decode_ptq1_0(raw: np.ndarray, width: int):
    """raw uint8 [rows, width/128*28] -> (codes int8 [rows, width], scale bits uint16 [rows, width/128])."""
    rows = raw.shape[0]
    g = width // QK
    b = np.ascontiguousarray(raw).reshape(rows * g, BLOCK_BYTES)
    nb = b.shape[0]
    codes = np.empty((nb, QK), np.int8)
    codes[:, 0:80] = _DIGITS[b[:, 0:16]].transpose(0, 2, 1).reshape(nb, 80)        # weight m + 16n
    codes[:, 80:120] = _DIGITS[b[:, 16:24]].transpose(0, 2, 1).reshape(nb, 40)     # 80 + m + 8n
    codes[:, 120:128] = _DIGITS[b[:, 24:26], :4].transpose(0, 2, 1).reshape(nb, 8)  # 120 + h + 2n
    codes -= 1
    sbits = b[:, 26:28].copy().view("<u2").reshape(rows, g)
    return codes.reshape(rows, width), sbits


def _spec_pack(codes, scales):
    """The spec's verified einsum packer, verbatim (selftest reference only)."""
    pw = np.array([81, 27, 9, 3, 1], np.int32)
    r, w = codes.shape
    d = codes.reshape(-1, 128).astype(np.int32) + 1
    o = np.empty((d.shape[0], 28), np.uint8)
    o[:, 0:16] = (np.einsum('bnm,n->bm', d[:, :80].reshape(-1, 5, 16), pw) * 256 + 242) // 243
    o[:, 16:24] = (np.einsum('bnm,n->bm', d[:, 80:120].reshape(-1, 5, 8), pw) * 256 + 242) // 243
    o[:, 24:26] = (np.einsum('bmh,m->bh', d[:, 120:].reshape(-1, 4, 2), pw[1:]) * 3 * 256 + 242) // 243
    o[:, 26:28] = scales.astype('<f2').reshape(-1, 1).view(np.uint8)
    return o.reshape(r, w // 128 * 28)


def _c_ref_quant_block(x):
    """Port of ggml-quants.c quantize_row_ptq1_0_ref for one 128-block (float32 semantics)."""
    x = x.astype(np.float32)
    d = np.float32(np.max(np.abs(x)))
    idv = np.float32(1.0) / d if d else np.float32(0.0)

    def lround(v):
        v = np.float32(v)
        return int(math.floor(abs(float(v)) + 0.5)) * (1 if v >= 0 else -1)

    qs, p, j = [0] * 24, 0, 0
    for c in (32, 16, 8):
        while j + c <= 24:
            for m in range(c):
                q = 0
                for n in range(5):
                    q = (q * 3 + lround(x[p + m + n * c] * idv) + 1) & 0xFF
                qs[j + m] = ((q * 256 + 242) // 243) & 0xFF
            p += 5 * c
            j += c
    qh = []
    for h in range(2):
        q = 0
        for m in range(4):
            q = (q * 3 + lround(x[p + h + m * 2] * idv) + 1) & 0xFF
        q = (q * 3) & 0xFF
        qh.append(((q * 256 + 242) // 243) & 0xFF)
    return bytes(qs + qh) + np.float16(d).tobytes()


def selftest():
    import itertools
    rng = np.random.default_rng(0)
    # every 5-trit and 4-trit digit pattern through every byte slot
    p5 = np.array(list(itertools.product([-1, 0, 1], repeat=5)), np.int8)
    p4 = np.array(list(itertools.product([-1, 0, 1], repeat=4)), np.int8)
    blk = np.zeros((243, 128), np.int8)
    for m in range(16):
        for n in range(5):
            blk[:, m + 16 * n] = np.roll(p5[:, n], m)
    for m in range(8):
        for n in range(5):
            blk[:, 80 + m + 8 * n] = np.roll(p5[:, n], 3 * m)
    for h in range(2):
        for n in range(4):
            blk[:81, 120 + h + 2 * n] = np.roll(p4[:, n], h)
    sb = np.full((243, 1), np.float16(0.5).view(np.uint16), np.uint16)
    raw = pack_ptq1_0(blk, sb)
    assert np.array_equal(raw, _spec_pack(blk, sb.view(np.float16))), "packer != spec packer (patterns)"
    c2, s2 = decode_ptq1_0(raw, 128)
    assert np.array_equal(c2, blk) and np.array_equal(s2, sb), "pattern round trip"
    # random rows incl. an all-zero group and a full-range scale set
    rows, width = 32, 4096
    codes = rng.choice(np.array([-1, 0, 1], np.int8), size=(rows, width))
    codes[0, :128] = 0
    sc = rng.uniform(0.004, 0.05, size=(rows, width // 128)).astype(np.float16)
    raw = pack_ptq1_0(codes, sc.view(np.uint16))
    assert np.array_equal(raw, _spec_pack(codes, sc)), "packer != spec packer (random)"
    c2, s2 = decode_ptq1_0(raw, width)
    assert np.array_equal(c2, codes) and np.array_equal(s2, sc.view(np.uint16)), "random round trip"
    # byte parity with the C reference quantizer on exact code*fp16(d) floats (nonzero groups)
    deq = (codes.reshape(rows, -1, 128).astype(np.float32) * sc.astype(np.float32)[..., None]).reshape(rows, width)
    bad = 0
    for r in range(2):
        for g in range(width // 128):
            ref = _c_ref_quant_block(deq[r, g * 128:(g + 1) * 128])
            mine = bytes(raw[r, g * 28:(g + 1) * 28])
            if codes[r, g * 128:(g + 1) * 128].any():
                bad += ref != mine
            else:
                bad += ref[:26] != mine[:26]  # C ref writes d = 0 for an all-zero group
    assert bad == 0, f"{bad} groups differ from quantize_row_ptq1_0_ref"
    # sanity bytes from the spec
    z = pack_ptq1_0(np.zeros((1, 128), np.int8), np.zeros((1, 1), np.uint16))[0]
    assert z[0] == 128 and z[23] == 128 and z[24] == 127 and z[25] == 127
    assert (pack_ptq1_0(np.full((1, 128), -1, np.int8), np.zeros((1, 1), np.uint16))[0, :26] == 0).all()
    ones = pack_ptq1_0(np.full((1, 128), 1, np.int8), np.zeros((1, 1), np.uint16))[0]
    assert (ones[:24] == 255).all() and (ones[24:26] == 253).all()  # qh holds 4 trits then *3: q = 240
    # scale-sign normalization keeps the function
    c3 = codes.copy()
    s3 = sc.view(np.uint16).copy()
    s3[1, 3] |= 0x8000
    c3[1, 3 * 128:4 * 128] *= -1
    f_before = c3.reshape(rows, -1, 128).astype(np.float32) * s3.view(np.float16).astype(np.float32)[..., None]
    assert normalize_scale_signs(c3, s3) == (1, 0)
    f_after = c3.reshape(rows, -1, 128).astype(np.float32) * s3.view(np.float16).astype(np.float32)[..., None]
    assert np.array_equal(f_before, f_after) and np.array_equal(c3, codes)
    # zero scales (-0.0 and +0) with nonzero codes: codes cleared, bytes == C ref quantizer on the (zero) weights
    c4, s4 = codes.copy(), sc.view(np.uint16).copy()
    s4[2, 5], s4[3, 7] = 0x8000, 0x0000
    assert c4[2, 5 * 128:6 * 128].any() and c4[3, 7 * 128:8 * 128].any()
    f_before = c4.reshape(rows, -1, 128).astype(np.float32) * s4.view(np.float16).astype(np.float32)[..., None]
    assert normalize_scale_signs(c4, s4) == (1, 2)
    f_after = c4.reshape(rows, -1, 128).astype(np.float32) * s4.view(np.float16).astype(np.float32)[..., None]
    assert np.array_equal(f_before, f_after) and not c4[2, 5 * 128:6 * 128].any() and not c4[3, 7 * 128:8 * 128].any()
    raw4 = pack_ptq1_0(c4, s4)
    zero_ref = _c_ref_quant_block(np.zeros(128, np.float32))
    assert bytes(raw4[2, 5 * 28:6 * 28]) == zero_ref and bytes(raw4[3, 7 * 28:8 * 28]) == zero_ref
    return ("PTQ1_0 codec selftest ok (243/81 digit patterns, random round trip, spec-packer parity, C-ref parity on "
            "64 groups, sign/zero-scale canonicalization)")


# ---------------------------------------------------------------------------------------------
# Rotations and permutations
# ---------------------------------------------------------------------------------------------
def hadamard_sylvester(n):
    h = np.ones((1, 1))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / math.sqrt(n)


def hadamard_popcount(n):
    """The runtime's construction: H[i,j] = (-1)^popcount(i&j) / sqrt(n) (llama-model.cpp)."""
    a = np.arange(n, dtype=np.uint32)
    p = np.bitwise_and.outer(a, a)
    for s in (16, 8, 4, 2, 1):
        p ^= p >> s
    return np.where(p & 1, -1.0, 1.0) / math.sqrt(n)


def signs_from_seed(seed, in_features):
    """gptq_ternary.signs_from_seed / e2e_kd.signs_for."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(0, 2, (in_features,), generator=g) * 2 - 1).to(torch.int8)


def v_head_perm(n_k, n_v, head_dim):
    """Prism converter's _reorder_v_heads as an index (grouped -> tiled): out[i] = in[perm[i]]."""
    rep = n_v // n_k
    return np.arange(n_v * head_dim).reshape(n_k, rep, head_dim).transpose(1, 0, 2).reshape(-1)


# ---------------------------------------------------------------------------------------------
# Tensor plan (file order)
# ---------------------------------------------------------------------------------------------
@dataclass
class TensorPlan:
    name: str
    gtype: int
    ne: tuple            # GGUF order: ne[0] innermost (input width)
    kind: str            # "tern" (state/endpoint key) or "f32" (HF tensor name)
    key: str
    op: str = ""         # f32 transform: copy | plus1 | neg_exp | squeeze
    perm: np.ndarray | None = None  # writer permutation of dim 0 (rows / channels / heads)
    nbytes: int = 0
    offset: int = 0


def tensor_nbytes(gtype, ne):
    n = int(np.prod(ne))
    if gtype == GGML_PTQ1_0:
        assert ne[0] % QK == 0
        return n // QK * BLOCK_BYTES
    assert gtype == GGML_F32
    return 4 * n


HF2GGUF = {"linear_attn.in_proj_qkv": "attn_qkv", "linear_attn.in_proj_z": "attn_gate",
           "linear_attn.out_proj": "ssm_out", "mlp.down_proj": "ffn_down", "mlp.gate_proj": "ffn_gate",
           "mlp.up_proj": "ffn_up", "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k",
           "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output"}


def build_plan(tc):
    hid, ffn, vocab = tc["hidden_size"], tc["intermediate_size"], tc["vocab_size"]
    n_k, n_v = tc["linear_num_key_heads"], tc["linear_num_value_heads"]
    dk, dv = tc["linear_key_head_dim"], tc["linear_value_head_dim"]
    nh, nkv, hd = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
    ksz = tc["linear_conv_kernel_dim"]
    qk_dim = 2 * n_k * dk
    vperm = v_head_perm(n_k, n_v, dv)
    qkv_perm = np.concatenate([np.arange(qk_dim), qk_dim + vperm])
    head_perm = v_head_perm(n_k, n_v, 1)
    q_rows = nh * hd * (2 if tc.get("attn_output_gate", False) else 1)

    def tern(name, key, out_f, in_f, perm=None):
        ne = (in_f, out_f)
        return TensorPlan(name, GGML_PTQ1_0, ne, "tern", key, perm=perm, nbytes=tensor_nbytes(GGML_PTQ1_0, ne))

    def f32(name, hf, shape, op, perm=None):
        ne = tuple(reversed(shape))
        return TensorPlan(name, GGML_F32, ne, "f32", hf, op, perm, tensor_nbytes(GGML_F32, ne))

    plan = [tern("output.weight", "lm_head", vocab, hid),
            f32("output_norm.weight", HF + "norm.weight", [hid], "plus1"),
            tern("token_embd.weight", "embed_tokens", vocab, hid)]
    for li, typ in enumerate(tc["layer_types"]):
        L, b, k = f"{HF}layers.{li}.", f"blk.{li}.", f"layers.{li}."
        blk = [f32(b + "attn_norm.weight", L + "input_layernorm.weight", [hid], "plus1"),
               f32(b + "post_attention_norm.weight", L + "post_attention_layernorm.weight", [hid], "plus1"),
               tern(b + "ffn_down.weight", k + "mlp.down_proj", hid, ffn),
               tern(b + "ffn_gate.weight", k + "mlp.gate_proj", ffn, hid),
               tern(b + "ffn_up.weight", k + "mlp.up_proj", ffn, hid)]
        if typ == "linear_attention":
            blk += [tern(b + "attn_qkv.weight", k + "linear_attn.in_proj_qkv", qk_dim + n_v * dv, hid, qkv_perm),
                    tern(b + "attn_gate.weight", k + "linear_attn.in_proj_z", n_v * dv, hid, vperm),
                    tern(b + "ssm_out.weight", k + "linear_attn.out_proj", hid, n_v * dv),  # columns stay grouped
                    f32(b + "ssm_a", L + "linear_attn.A_log", [n_v], "neg_exp", head_perm),
                    f32(b + "ssm_alpha.weight", L + "linear_attn.in_proj_a.weight", [n_v, hid], "copy", head_perm),
                    f32(b + "ssm_beta.weight", L + "linear_attn.in_proj_b.weight", [n_v, hid], "copy", head_perm),
                    f32(b + "ssm_conv1d.weight", L + "linear_attn.conv1d.weight", [qk_dim + n_v * dv, ksz], "squeeze", qkv_perm),
                    f32(b + "ssm_dt.bias", L + "linear_attn.dt_bias", [n_v], "copy", head_perm),
                    f32(b + "ssm_norm.weight", L + "linear_attn.norm.weight", [dv], "copy")]
        elif typ == "full_attention":
            blk += [tern(b + "attn_q.weight", k + "self_attn.q_proj", q_rows, hid),
                    tern(b + "attn_k.weight", k + "self_attn.k_proj", nkv * hd, hid),
                    tern(b + "attn_v.weight", k + "self_attn.v_proj", nkv * hd, hid),
                    tern(b + "attn_output.weight", k + "self_attn.o_proj", hid, nh * hd),
                    f32(b + "attn_q_norm.weight", L + "self_attn.q_norm.weight", [hd], "plus1"),
                    f32(b + "attn_k_norm.weight", L + "self_attn.k_norm.weight", [hd], "plus1")]
        else:
            raise ValueError(f"unknown layer type {typ}")
        plan += sorted(blk, key=lambda t: t.name)
    off = 0
    for t in plan:
        t.offset = off
        off = align_up(off + t.nbytes)
    return plan


def hadamard_weight_names(tc):
    """output.weight, then the folded decoder weights in sorted-HF-name order (the official order)."""
    pairs = []
    for li, typ in enumerate(tc["layer_types"]):
        subs = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"] if typ == "linear_attention" \
            else ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
        for sub in subs + ["mlp.down_proj", "mlp.gate_proj", "mlp.up_proj"]:
            pairs.append((f"{HF}layers.{li}.{sub}.weight", f"blk.{li}.{HF2GGUF[sub]}.weight"))
    return ["output.weight"] + [g for _, g in sorted(pairs)]


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------
class Snapshot:
    def __init__(self, root):
        self.root = Path(root)
        self.map = json.loads((self.root / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
        self.handles = {}

    def _h(self, name):
        f = self.map[name]
        if f not in self.handles:
            self.handles[f] = safe_open(str(self.root / f), framework="pt")
        return self.handles[f]

    def get(self, name):
        return self._h(name).get_tensor(name)

    def rows(self, name, idx):
        sl = self._h(name).get_slice(name)
        return torch.cat([sl[int(i):int(i) + 1] for i in idx], 0)


class TernarySources:
    """Read-only, mmap'd view of ternary_state.pt / endpoint_state.pt / e2e_state.pt."""

    def __init__(self, state, endpoints, e2e=None):
        t0 = time.time()
        self.state = torch.load(state, map_location="cpu", mmap=True, weights_only=True)
        self.ep = torch.load(endpoints, map_location="cpu", mmap=True, weights_only=True)
        self.e2e = torch.load(e2e, map_location="cpu", mmap=True, weights_only=True) if e2e else None
        self.e2e_scales = (self.e2e or {}).get("scales", {})
        self.e2e_norms = (self.e2e or {}).get("norms", {})
        log(f"loaded states (mmap) in {time.time() - t0:.1f}s: {len(self.state['modules'])} modules, "
            f"endpoints {sorted(self.ep)}, e2e scales {len(self.e2e_scales)}, e2e norms {len(self.e2e_norms)}")

    def entry(self, key):
        return self.ep[key] if key in ("lm_head", "embed_tokens") else self.state["modules"][key]

    def has(self, key):
        return key in self.ep if key in ("lm_head", "embed_tokens") else key in self.state["modules"]

    def codes(self, key):
        return self.entry(key)["codes"]

    def scales(self, key):
        """fp16 [out, in/128]; the e2e-trained scales override the GPTQ ones when given."""
        if key in self.e2e_scales:
            return self.e2e_scales[key]
        return self.entry(key)["scales"]

    def signs(self, key, in_f):
        st = self.entry(key)
        s = st.get("signs")
        return (s if s is not None else signs_from_seed(st["seed"], in_f)).to(torch.int8)


def check_inputs(plan, src):
    """Shapes/dtypes/block, shared signs per width, e2e coverage. Returns ({width: int8 signs}, notes)."""
    notes = []
    args = src.state.get("args", {}) or {}
    if args.get("shared_signs") is False:
        raise SystemExit("state was quantized without --shared-signs: one sign vector per module cannot be "
                         "expressed in Prism's GGUF (one vector per input width). Re-run gptq_ternary.py --shared-signs.")
    signs, origin, seed_cache = {}, {}, {}
    tern = [t for t in plan if t.kind == "tern"]
    for t in tern:
        in_f, out_f = t.ne
        if not src.has(t.key):
            raise SystemExit(f"missing ternary module {t.key} for {t.name}")
        st = src.entry(t.key)
        c, s = st["codes"], src.scales(t.key)
        if c.dtype != torch.int8 or tuple(c.shape) != (out_f, in_f):
            raise SystemExit(f"{t.key}: codes {c.dtype} {tuple(c.shape)}, expected int8 {(out_f, in_f)}")
        if s.dtype != torch.float16 or tuple(s.shape) != (out_f, in_f // QK):
            raise SystemExit(f"{t.key}: scales {s.dtype} {tuple(s.shape)}, expected float16 {(out_f, in_f // QK)}")
        if tuple(st["scales"].shape) != (out_f, in_f // QK):
            raise SystemExit(f"{t.key}: stored GPTQ scales shape {tuple(st['scales'].shape)}")
        if int(st.get("block", 0)) != HADAMARD_BLOCK:
            raise SystemExit(f"{t.key}: block {st.get('block')} != {HADAMARD_BLOCK} (prism.hadamard.block_size)")
        sv = src.signs(t.key, in_f)
        if tuple(sv.shape) != (in_f,) or not bool(((sv == 1) | (sv == -1)).all()):
            raise SystemExit(f"{t.key}: signs must be {in_f} values of +-1")
        # e2e_kd.py (install_ternary / install_endpoints) and gptq_ternary.apply_state rebuild the rotation from
        # "seed", not from the stored "signs": the file must carry the vector the model was trained/evaluated with.
        if "seed" in st:
            ck = (int(st["seed"]), in_f)
            if ck not in seed_cache:
                seed_cache[ck] = signs_from_seed(*ck)
            if not torch.equal(sv, seed_cache[ck]):
                n_diff = int((sv != seed_cache[ck]).sum())
                raise SystemExit(f"{t.key}: stored signs differ from signs_from_seed(seed={ck[0]}, {in_f}) in {n_diff} "
                                 "entries; e2e_kd.py/gptq_ternary.py rebuild the rotation from the seed, so the GGUF "
                                 "would not match the trained/evaluated model. Fix the state (signs or seed).")
        if in_f not in signs:
            signs[in_f], origin[in_f] = sv.clone(), t.key
        elif not torch.equal(signs[in_f], sv):
            raise SystemExit(f"sign vector of {t.key} differs from {origin[in_f]} (width {in_f}): Prism's GGUF holds "
                             "exactly one sign vector per input width; the state is not shared-sign.")
    if sorted(signs) != [4096, 12288]:
        raise SystemExit(f"expected sign widths [4096, 12288], got {sorted(signs)}")
    for w, sv in signs.items():
        if torch.equal(sv, signs_from_seed(w, w)):
            notes.append(f"s{w}: equals signs_from_seed({w}, {w}) (gptq_ternary --shared-signs); {int((sv == 1).sum())} of {w} are +1")
        else:
            notes.append(f"s{w}: explicit vector (not signs_from_seed({w}, {w})); {int((sv == 1).sum())} of {w} are +1")
    plan_keys = {t.key for t in tern}
    extra = sorted(set(src.state["modules"]) - plan_keys)
    if extra:
        raise SystemExit(f"state has modules with no PTQ1_0 slot in the Prism layout: {extra[:5]} ...")
    if src.e2e is not None:
        used = [k for k in src.e2e_scales if k in plan_keys]
        unused = sorted(set(src.e2e_scales) - plan_keys)
        notes.append(f"e2e scale override for {len(used)} of {len(tern)} ternary tensors"
                     + (f"; unused e2e keys: {unused[:5]}" if unused else ""))
        for k in used:
            if tuple(src.e2e_scales[k].shape) != tuple(src.entry(k)["scales"].shape):
                raise SystemExit(f"e2e scales for {k} have shape {tuple(src.e2e_scales[k].shape)}")
    return signs, notes


def check_e2e_norms(plan, src, snap, use_e2e_norms, ignore_e2e_norms):
    """e2e_kd.py always saves e2e_state.pt['norms'] (every RMSNorm weight). Exporting the snapshot norms while
    the state carries trained ones would silently drop them, so that needs an explicit choice."""
    if use_e2e_norms and ignore_e2e_norms:
        raise SystemExit("--e2e-norms and --ignore-e2e-norms are mutually exclusive")
    if use_e2e_norms and not src.e2e_norms:
        raise SystemExit("--e2e-norms given, but there is no e2e_state.pt with a non-empty 'norms' entry (--e2e)")
    if not src.e2e_norms:
        return []
    slots = {t.key[len(HF):] for t in plan if t.kind == "f32" and t.op in ("plus1", "copy")}
    unknown = sorted(set(src.e2e_norms) - slots)
    changed = []
    for k, nv in src.e2e_norms.items():
        if k not in slots:
            continue
        x = snap.get(HF + k).to(torch.float32)
        if tuple(nv.shape) != tuple(x.shape):
            raise SystemExit(f"e2e norm {k}: shape {tuple(nv.shape)} != snapshot {tuple(x.shape)}")
        if not torch.equal(nv.to(torch.float32), x):
            changed.append(k)
    n = len(src.e2e_norms)
    if use_e2e_norms:
        if unknown:
            raise SystemExit(f"e2e norms with no F32 slot in the Prism layout (would be dropped): {unknown[:5]}")
        return [f"e2e norms: {len(changed)} of {n} differ from the snapshot; exporting the e2e values (--e2e-norms)"]
    if changed and not ignore_e2e_norms:
        raise SystemExit(f"e2e_state.pt carries {len(changed)} of {n} norm tensors that differ from the snapshot "
                         f"(e.g. {changed[:3]}): pass --e2e-norms to export them, or --ignore-e2e-norms to export "
                         "the snapshot norms deliberately")
    if changed:
        return [f"e2e norms: {len(changed)} of {n} differ from the snapshot; IGNORED (--ignore-e2e-norms), "
                "snapshot norms exported"]
    return [f"e2e norms: all {n - len(unknown)} matched tensors equal the snapshot (untrained); snapshot norms exported"
            + (f"; {len(unknown)} keys without an F32 slot: {unknown[:5]}" if unknown else "")]


# ---------------------------------------------------------------------------------------------
# Tokenizer (Prism converter logic: TextModel._set_vocab_gpt2 + gguf.SpecialVocab)
# ---------------------------------------------------------------------------------------------
def does_token_look_special(t):
    return (t in ("<pad>", "<mask>", "<2mass>", "[@BOS@]") or (t.startswith("<|") and t.endswith("|>"))
            or (t.startswith("<\uff5c") and t.endswith("\uff5c>")) or (t.startswith("<unused") and t.endswith(">")))


def derive_tokenizer(snapshot_dir, vocab_size):
    root = Path(snapshot_dir)
    tj = json.loads((root / "tokenizer.json").read_text(encoding="utf-8"))
    tcfg = json.loads((root / "tokenizer_config.json").read_text(encoding="utf-8"))
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    notes = []
    assert tj["model"]["type"] == "BPE"
    # AutoTokenizer.vocab = base vocab + added tokens (tokenizer.json added_tokens, then tokenizer_config's
    # added_tokens_decoder, which adds ids 248070-248076 for Qwen3.5)
    reverse = {i: t for t, i in tj["model"]["vocab"].items()}
    added = {}
    for a in tj.get("added_tokens", []):
        added[int(a["id"])] = {"content": a["content"], "special": bool(a.get("special", False)),
                               "normalized": bool(a.get("normalized", False))}
    for k, a in (tcfg.get("added_tokens_decoder") or {}).items():
        new = {"content": a["content"], "special": bool(a.get("special", False)), "normalized": bool(a.get("normalized", False))}
        if int(k) in added and added[int(k)] != new:
            notes.append(f"added token {k}: tokenizer.json {added[int(k)]} vs tokenizer_config {new}; using tokenizer_config")
        added[int(k)] = new
    for i, a in added.items():
        reverse[i] = a["content"]
    added_vocab = {a["content"]: i for i, a in added.items()}
    assert max(reverse) < vocab_size
    try:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(str(root / "tokenizer.json"))
    except Exception as e:  # pragma: no cover
        tk = None
        notes.append(f"tokenizers unavailable ({e}); non-normalized added tokens not round-tripped")
    tokens, types = [], []
    for i in range(vocab_size):
        if i not in reverse:
            tokens.append(f"[PAD{i}]")
            types.append(TOKEN_UNUSED)
            continue
        tok = reverse[i]
        if tok in added_vocab:
            if not added[i]["normalized"] and tk is not None:
                rt = tk.decode(tk.encode(tok, add_special_tokens=False).ids, skip_special_tokens=False)
                if rt != tok:
                    notes.append(f"added token {i} {tok!r} round-trips to {rt!r}")
                    tok = rt
            if added[i]["special"] or does_token_look_special(tok):
                types.append(TOKEN_CONTROL)
            else:
                tok = tok.replace("\u2581", " ")
                types.append(TOKEN_USER_DEFINED)
        else:
            types.append(TOKEN_NORMAL)
        tokens.append(tok)
    # tokenizer.ggml.pre via the converter's chkhsh
    pre = "qwen35"
    if tk is not None:
        assert hashlib.sha256(CHKTXT.encode()).hexdigest() == CHKTXT_SHA256, "probe text transcription"
        chkhsh = hashlib.sha256(str(tk.encode(CHKTXT).ids).encode()).hexdigest()
        if chkhsh not in QWEN35_PRE_HASHES:
            notes.append(f"pre-tokenizer hash {chkhsh} is not a known qwen35 hash; writing 'qwen35' anyway")
        else:
            notes.append(f"pre-tokenizer hash {chkhsh[:16]}... -> qwen35")
    # merges (SpecialVocab, load_merges=True)
    merges = tj["model"]["merges"]
    if merges and not isinstance(merges[0], str):
        merges = [" ".join("".join(chr(ord(c) + 256) if c == " " else c for c in part) for part in pair) for pair in merges]
    # special token ids: tokenizer_config.json first, then config.json (incl. text_config)
    special_types = ("bos", "eos", "unk", "sep", "pad", "cls", "mask")
    ids, add_flags = {}, {}

    def set_special(typ, tid):
        if isinstance(tid, int) and 0 <= tid < vocab_size and typ not in ids:
            ids[typ] = tid

    for typ in special_types:
        a = tcfg.get(f"add_{typ}_token")
        if isinstance(a, bool):
            add_flags[typ] = a
        e = tcfg.get(f"{typ}_token")
        content = e if isinstance(e, str) else (e.get("content") if isinstance(e, dict) else None)
        if content is None:
            continue
        set_special(typ, next((at["id"] for at in tj.get("added_tokens", []) if at["content"] == content), None))
    for typ in special_types:
        tid = cfg.get(f"{typ}_token_id")
        if tid is None and isinstance(cfg.get("text_config"), dict):
            tid = cfg["text_config"].get(f"{typ}_token_id")
        set_special(typ, tid)
    template = tcfg.get("chat_template")
    jinja = root / "chat_template.jinja"
    if jinja.is_file():
        jt = jinja.read_text(encoding="utf-8")
        if template is None:
            template = jt
        elif template != jt:
            notes.append("tokenizer_config chat_template differs from chat_template.jinja; using tokenizer_config's")
    return {"tokens": tokens, "token_type": types, "merges": merges, "pre": pre, "ids": ids,
            "add": add_flags, "chat_template": template, "notes": notes}


# ---------------------------------------------------------------------------------------------
# KVs
# ---------------------------------------------------------------------------------------------
def sha_items(items):
    """Array hash scheme of gguf_range_header.py: sha256(json.dumps(items, ensure_ascii=False))."""
    return hashlib.sha256(json.dumps(list(items), ensure_ascii=False).encode()).hexdigest()


_TRUNC = re.compile(r"^(.*)\.\.\.<(\d+) chars>$", re.S)


def compare_kv(h, typ, val):
    """Compare a value against the official header entry. Returns (equal: bool, detail: str)."""
    if typ.startswith("arr["):
        ok = h.get("len") == len(val) and h.get("sha256") == sha_items(val)
        return ok, f"len {len(val)} vs {h.get('len')}, sha256 {'equal' if ok else 'DIFFERENT'}"
    hv = h["value"]
    if typ == "str":
        m = _TRUNC.match(hv) if isinstance(hv, str) else None
        if m:  # the header dump keeps only 120 chars of long strings
            ok = val.startswith(m.group(1)) and len(val) == int(m.group(2))
            return ok, f"prefix({len(m.group(1))}) and length {len(val)} vs {m.group(2)} (official value truncated in header dump)"
        return val == hv, f"{val!r} vs {hv!r}"
    if typ == "f32":
        ok = struct.pack("<f", val) == struct.pack("<f", hv)
        return ok, f"{val!r} vs {hv!r}"
    return val == hv, f"{val!r} vs {hv!r}"


def build_kvs(header, snapshot_dir, tc, name, signs, report):
    tok = derive_tokenizer(snapshot_dir, tc["vocab_size"])
    rp = tc["rope_parameters"]
    sections = list(rp["mrope_section"]) + [0] * (4 - len(rp["mrope_section"]))
    arch = {"qwen3_5_text": "qwen35"}[tc["model_type"]]
    wn = hadamard_weight_names(tc)
    d = {
        "general.architecture": ("str", arch),
        "general.name": ("str", name),
        f"{arch}.block_count": ("u32", tc["num_hidden_layers"]),  # no MTP block (Prism: --no-mtp)
        f"{arch}.context_length": ("u32", tc["max_position_embeddings"]),
        f"{arch}.embedding_length": ("u32", tc["hidden_size"]),
        f"{arch}.feed_forward_length": ("u32", tc["intermediate_size"]),
        f"{arch}.attention.head_count": ("u32", tc["num_attention_heads"]),
        f"{arch}.attention.head_count_kv": ("u32", tc["num_key_value_heads"]),
        f"{arch}.rope.dimension_sections": ("arr[i32]", sections),
        f"{arch}.rope.freq_base": ("f32", float(rp["rope_theta"])),
        f"{arch}.attention.layer_norm_rms_epsilon": ("f32", float(tc["rms_norm_eps"])),
        f"{arch}.attention.key_length": ("u32", tc["head_dim"]),
        f"{arch}.attention.value_length": ("u32", tc["head_dim"]),
        f"{arch}.ssm.conv_kernel": ("u32", tc["linear_conv_kernel_dim"]),
        f"{arch}.ssm.state_size": ("u32", tc["linear_key_head_dim"]),
        f"{arch}.ssm.group_count": ("u32", tc["linear_num_key_heads"]),
        f"{arch}.ssm.time_step_rank": ("u32", tc["linear_num_value_heads"]),
        f"{arch}.ssm.inner_size": ("u32", tc["linear_value_head_dim"] * tc["linear_num_value_heads"]),
        f"{arch}.full_attention_interval": ("u32", tc["full_attention_interval"]),
        f"{arch}.rope.dimension_count": ("u32", int(tc["head_dim"] * rp["partial_rotary_factor"])),
        "prism.hadamard.version": ("u32", 1),
        "prism.hadamard.block_size": ("u32", HADAMARD_BLOCK),
        "prism.hadamard.transform": ("str", "normalized-sylvester-walsh-hadamard"),
        "prism.hadamard.axis": ("str", "input-last-dimension"),
        "prism.hadamard.sign_mode": ("str", "explicit"),
        "prism.hadamard.weight_names": ("arr[str]", wn),
        "prism.hadamard.sign_widths": ("arr[i32]", [4096, 12288]),
        "prism.hadamard.sign_values": ("arr[i32]", [int(v) for v in signs[4096].tolist()] + [int(v) for v in signs[12288].tolist()]),
        "prism.hadamard.inverse_weight_names": ("arr[str]", ["token_embd.weight"]),
        "prism.hadamard.gdn_v_grouped": ("bool", True),
        "tokenizer.ggml.model": ("str", "gpt2"),
        "tokenizer.ggml.pre": ("str", tok["pre"]),
        "tokenizer.ggml.tokens": ("arr[str]", tok["tokens"]),
        "tokenizer.ggml.token_type": ("arr[i32]", tok["token_type"]),
        "tokenizer.ggml.merges": ("arr[str]", tok["merges"]),
        "tokenizer.chat_template": ("str", tok["chat_template"]),
        "general.quantization_version": ("u32", 2),
        "general.file_type": ("u32", FTYPE_MOSTLY_PTQ1_0),
    }
    kv_name = {"bos": "bos_token_id", "eos": "eos_token_id", "unk": "unknown_token_id", "sep": "seperator_token_id",
               "pad": "padding_token_id", "cls": "cls_token_id", "mask": "mask_token_id"}
    for typ, tid in tok["ids"].items():
        d[f"tokenizer.ggml.{kv_name[typ]}"] = ("u32", tid)
    for typ, flag in tok["add"].items():
        d[f"tokenizer.ggml.add_{typ}_token"] = ("bool", flag)
    for n in tok["notes"]:
        report.append(("note", "tokenizer", n))
    hkeys = [h["key"] for h in header["kv"]]
    extra = sorted(set(d) - set(hkeys))
    if extra:
        raise SystemExit(f"derived KVs not present in the official header: {extra}")
    kvs = []
    for h in header["kv"]:
        key, typ = h["key"], h["type"]
        if key in d:
            dt, val = d[key]
            if dt != typ:
                raise SystemExit(f"KV {key}: derived type {dt} != official {typ}")
            ok, detail = compare_kv(h, typ, val)
            status = "match" if ok else ("differs (by design)" if key in BY_DESIGN else "MISMATCH")
            report.append((status, key, detail))
        else:
            if "value" not in h or (typ == "str" and _TRUNC.match(h["value"])):
                raise SystemExit(f"KV {key} is neither derivable nor fully present in the header dump")
            val = h["value"]
            if key in COPY_FROM_HEADER:
                report.append(("copied", key, f"{val!r} from the official header (not derivable from the HF snapshot)"))
            elif key == "tokenizer.ggml.bos_token_id":
                report.append(("copied", key, f"{val!r} from the official header: the converter's SpecialVocab finds no bos "
                               "(tokenizer_config bos_token is null, config.json has no bos_token_id); the official "
                               "file carries it in the trailing post-conversion KV group"))
            else:
                raise SystemExit(f"KV {key} not derived and not whitelisted for copying")
        kvs.append((key, typ, val))
    if "prism.hadamard.weight_names" in header.get("prism_arrays", {}):
        if wn != header["prism_arrays"]["prism.hadamard.weight_names"]:
            raise SystemExit("generated prism.hadamard.weight_names differ from the official list")
    return kvs, tok


# ---------------------------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------------------------
def _enc_str(s):
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _enc_value(typ, val):
    if typ == "str":
        return _enc_str(val)
    if typ.startswith("arr["):
        et = typ[4:-1]
        head = struct.pack("<IQ", VT[et], len(val))
        if et == "str":
            return head + b"".join(_enc_str(s) for s in val)
        return head + np.asarray(val, dtype=NP_ELEM[et]).tobytes()
    return struct.pack(SCALAR_FMT[typ], val)


def encode_kv(key, typ, val):
    vt = VT["arr"] if typ.startswith("arr[") else VT[typ]
    return _enc_str(key) + struct.pack("<I", vt) + _enc_value(typ, val)


class TensorProducer:
    def __init__(self, src, snap, chunk_rows, use_e2e_norms):
        self.src, self.snap, self.chunk_rows, self.use_e2e_norms = src, snap, chunk_rows, use_e2e_norms
        self.stats = {"neg_scale_groups": 0, "zero_scale_groups": 0, "zero_scale_codes_cleared": 0,
                      "ternary_weights": 0, "e2e_norm_overrides": []}

    def ternary_chunks(self, t):
        in_f, out_f = t.ne
        codes, scales = self.src.codes(t.key), self.src.scales(t.key)
        step = max(1, self.chunk_rows * HADAMARD_BLOCK // in_f)  # ~chunk_rows x 4096 codes per chunk
        for r0 in range(0, out_f, step):
            r1 = min(out_f, r0 + step)
            if t.perm is None:
                c = codes[r0:r1].numpy().copy()
                s = scales[r0:r1].numpy().view(np.uint16).copy()
            else:
                idx = torch.from_numpy(t.perm[r0:r1].astype(np.int64))
                c = codes.index_select(0, idx).numpy()
                s = scales.index_select(0, idx).numpy().view(np.uint16).copy()
            if c.min() < -1 or c.max() > 1:
                raise SystemExit(f"{t.key}: codes outside {{-1,0,1}} in rows {r0}..{r1}")
            if ((s & 0x7C00) == 0x7C00).any():
                raise SystemExit(f"{t.key}: non-finite fp16 scale in rows {r0}..{r1}")
            n_neg, n_dead = normalize_scale_signs(c, s)
            self.stats["neg_scale_groups"] += n_neg
            self.stats["zero_scale_codes_cleared"] += n_dead
            self.stats["zero_scale_groups"] += int((s == 0).sum())
            self.stats["ternary_weights"] += c.size
            yield pack_ptq1_0(c, s)

    def f32_array(self, t):
        x = self.snap.get(t.key).to(torch.float32)  # exact for BF16 and F32 sources
        short = t.key[len(HF):]
        if self.use_e2e_norms and t.op in ("plus1", "copy") and short in self.src.e2e_norms:
            nv = self.src.e2e_norms[short].to(torch.float32)
            if tuple(nv.shape) != tuple(x.shape):
                raise SystemExit(f"e2e norm {short}: shape {tuple(nv.shape)} != {tuple(x.shape)}")
            self.stats["e2e_norm_overrides"].append((short, float((nv - x).abs().max())))
            x = nv
        if t.op == "plus1":
            x = x + 1
        elif t.op == "neg_exp":
            x = -torch.exp(x)
        elif t.op == "squeeze":
            x = x.squeeze()
        elif t.op != "copy":
            raise ValueError(t.op)
        if t.perm is not None:
            x = x[torch.from_numpy(t.perm.astype(np.int64))]
        a = np.ascontiguousarray(x.numpy(), dtype="<f4")
        if tuple(reversed(a.shape)) != tuple(t.ne):
            raise SystemExit(f"{t.name}: source shape {a.shape} does not give ne {t.ne}")
        return a

    def chunks(self, t):
        if t.kind == "tern":
            yield from self.ternary_chunks(t)
        else:
            yield self.f32_array(t)


def write_gguf(path, kvs, plan, producer):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    t0 = time.time()
    with open(tmp, "wb", buffering=32 << 20) as f:
        f.write(b"GGUF" + struct.pack("<IQQ", GGUF_VERSION, len(plan), len(kvs)))
        for key, typ, val in kvs:
            f.write(encode_kv(key, typ, val))
        for t in plan:
            f.write(_enc_str(t.name) + struct.pack("<I", len(t.ne)) + struct.pack(f"<{len(t.ne)}Q", *t.ne)
                    + struct.pack("<IQ", t.gtype, t.offset))
        header_end = f.tell()
        f.write(b"\0" * (align_up(header_end) - header_end))
        data_start = f.tell()
        log(f"header: {header_end} B, data starts at {data_start} ({time.time() - t0:.1f}s)")
        for i, t in enumerate(plan):
            assert f.tell() - data_start == t.offset, (t.name, f.tell() - data_start, t.offset)
            n = 0
            for chunk in producer.chunks(t):
                f.write(np.ascontiguousarray(chunk).reshape(-1).view(np.uint8))
                n += chunk.nbytes
            if n != t.nbytes:
                raise SystemExit(f"{t.name}: wrote {n} B, expected {t.nbytes}")
            f.write(b"\0" * (align_up(n) - n))
            if t.gtype == GGML_PTQ1_0 and (t.nbytes > 50_000_000 or i % 40 == 0):
                log(f"  [{i + 1}/{len(plan)}] {t.name} {t.nbytes / 1e6:.1f} MB  t={time.time() - t0:.0f}s")
        f.flush()
        os.fsync(f.fileno())
        end = f.tell()
    os.replace(tmp, path)
    return {"header_end": header_end, "data_start": data_start, "file_size": end, "seconds": time.time() - t0}


# ---------------------------------------------------------------------------------------------
# Independent reader
# ---------------------------------------------------------------------------------------------
class GGUFReader:
    def __init__(self, path):
        self.path = Path(path)
        self.size = self.path.stat().st_size
        want = 16 << 20
        while True:
            with open(self.path, "rb") as f:
                self.buf = f.read(min(want, self.size))
            try:
                self._parse()
                break
            except (struct.error, IndexError, ValueError):
                if want >= self.size:
                    raise
                want *= 4
        self.data = np.memmap(self.path, dtype=np.uint8, mode="r", offset=self.data_start)

    def _u(self, fmt):
        v = struct.unpack_from(fmt, self.buf, self.pos)[0]
        self.pos += struct.calcsize(fmt)
        return v

    def _s(self):
        n = self._u("<Q")
        b = self.buf[self.pos:self.pos + n]
        if len(b) != n:
            raise ValueError("short read")
        self.pos += n
        return b.decode("utf-8")

    def _val(self, vt):
        if vt == VT["str"]:
            return "str", self._s()
        if vt == VT["arr"]:
            et, n = self._u("<I"), self._u("<Q")
            name = VT_NAME[et]
            if name == "str":
                return "arr[str]", [self._s() for _ in range(n)]
            dt = np.dtype(NP_ELEM[name])
            if self.pos + n * dt.itemsize > len(self.buf):
                raise ValueError("short read")
            a = np.frombuffer(self.buf, dtype=dt, count=n, offset=self.pos)
            self.pos += n * dt.itemsize
            return f"arr[{name}]", (a.astype(bool) if name == "bool" else a).tolist()
        name = VT_NAME[vt]
        return name, self._u(SCALAR_FMT[name])

    def _parse(self):
        self.pos = 0
        if self.buf[:4] != b"GGUF":
            raise SystemExit("not a GGUF file")
        self.pos = 4
        self.version, self.n_tensors, self.n_kv = self._u("<I"), self._u("<Q"), self._u("<Q")
        self.kv = []
        for _ in range(self.n_kv):
            key = self._s()
            typ, val = self._val(self._u("<I"))
            self.kv.append((key, typ, val))
        self.kvd = {k: (t, v) for k, t, v in self.kv}
        self.tensors = []
        for _ in range(self.n_tensors):
            name = self._s()
            nd = self._u("<I")
            ne = [self._u("<Q") for _ in range(nd)]
            self.tensors.append({"name": name, "ne": ne, "type": self._u("<I"), "offset": self._u("<Q")})
        self.header_end = self.pos
        self.alignment = self.kvd.get("general.alignment", ("u32", ALIGN))[1]
        self.data_start = align_up(self.header_end, self.alignment)

    @staticmethod
    def nbytes(t):
        n = int(np.prod(t["ne"]))
        if t["type"] == GGML_PTQ1_0:
            return n // 128 * 28
        if t["type"] == GGML_F32:
            return n * 4
        raise ValueError(t["type"])

    def raw(self, t, r0=0, r1=None):
        rb = self.nbytes(t) // int(np.prod(t["ne"][1:]) or 1)
        rows = int(np.prod(t["ne"][1:])) if len(t["ne"]) > 1 else 1
        r1 = rows if r1 is None else r1
        return self.data[t["offset"] + r0 * rb: t["offset"] + r1 * rb].reshape(r1 - r0, rb)


# ---------------------------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------------------------
# GGUF tensor suffix -> (source, HF sub-name or state sub, f32 op, verify permutation kind)
V_TERN = {"attn_qkv": ("linear_attn.in_proj_qkv", "qkv"), "attn_gate": ("linear_attn.in_proj_z", "v"),
          "ssm_out": ("linear_attn.out_proj", None), "ffn_down": ("mlp.down_proj", None),
          "ffn_gate": ("mlp.gate_proj", None), "ffn_up": ("mlp.up_proj", None), "attn_q": ("self_attn.q_proj", None),
          "attn_k": ("self_attn.k_proj", None), "attn_v": ("self_attn.v_proj", None),
          "attn_output": ("self_attn.o_proj", None)}
V_F32 = {"attn_norm.weight": ("input_layernorm.weight", "plus1", None),
         "post_attention_norm.weight": ("post_attention_layernorm.weight", "plus1", None),
         "attn_q_norm.weight": ("self_attn.q_norm.weight", "plus1", None),
         "attn_k_norm.weight": ("self_attn.k_norm.weight", "plus1", None),
         "ssm_norm.weight": ("linear_attn.norm.weight", "copy", None),
         "ssm_a": ("linear_attn.A_log", "neg_exp", "head"),
         "ssm_alpha.weight": ("linear_attn.in_proj_a.weight", "copy", "head"),
         "ssm_beta.weight": ("linear_attn.in_proj_b.weight", "copy", "head"),
         "ssm_dt.bias": ("linear_attn.dt_bias", "copy", "head"),
         "ssm_conv1d.weight": ("linear_attn.conv1d.weight", "conv", "qkv")}


class Verifier:
    def __init__(self, path, header, tc, src, snap, tokens_json, name, use_e2e_norms, func_layers, func_rows, func_vecs):
        self.r = GGUFReader(path)
        self.header, self.tc, self.src, self.snap = header, tc, src, snap
        self.tokens_json, self.name, self.use_e2e_norms = tokens_json, name, use_e2e_norms
        self.func_layers, self.func_rows, self.func_vecs = func_layers, func_rows, func_vecs
        self.results = []
        self.n_k, self.n_v = tc["linear_num_key_heads"], tc["linear_num_value_heads"]
        self.dv, self.qk = tc["linear_value_head_dim"], 2 * tc["linear_num_key_heads"] * tc["linear_key_head_dim"]

    def check(self, ok, what, detail=""):
        self.results.append({"ok": bool(ok), "check": what, "detail": detail})
        log(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f": {detail}" if detail else ""))
        return ok

    # independent permutation: GGUF index -> source (HF) index
    def src_index(self, kind, n):
        i = np.arange(n)
        if kind is None:
            return i
        if kind == "head":
            hd, base = 1, 0
        elif kind == "v":
            hd, base = self.dv, 0
        elif kind == "qkv":
            hd, base = self.dv, self.qk
        else:
            raise ValueError(kind)
        out = i.copy()
        j = i[base:] - base
        t, e = j // hd, j % hd
        r, k = t // self.n_k, t % self.n_k
        out[base:] = base + (k * (self.n_v // self.n_k) + r) * hd + e
        return out

    def tern_source(self, name):
        if name == "output.weight":
            return "lm_head", None
        if name == "token_embd.weight":
            return "embed_tokens", None
        m = re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", name)
        sub, kind = V_TERN[m.group(2)]
        return f"layers.{m.group(1)}.{sub}", kind

    # ---- structure
    def structure(self):
        r, h = self.r, self.header
        log("structure:")
        self.check(r.version == 3 and r.alignment == 32, "GGUF v3, alignment 32", f"version {r.version}, alignment {r.alignment}")
        self.check(r.n_kv == h["n_kv"] and r.n_tensors == h["n_tensors"], "KV and tensor counts",
                   f"{r.n_kv} KVs / {r.n_tensors} tensors vs {h['n_kv']} / {h['n_tensors']}")
        keys, hkeys = [k for k, _, _ in r.kv], [x["key"] for x in h["kv"]]
        self.check(keys == hkeys, "KV keys and order identical to official")
        types, htypes = [t for _, t, _ in r.kv], [x["type"] for x in h["kv"]]
        bad = [k for k, a, b in zip(keys, types, htypes) if a != b]
        self.check(not bad, "KV types identical to official", f"different: {bad}" if bad else "")
        ht = h["tensors"]
        same = [(a["name"], a["type"], list(a["ne"])) == (b["name"], b["type"], list(b["ne"])) for a, b in zip(r.tensors, ht)]
        self.check(len(r.tensors) == len(ht) and all(same), "tensor names, order, types and ne identical to official",
                   "" if all(same) else f"first difference at {same.index(False)}: {r.tensors[same.index(False)]} vs {ht[same.index(False)]}")
        offs = [a["offset"] == b["offset"] for a, b in zip(r.tensors, ht)]
        self.check(all(offs), "tensor data offsets identical to official")
        last = r.tensors[-1]
        expect = r.data_start + last["offset"] + r.nbytes(last)
        self.check(r.size == expect, "file size = data start + tensor data (no trailing bytes)", f"{r.size} vs {expect}")
        self.check(r.data_start == h["data_start"], "data section starts where the official one does",
                   f"header_end {r.header_end} (official {h['header_end']}), data_start {r.data_start} (official {h['data_start']})")
        diff = r.size - OFFICIAL_SIZE
        self.check(diff == 0 or abs(diff) <= 4096, "file size vs official 1,997,486,976 B", f"{r.size} B ({diff:+d})")
        n_tern = sum(t["type"] == GGML_PTQ1_0 for t in r.tensors)
        n_f32 = sum(t["type"] == GGML_F32 for t in r.tensors)
        self.check(n_tern == 202 and n_f32 == 225, "202 PTQ1_0 + 225 F32 tensors", f"{n_tern} + {n_f32}")

    # ---- KV values
    def kv_values(self):
        log("KV values:")
        r = self.r
        signs = self.signs
        sv_expect = [int(v) for v in signs[4096].tolist()] + [int(v) for v in signs[12288].tolist()]
        for h in self.header["kv"]:
            key = h["key"]
            typ, val = r.kvd[key]
            if key == "general.name":
                self.check(val == self.name, key, f"{val!r} (official {h['value']!r}, differs by design)")
            elif key == "prism.hadamard.sign_values":
                off = self.header["prism_arrays"][key]
                self.check(val == sv_expect, key, f"equals the state's s4096+s12288 ({len(val)} values); "
                           f"{'EQUALS' if val == off else 'differs from'} Prism's official vectors "
                           f"({sum(a != b for a, b in zip(val, off))} entries differ)")
            else:
                ok, detail = compare_kv(h, typ, val)
                self.check(ok, key, detail if (not ok or typ.startswith("arr[") or typ == "str" and len(val) > 64) else "")
        # loader-rule consistency (llama-model.cpp:1206-1372)
        wn = r.kvd["prism.hadamard.weight_names"][1]
        inv = r.kvd["prism.hadamard.inverse_weight_names"][1]
        widths = r.kvd["prism.hadamard.sign_widths"][1]
        vals = r.kvd["prism.hadamard.sign_values"][1]
        tmap = {t["name"]: t for t in r.tensors}
        folded = [t["name"] for t in r.tensors if t["type"] == GGML_PTQ1_0]
        self.check(len(set(wn)) == len(wn) == 201 and set(wn) | set(inv) == set(folded) and not set(wn) & set(inv),
                   "weight_names (201) + inverse_weight_names (1) cover exactly the 202 PTQ1_0 tensors")
        self.check(wn == hadamard_weight_names(self.tc), "weight_names in sorted-HF-name order")
        self.check(sum(widths) == len(vals) and set(vals) <= {-1, 1} and
                   all(tmap[n]["ne"][0] in widths for n in wn + inv), "sign table covers every folded input width",
                   f"widths {widths}, {len(vals)} values")
        self.check(r.kvd["general.file_type"][1] == FTYPE_MOSTLY_PTQ1_0, "general.file_type = 143 (LLAMA_FTYPE_MOSTLY_PTQ1_0)")
        # tokenizer details
        toks = r.kvd["tokenizer.ggml.tokens"][1]
        if self.tokens_json and Path(self.tokens_json).is_file():
            p = Path(self.tokens_json)
            off = json.loads((gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()).decode("utf-8"))
            first = next((i for i, (a, b) in enumerate(zip(toks, off)) if a != b), None)
            self.check(len(toks) == len(off) and first is None, "tokenizer.ggml.tokens == official tokens.json exactly",
                       f"{len(toks)} tokens" + (f"; first difference at id {first}: {toks[first]!r} vs {off[first]!r}" if first is not None else ""))
        else:
            self.check(True, "tokens.json not available; tokens checked by sha256 only")
        tt = r.kvd["tokenizer.ggml.token_type"][1]
        counts = {k: tt.count(k) for k in sorted(set(tt))}
        self.check(True, "token_type histogram", json.dumps(counts))
        jinja = Path(self.snap.root) / "chat_template.jinja"
        if jinja.is_file():
            self.check(r.kvd["tokenizer.chat_template"][1] == jinja.read_text(encoding="utf-8"),
                       "chat_template == snapshot chat_template.jinja")

    # ---- content
    def ptq_content(self, chunk_rows):
        log("PTQ1_0 content (every tensor, every block):")
        t0 = time.time()
        bad_tensors, n_w, n_neg, n_negzero, n_zero, n_dead = [], 0, 0, 0, 0, 0
        e2e_used = 0
        for t in self.r.tensors:
            if t["type"] != GGML_PTQ1_0:
                continue
            key, kind = self.tern_source(t["name"])
            in_f, out_f = t["ne"]
            idx = self.src_index(kind, out_f)
            codes = self.src.codes(key)
            scales = self.src.e2e_scales[key] if key in self.src.e2e_scales else self.src.entry(key)["scales"]
            e2e_used += key in self.src.e2e_scales
            for r0 in range(0, out_f, chunk_rows):
                r1 = min(out_f, r0 + chunk_rows)
                gc, gs = decode_ptq1_0(np.asarray(self.r.raw(t, r0, r1)), in_f)
                ii = torch.from_numpy(idx[r0:r1].astype(np.int64))
                sc = codes.index_select(0, ii).numpy()
                ss = scales.index_select(0, ii).numpy().view(np.uint16)
                neg = ss >= 0x8000
                n_neg += int(neg.sum())
                n_negzero += int((ss == 0x8000).sum())
                exp_c = np.where(np.repeat(neg, QK, axis=1), -sc, sc)
                exp_s = np.where(neg, ss - 0x8000, ss).astype(np.uint16)
                zero = exp_s == 0
                n_zero += int(zero.sum())
                n_dead += int((zero & sc.reshape(r1 - r0, -1, QK).any(-1)).sum())
                exp_c = np.where(np.repeat(zero, QK, axis=1), np.int8(0), exp_c)  # dead codes are written as 0
                if not (np.array_equal(gc, exp_c) and np.array_equal(gs, exp_s)):
                    rr, cc = np.argwhere(np.repeat(gs != exp_s, QK, axis=1) | (gc != exp_c))[0]
                    bad_tensors.append(f"{t['name']} row {r0 + rr} col {cc}")
                    break
                n_w += gc.size
        self.check(not bad_tensors, "decoded codes and fp16 scale bits == permuted, sign-normalized source (exact)",
                   f"{n_w:,} weights in {time.time() - t0:.0f}s; {n_neg} negative-scale groups normalized "
                   f"({n_negzero} were -0.0), {n_zero} zero-scale groups ({n_dead} with nonzero source codes, "
                   f"written as 0); e2e scales used for {e2e_used} tensors"
                   + (f"; MISMATCH in {bad_tensors[:3]}" if bad_tensors else ""))

    def f32_content(self):
        log("F32 content:")
        bad, n, maxd_exp = [], 0, 0.0
        for t in self.r.tensors:
            if t["type"] != GGML_F32:
                continue
            got = np.asarray(self.r.raw(t)).reshape(-1).view("<f4")
            if t["name"] == "output_norm.weight":
                hf, op, kind = HF + "norm.weight", "plus1", None
            else:
                m = re.fullmatch(r"blk\.(\d+)\.(.+)", t["name"])
                sub, op, kind = V_F32[m.group(2)]
                hf = f"{HF}layers.{m.group(1)}.{sub}"
            x = self.snap.get(hf).float()
            short = hf[len(HF):]
            if self.use_e2e_norms and op in ("plus1", "copy") and short in self.src.e2e_norms:
                x = self.src.e2e_norms[short].float()
            a = x.numpy()
            if op == "plus1":
                a = a + np.float32(1)
            elif op == "neg_exp":
                ref = -torch.exp(x).numpy()  # the converter's torch op
                maxd_exp = max(maxd_exp, float(np.abs(ref.astype(np.float64) + np.exp(a.astype(np.float64))).max()))
                a = ref
            elif op == "conv":
                a = a.reshape(a.shape[0], -1)
            a = a[self.src_index(kind, a.shape[0])]
            exp = np.ascontiguousarray(a, dtype="<f4").reshape(-1)
            if got.shape != exp.shape or not np.array_equal(got.view(np.uint32), exp.view(np.uint32)):
                bad.append(t["name"])
            n += 1
        self.check(not bad, "F32 tensors bit-identical to transformed snapshot tensors",
                   f"{n} tensors; ssm_a vs fp64 -exp max |diff| {maxd_exp:.2e}" + (f"; MISMATCH {bad[:5]}" if bad else ""))

    # ---- function
    def functional(self):
        log("functional equivalence (fp64):")
        rng = np.random.default_rng(1234)
        hp, hs = hadamard_popcount(HADAMARD_BLOCK), hadamard_sylvester(HADAMARD_BLOCK)
        self.check(np.array_equal(hp, hs), "runtime popcount Hadamard == Sylvester recursion (4096)")
        vals = self.r.kvd["prism.hadamard.sign_values"][1]
        kv_signs = {4096: np.array(vals[:4096], np.float64), 12288: np.array(vals[4096:], np.float64)}
        B = HADAMARD_BLOCK
        names = ["output.weight", "token_embd.weight"] + [
            t["name"] for t in self.r.tensors if t["type"] == GGML_PTQ1_0 and t["name"].startswith("blk.")
            and int(t["name"].split(".")[1]) in self.func_layers]
        tmap = {t["name"]: t for t in self.r.tensors}
        worst = 0.0
        for name in names:
            t = tmap[name]
            key, kind = self.tern_source(name)
            in_f, out_f = t["ne"]
            idx = self.src_index(kind, out_f)
            k = self.func_rows
            if kind is not None:
                moved = np.nonzero(idx != np.arange(out_f))[0]
                rows = np.concatenate([rng.choice(moved, k // 2, replace=False),
                                       rng.choice(np.setdiff1d(np.arange(out_f), moved), k - k // 2, replace=False)
                                       if len(moved) < out_f else rng.choice(moved, k - k // 2, replace=False)])
            else:
                rows = rng.choice(out_f, k, replace=False)
            rows = np.sort(rows)
            # GGUF side: decode, then the runtime's activation transform H(s*x) with the file's own signs
            gc, gs = decode_ptq1_0(np.stack([np.asarray(self.r.raw(t, int(i), int(i) + 1))[0] for i in rows]), in_f)
            wg = (gc.reshape(k, -1, QK).astype(np.float64) * gs.view(np.float16).astype(np.float64)[..., None]).reshape(k, in_f)
            s_kv = kv_signs[in_f]
            # source side: fake-quant (C*s) R^T in HF row order, R = diag(s) blockdiag(H), from the state as stored
            src_rows = torch.from_numpy(idx[rows].astype(np.int64))
            c = self.src.codes(key).index_select(0, src_rows).numpy().astype(np.float64)
            sc_t = self.src.e2e_scales[key] if key in self.src.e2e_scales else self.src.entry(key)["scales"]
            s = sc_t.index_select(0, src_rows).float().numpy().astype(np.float64)
            wq = (c.reshape(k, -1, QK) * s[..., None]).reshape(k, in_f)
            s_st = self.src.signs(key, in_f).numpy().astype(np.float64)
            w_ref = np.concatenate([wq[:, b:b + B] @ hs.T for b in range(0, in_f, B)], 1) * s_st[None, :]
            hf_name = "lm_head.weight" if key == "lm_head" else (HF + "embed_tokens.weight" if key == "embed_tokens"
                                                                 else f"{HF}{key}.weight")
            w_orig = self.snap.rows(hf_name, idx[rows]).double().numpy()
            if name == "token_embd.weight":
                e_g = np.concatenate([wg[:, b:b + B] @ hp.T for b in range(0, in_f, B)], 1) * s_kv[None, :]  # e = s*(H z)
                rel = np.linalg.norm(e_g - w_ref) / np.linalg.norm(w_ref)
                cos = float((e_g * w_orig).sum() / (np.linalg.norm(e_g) * np.linalg.norm(w_orig)))
                detail = f"lookup s*(H z) vs (C*s)R^T rows: rel {rel:.2e}; cos vs original BF16 rows {cos:.3f}"
            else:
                x = rng.standard_normal((in_f, self.func_vecs))
                xr = np.concatenate([hp @ (s_kv[b:b + B, None] * x[b:b + B]) for b in range(0, in_f, B)], 0)
                y_g = wg @ xr                      # runtime: W_gguf @ H(s*x); rows are GGUF (tiled) rows
                y_ref = w_ref @ x                  # fake-quant HF computation on the matching HF rows
                rel = np.linalg.norm(y_g - y_ref) / np.linalg.norm(y_ref)
                y_o = w_orig @ x
                cos = float((y_g * y_o).sum() / (np.linalg.norm(y_g) * np.linalg.norm(y_o)))
                detail = f"rel {rel:.2e}; cos vs original BF16 W x {cos:.3f}"
                if kind is not None:
                    moved_rows = idx[rows] != rows
                    y_n = self.snap.rows(hf_name, rows[moved_rows]).double().numpy() @ x
                    yg_m = y_g[moved_rows]
                    cn = float((yg_m * y_n).sum() / (np.linalg.norm(yg_m) * np.linalg.norm(y_n)))
                    detail += f"; control without un-permuting rows: cos {cn:.3f}"
            worst = max(worst, rel)
            self.check(rel < 1e-5, f"function {name} ({k} rows)", detail)
        return worst

    def run(self, chunk_rows):
        t0 = time.time()
        self.signs = {w: self.src_signs_by_width(w) for w in (4096, 12288)}
        self.structure()
        self.kv_values()
        self.ptq_content(chunk_rows)
        self.f32_content()
        self.functional()
        n_fail = sum(not x["ok"] for x in self.results)
        log(f"verify: {len(self.results) - n_fail} passed, {n_fail} failed in {time.time() - t0:.0f}s")
        return n_fail

    def src_signs_by_width(self, w):
        key = "layers.0.mlp.down_proj" if w == 12288 else "layers.0.mlp.gate_proj"
        return self.src.signs(key, w)


# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", help="ternary_state.pt (gptq_ternary.py --shared-signs)")
    ap.add_argument("--endpoints", help="endpoint_state.pt (endpoints.py)")
    ap.add_argument("--e2e", default="", help="optional e2e_state.pt: trained fp16 scales override GPTQ scales")
    ap.add_argument("--e2e-norms", action="store_true", help="also take RMSNorm weights from e2e_state.pt['norms']")
    ap.add_argument("--ignore-e2e-norms", action="store_true",
                    help="export the snapshot norms even though e2e_state.pt['norms'] differ from them")
    ap.add_argument("--snapshot", default="D:/openbonsai/runs/hf-cache/hub/models--Qwen--Qwen3.5-9B/snapshots/"
                                          "c202236235762e1c871ad0ccb60c8ee5ba337b9a")
    ap.add_argument("--out", help="output .gguf")
    ap.add_argument("--name", default=DEFAULT_NAME, help="general.name")
    ap.add_argument("--header", default=str(DEFAULT_HEADER), help="official header dump (KV order/types, tensor infos)")
    ap.add_argument("--tokens-json", default=str(DEFAULT_TOKENS),
                    help="official tokenizer.ggml.tokens dump, .json or .json.gz (optional; else sha256 check only)")
    ap.add_argument("--chunk-rows", type=int, default=16384)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--verify", action="store_true", help="verify after writing")
    ap.add_argument("--verify-only", action="store_true", help="verify an existing --out file")
    ap.add_argument("--func-layers", default="0,3,30,31")
    ap.add_argument("--func-rows", type=int, default=16)
    ap.add_argument("--func-vecs", type=int, default=4)
    ap.add_argument("--selftest", action="store_true", help="codec selftest only")
    ap.add_argument("--report", default="", help="JSON report path (default: <out>.export.json)")
    args = ap.parse_args()
    assert torch.cuda.device_count() == 0, "CPU only: CUDA devices must be hidden"
    torch.set_num_threads(args.threads)
    log(selftest())
    if args.selftest:
        return 0
    for req in ("state", "endpoints", "out"):
        if not getattr(args, req):
            ap.error(f"--{req} is required")
    header = json.loads(Path(args.header).read_text(encoding="utf-8"))
    cfg = json.loads((Path(args.snapshot) / "config.json").read_text(encoding="utf-8"))
    tc = cfg["text_config"]
    if cfg.get("tie_word_embeddings"):
        raise SystemExit("tied embeddings need prism.hadamard version 2; this exporter writes version 1")
    plan = build_plan(tc)
    hp = [(t["name"], t["type"], list(t["ne"]), t["offset"]) for t in header["tensors"]]
    mine = [(t.name, t.gtype, list(t.ne), t.offset) for t in plan]
    if mine != hp:
        first = next(i for i, (a, b) in enumerate(zip(mine, hp + [None] * len(mine))) if a != b)
        raise SystemExit(f"tensor plan differs from the official header at {first}: {mine[first]} vs {hp[first] if first < len(hp) else None}")
    log(f"tensor plan: {len(plan)} tensors identical to the official names/order/types/ne/offsets")
    src = TernarySources(args.state, args.endpoints, args.e2e or None)
    snap = Snapshot(args.snapshot)
    report = {"args": vars(args), "kv": [], "notes": []}
    signs, notes = check_inputs(plan, src)
    notes += check_e2e_norms(plan, src, snap, args.e2e_norms, args.ignore_e2e_norms)
    report["notes"] += notes
    for n in notes:
        log("input: " + n)
    out = Path(args.out)
    rep_path = Path(args.report) if args.report else out.with_name(out.name + ".export.json")
    if not args.verify_only:
        t0 = time.time()
        kv_report = []
        kvs, _ = build_kvs(header, args.snapshot, tc, args.name, signs, kv_report)
        log(f"KVs: {len(kvs)} in official order ({time.time() - t0:.1f}s)")
        for status, key, detail in kv_report:
            if status != "match":
                log(f"  [{status}] {key}: {detail}")
        mism = [k for s, k, _ in kv_report if s == "MISMATCH"]
        log(f"  {sum(s == 'match' for s, _, _ in kv_report)} KVs match the official values; "
            f"MISMATCH: {mism if mism else 'none'}")
        report["kv"] = [{"status": s, "key": k, "detail": d} for s, k, d in kv_report]
        out.parent.mkdir(parents=True, exist_ok=True)
        producer = TensorProducer(src, snap, args.chunk_rows, args.e2e_norms)
        info = write_gguf(out, kvs, plan, producer)
        info.update(producer.stats)
        info["size_vs_official"] = info["file_size"] - OFFICIAL_SIZE
        report["write"] = info
        log(f"wrote {out} : {info['file_size']:,} B ({info['size_vs_official']:+d} vs official) in {info['seconds']:.0f}s; "
            f"{info['ternary_weights']:,} ternary weights, {info['neg_scale_groups']} negative-scale groups normalized, "
            f"{info['zero_scale_groups']} zero-scale groups ({info['zero_scale_codes_cleared']} had nonzero codes, cleared)")
        rep_path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    n_fail = 0
    if args.verify or args.verify_only:
        log(f"verifying {out}")
        v = Verifier(out, header, tc, src, snap, args.tokens_json, args.name, args.e2e_norms,
                     [int(x) for x in args.func_layers.split(",") if x != ""], args.func_rows, args.func_vecs)
        n_fail = v.run(args.chunk_rows)
        report["verify"] = v.results
        rep_path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    log(f"report: {rep_path}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
