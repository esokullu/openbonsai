"""Export a ternary state to Prism's PTQ1_0 GGUF (runs in Prism's llama.cpp fork).

The run's final state (quantize codes, ternary endpoints, distilled scales,
and the distilled high-precision tensors when ``distill.train.small``) is
written in the adapter's layout (``adapter.gguf_tensors()`` /
``adapter.gguf_kvs()``); nothing here knows an architecture. Port of
research/experiments/export_gguf.py, which wrote a 9B file Prism's CPU runtime
loads; for the 9B config the bytes are identical (tests/test_export_parity.py).

- PTQ1_0 tensors: codes and fp16 scales rows permuted by the layout, groups
  canonicalized (formats.ptq1_0.canonicalize), packed in row chunks.
- F32 tensors: the checkpoint's tensor, or the distilled one, transformed
  (``+1``, ``-exp``, squeeze) and permuted in FP32.
- Refused: a state without ternary ``lm_head``/``embed_tokens`` (the layout
  stores both as PTQ1_0), a state with per-module signs (the file holds one
  sign vector per input width), group sizes other than 128, codes outside
  {-1, 0, 1}, non-finite scales, distilled tensors without a slot, and
  checkpoint tensors of the text model without a slot (e.g. biases).
- When the config is a published file's (``adapter.gguf_reference()``, the
  9B), the tensor layout and the runtime KVs (architecture, hyperparameters,
  ``prism.hadamard.*`` except the sign values, file type) must equal it, by
  key; tokenizer and descriptive KVs may differ (a fine-tune's own).
- Written to ``<out>.tmp`` (removed if writing fails), fsynced, verified
  (re-read: structure and KVs, every PTQ1_0 block decoded against the state,
  every F32 tensor bit for bit, and a FP64 function check of sample rows
  through the runtime's own transform), then renamed. The content checks
  take rows and F32 transforms from the adapter's independent rule
  (``gguf_source_rows`` / ``gguf_f32_op``), so a wrong permutation in the
  layout fails. A report goes to ``<out>.export.json``.
"""
from __future__ import annotations

import os
from pathlib import Path
import struct
import time

import numpy as np
import torch

from ..adapters.gguf_common import model_basename, reference_kv_equal
from ..formats import ptq1_0
from ..formats.atomic import atomic_write_json
from ..formats.checkpoint import Checkpoint
from ..formats.gguf import GGML_PTQ1_0, GGUFReader, f32_forward, layout, prism_signs, write_gguf
from ..formats.state import LegacyTensor, TernaryState, legacy_to_state

ENDPOINTS = ("lm_head", "embed_tokens")
FUNC_TOL = 1e-6
EXPORT_NEEDS_DISTILLED_ENDPOINTS = (
    "export writes the distilled state, but distill.endpoints is false: the distilled model keeps BF16 "
    "endpoints, and the PTQ1_0 GGUF stores the LM head and the token embedding as ternary. Set "
    "distill.endpoints=true, or export.enabled=false")


class ExportError(RuntimeError):
    """The state cannot be expressed in the GGUF layout, or verification failed."""


def default_name(model_path) -> str:
    return f"OpenBonsai-{model_basename(model_path)}-PTQ1_0"


# -- inputs ------------------------------------------------------------------

def as_state(source) -> TernaryState:
    """A TernaryState, or a research ``load_legacy_pt`` dict (shared signs only)."""
    if isinstance(source, TernaryState):
        return source.validate()
    if isinstance(source, dict) and source and all(isinstance(v, LegacyTensor) for v in source.values()):
        try:
            return legacy_to_state(source)
        except ValueError as error:
            raise ExportError(f"{error}. Prism's GGUF holds exactly one sign vector per input width: "
                              "re-quantize with shared signs (rotation.signs, the default)") from error
    raise ExportError(f"cannot export a {type(source).__name__}; expected a TernaryState")


def check_inputs(plan, state: TernaryState, small: dict) -> list[str]:
    """Refuse what the layout cannot hold. Returns notes."""
    notes = []
    tern = [t for t in plan if t.ternary]
    keys = {t.source for t in tern}
    missing_ep = [k for k in ENDPOINTS if k in keys and k not in state.tensors]
    if missing_ep:
        raise ExportError(f"the state has no ternary {' and '.join(missing_ep)}: Prism's PTQ1_0 layout stores the "
                          "LM head and the token embedding as PTQ1_0. Run the endpoints stage with "
                          "endpoints.head: ternary and endpoints.embedding: ternary, and keep "
                          "distill.endpoints: true when the distilled state is exported")
    missing = sorted(keys - set(state.tensors))
    if missing:
        raise ExportError(f"the state lacks {len(missing)} ternary tensors of the layout, e.g. {missing[:4]}")
    extra = sorted(set(state.tensors) - keys)
    if extra:
        raise ExportError(f"state tensors with no PTQ1_0 slot in the layout: {extra[:4]}")
    if state.group != ptq1_0.QK:
        raise ExportError(f"PTQ1_0 groups are {ptq1_0.QK} weights; the state uses group {state.group}")
    for t in tern:
        in_f, out_f = t.ne
        tt = state.tensors[t.source]
        if tuple(tt.codes.shape) != (out_f, in_f):
            raise ExportError(f"{t.source}: codes {tuple(tt.codes.shape)}, the layout expects {(out_f, in_f)}")
        if in_f not in state.signs:
            raise ExportError(f"{t.source}: no sign vector for input width {in_f}")
    floats = {t.source: t for t in plan if not t.ternary}
    unknown = sorted(set(small) - set(floats))
    if unknown:
        raise ExportError(f"distilled high-precision tensors with no F32 slot in the layout: {unknown[:4]}")
    for key, v in small.items():
        if tuple(v.shape) != tuple(floats[key].shape):
            raise ExportError(f"distilled {key}: shape {tuple(v.shape)} != {floats[key].shape}")
    if small:
        notes.append(f"distilled high-precision tensors: {len(small)} of {len(floats)} F32 tensors")
    if state.block != 4096:
        notes.append(f"Hadamard block {state.block}: Prism's runtime was verified with block 4096 only")
    return notes


def check_coverage(adapter, plan, ckpt) -> None:
    """Every checkpoint tensor of the text model and the head has a slot in the layout (a bias, say,
    would otherwise be dropped without a word). MTP and vision tensors are outside the text model."""
    prefix = adapter.checkpoint_name("")
    slots = {adapter.checkpoint_name(t.source + ".weight" if t.ternary else t.source) for t in plan}
    extra = sorted(n for n in ckpt.map if (n.startswith(prefix) or n.split(".", 1)[0] == "lm_head")
                   and n not in slots)
    if extra:
        raise ExportError(f"{len(extra)} checkpoint tensors have no slot in the {adapter.gguf_arch} GGUF layout "
                          f"and would be dropped: {extra[:4]}")


class _Producer:
    """Tensor payloads in file order (``write_gguf``'s ``produce``)."""

    def __init__(self, adapter, plan, state, ckpt, small, chunk_rows):
        self.adapter, self.state, self.ckpt, self.small = adapter, state, ckpt, small
        self.maps = {t.name: t for t in plan}
        self.chunk_rows = int(chunk_rows)
        self.stats = {"ternary_weights": 0, "neg_scale_groups": 0, "zero_scale_groups": 0,
                      "zero_scale_codes_cleared": 0, "distilled_f32": 0}

    def source_float(self, t) -> torch.Tensor:
        if t.source in self.small:
            return self.small[t.source]
        return self.ckpt.get(self.adapter.checkpoint_name(t.source))

    def __call__(self, info):
        t = self.maps[info.name]
        if t.ternary:
            yield from self._ternary(t)
        else:
            self.stats["distilled_f32"] += t.source in self.small
            yield self.float_array(t)

    def _ternary(self, t):
        in_f, out_f = t.ne
        tt = self.state.tensors[t.source]
        codes, scales = tt.codes, tt.scales
        step = max(1, self.chunk_rows * 4096 // in_f)   # ~chunk_rows x 4096 codes per chunk
        for r0 in range(0, out_f, step):
            r1 = min(out_f, r0 + step)
            if t.perm is None:
                c = codes[r0:r1].numpy().copy()
                s = scales[r0:r1].numpy().view(np.uint16).copy()
            else:
                idx = torch.from_numpy(t.perm[r0:r1].astype(np.int64))
                c = np.ascontiguousarray(codes.index_select(0, idx).numpy())
                s = scales.index_select(0, idx).numpy().view(np.uint16).copy()
            try:
                ptq1_0.check(c, s)
            except ValueError as error:
                raise ExportError(f"{t.source} rows {r0}..{r1}: {error}") from error
            n_neg, n_dead = ptq1_0.canonicalize(c, s)
            self.stats["neg_scale_groups"] += n_neg
            self.stats["zero_scale_codes_cleared"] += n_dead
            self.stats["zero_scale_groups"] += int((s == 0).sum())
            self.stats["ternary_weights"] += c.size
            yield ptq1_0.pack(c, s)

    def float_array(self, t) -> np.ndarray:
        x = f32_forward(t.op, self.source_float(t))
        if t.perm is not None:
            x = x[torch.from_numpy(t.perm.astype(np.int64))]
        a = np.ascontiguousarray(x.numpy(), dtype="<f4")
        if tuple(reversed(a.shape)) != tuple(t.ne):
            raise ExportError(f"{t.name}: source shape {a.shape} does not give ne {t.ne}")
        return a


# -- verification -------------------------------------------------------------

def _hadamard_popcount(n: int) -> np.ndarray:
    """The runtime's construction: H[i, j] = (-1)^popcount(i & j) / sqrt(n)."""
    a = np.arange(n, dtype=np.uint32)
    p = np.bitwise_and.outer(a, a)
    for s in (16, 8, 4, 2, 1):
        p ^= p >> s
    return np.where(p & 1, -1.0, 1.0) / np.sqrt(n)


def _hadamard_sylvester(n: int) -> np.ndarray:
    """The normalized Sylvester recursion in FP64 (the state's convention, core/hadamard.py)."""
    h = np.ones((1, 1))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def _kv_equal(typ, a, b) -> bool:
    if typ == "f32":
        return struct.pack("<f", a) == struct.pack("<f", b)
    if typ == "arr[f32]":
        return np.array_equal(np.asarray(a, "<f4").view("<u4"), np.asarray(b, "<f4").view("<u4"))
    return a == b


def runtime_kv(key: str, arch: str) -> bool:
    """KVs the runtime builds the model from (must equal the reference file's for its config). The rest
    (general.name and other descriptive general.*, tokenizer.*, the chat template) may differ, e.g. in a
    fine-tune with its own tokenizer settings."""
    return (key.startswith(f"{arch}.") or key.startswith("prism.hadamard.")
            or key in ("general.architecture", "general.file_type", "general.quantization_version"))


def reference_kv_diff(kvs, reference: dict, arch: str) -> tuple[list[str], list[str]]:
    """(runtime differences, other differences) of ``kvs`` [(key, type, value)] against a header dump,
    compared by key. ``prism.hadamard.sign_values`` is compared by type only (the signs are the state's)."""
    ours = {k: (t, v) for k, t, v in kvs}
    theirs = {h["key"]: h for h in reference["kv"]}
    hard, soft = [], []
    for key in sorted(set(ours) | set(theirs)):
        out = hard if runtime_kv(key, arch) else soft
        if key not in ours:
            out.append(f"{key} missing")
            continue
        if key not in theirs:
            out.append(f"{key} not in the reference")
            continue
        typ, val = ours[key]
        h = theirs[key]
        if typ != h["type"]:
            out.append(f"{key} type {typ} vs {h['type']}")
        elif key != "prism.hadamard.sign_values" and ("value" in h or "sha256" in h):
            ok, detail = reference_kv_equal(h, typ, val)
            if not ok:
                out.append(f"{key} ({detail})" if out is hard else key)
    if not soft and [k for k, _, _ in kvs] != [h["key"] for h in reference["kv"]]:
        soft.append("KV order")
    return hard, soft


def _f32_reference(op: str, x: torch.Tensor) -> np.ndarray:
    """The F32 transforms written out again for verification (numpy FP32; -exp through torch.exp, the
    converter's op), not ``formats.gguf.f32_forward``."""
    a = x.to(torch.float32).numpy()
    if op == "plus1":
        return a + np.float32(1)
    if op == "neg_exp":
        return -torch.exp(x.to(torch.float32)).numpy()
    if op == "squeeze":
        return a.reshape(a.shape[0], -1)
    if op == "copy":
        return a
    raise ExportError(f"unknown F32 op {op!r}")


class Verifier:
    def __init__(self, reader, adapter, plan, kvs, state, producer, reference, log):
        self.r, self.adapter, self.plan, self.kvs, self.state = reader, adapter, plan, kvs, state
        self.producer, self.reference, self.log = producer, reference, log
        self.results = []
        # Row maps and F32 transforms the content checks use: the adapter's independent statement when it
        # has one (so a wrong permutation or transform in the layout fails), else the layout's own.
        self.rows, self.ops, self.independent = {}, {}, True
        for t in plan:
            n = t.ne[1] if t.ternary else t.shape[0]
            rows = adapter.gguf_source_rows(t.name, n)
            if rows is None:
                self.independent = False
                rows = np.arange(n) if t.perm is None else np.asarray(t.perm)
            self.rows[t.name] = np.asarray(rows, dtype=np.int64)
            if not t.ternary:
                op = adapter.gguf_f32_op(t.name)
                if op is None:
                    self.independent = False
                    op = t.op
                self.ops[t.name] = op
        self.basis = ("the adapter's independent row and transform rule" if self.independent
                      else "the layout's own permutation (no independent rule: checks the codec and I/O only)")

    def check(self, ok, what, detail=""):
        self.results.append({"ok": bool(ok), "check": what, "detail": detail})
        self.log(f"  {'PASS' if ok else 'FAIL'}  {what}" + (f": {detail}" if detail else ""))
        return ok

    def layout_rule(self):
        """The layout's permutations and F32 transforms against the adapter's independent statement."""
        bad = []
        for t in self.plan:
            n = len(self.rows[t.name])
            perm = np.arange(n) if t.perm is None else np.asarray(t.perm)
            if perm.shape != (n,) or not np.array_equal(perm, self.rows[t.name]):
                bad.append(f"{t.name} rows")
            if not t.ternary and t.op != self.ops[t.name]:
                bad.append(f"{t.name} op {t.op} (rule: {self.ops[t.name]})")
        self.check(not bad, "layout permutations and F32 transforms == the adapter's independent rule",
                   f"{len(self.plan)} tensors" + (f"; DIFFERENT: {bad[:4]}" if bad else ""))

    def structure(self, infos):
        r = self.r
        self.check(r.version == 3 and r.alignment == 32, "GGUF v3, alignment 32", f"{r.version}, {r.alignment}")
        got = [(t.name, t.gtype, tuple(t.ne), t.offset) for t in r.tensors]
        want = [(t.name, t.gtype, tuple(t.ne), t.offset) for t in infos]
        first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None)
        self.check(got == want, "tensor names, order, types, ne and offsets as planned",
                   f"{len(got)} tensors" + (f"; first difference at {first}: {got[first]} vs {want[first]}"
                                            if first is not None else ""))
        last = r.tensors[-1]
        expect = r.data_start + last.offset + last.nbytes
        self.check(r.size == expect, "file size = data start + tensor data", f"{r.size} vs {expect}")
        bad = [k for (k, t, v), (k2, t2, v2) in zip(r.kv, self.kvs) if (k, t) != (k2, t2) or not _kv_equal(t, v, v2)]
        self.check(len(r.kv) == len(self.kvs) and not bad, "KVs read back equal the KVs written",
                   f"{len(r.kv)} KVs" + (f"; different: {bad[:4]}" if bad else ""))
        # Prism's loader rules (llama-model.cpp 1206-1372): every PTQ1_0 tensor folded exactly once.
        wn = r.value("prism.hadamard.weight_names", [])
        inv = r.value("prism.hadamard.inverse_weight_names", [])
        folded = {t.name for t in r.tensors if t.gtype == GGML_PTQ1_0}
        self.check(len(set(wn)) == len(wn) and set(wn) | set(inv) == folded and not set(wn) & set(inv),
                   "weight_names + inverse_weight_names cover each PTQ1_0 tensor once",
                   f"{len(wn)} + {len(inv)} of {len(folded)}")
        signs = prism_signs(r)
        self.check(all(r.info(n).ne[0] in signs for n in folded), "sign table covers every PTQ1_0 input width",
                   f"widths {sorted(signs)}")
        self.check(r.value("general.file_type") == ptq1_0.LLAMA_FTYPE_MOSTLY_PTQ1_0,
                   "general.file_type = 143 (LLAMA_FTYPE_MOSTLY_PTQ1_0)")

    def against_reference(self):
        ref, r = self.reference, self.r
        got = [(t.name, t.gtype, list(t.ne), t.offset) for t in r.tensors]
        want = [(h["name"], h["type"], list(h["ne"]), h["offset"]) for h in ref["tensors"]]
        # Offsets are relative to the data section, whose start moves with the header (name, tokenizer).
        self.check(got == want, "tensor names, order, types, ne and offsets identical to the reference file",
                   f"data section at {r.data_start} (reference {ref['data_start']})")
        hard, soft = reference_kv_diff(r.kv, ref, self.adapter.gguf_arch)
        self.check(not hard, "runtime KVs (architecture, hyperparameters, prism.hadamard.*, file type) equal the "
                             "reference file's, by key",
                   f"other differences (expected: general.name; a fine-tune may change tokenizer KVs): {soft}"
                   + (f"; RUNTIME: {hard}" if hard else ""))

    def ptq_content(self, chunk_rows=16384):
        t0 = time.time()
        bad, n_w = [], 0
        for t in self.plan:
            if not t.ternary:
                continue
            in_f, out_f = t.ne
            tt = self.state.tensors[t.source]
            idx = self.rows[t.name]
            for r0 in range(0, out_f, chunk_rows):
                r1 = min(out_f, r0 + chunk_rows)
                gc, gs = self.r.ptq1_0(t.name, r0, r1)
                ii = torch.from_numpy(idx[r0:r1].astype(np.int64))
                sc = tt.codes.index_select(0, ii).numpy()
                ss = tt.scales.index_select(0, ii).numpy().view(np.uint16)
                neg = ss >= 0x8000
                exp_c = np.where(np.repeat(neg, ptq1_0.QK, axis=1), -sc, sc)
                exp_s = np.where(neg, ss - 0x8000, ss).astype(np.uint16)
                exp_c = np.where(np.repeat(exp_s == 0, ptq1_0.QK, axis=1), np.int8(0), exp_c)
                if not (np.array_equal(gc, exp_c) and np.array_equal(gs.view(np.uint16), exp_s)):
                    bad.append(f"{t.name} rows {r0}..{r1}")
                    break
                n_w += gc.size
        self.check(not bad, "decoded PTQ1_0 codes and fp16 scale bits == permuted, canonical state (exact)",
                   f"{n_w:,} weights in {time.time() - t0:.0f}s, rows by {self.basis}"
                   + (f"; MISMATCH in {bad[:3]}" if bad else ""))

    def f32_content(self):
        bad, n = [], 0
        for t in self.plan:
            if t.ternary:
                continue
            got = self.r.read(t.name).reshape(-1).view("<u4")
            a = _f32_reference(self.ops[t.name], self.producer.source_float(t))
            want = np.ascontiguousarray(a[self.rows[t.name]], dtype="<f4").reshape(-1).view("<u4")
            if got.shape != want.shape or not np.array_equal(got, want):
                bad.append(t.name)
            n += 1
        self.check(not bad, "F32 tensors bit-identical to the transformed sources",
                   f"{n} tensors, rows and transforms by {self.basis}" + (f"; MISMATCH {bad[:4]}" if bad else ""))

    def functional(self, layers=None, rows=8, vecs=4):
        """FP64: the runtime's W_gguf @ H(s * x) (signs from the file, popcount Hadamard) against
        the fake-quant (C * s) R^T x of the matching source rows (state signs, Sylvester H)."""
        rng = np.random.default_rng(1234)
        r = self.r
        kv_signs = {w: s.astype(np.float64) for w, s in prism_signs(r).items()}
        fb = int(r.value("prism.hadamard.block_size"))
        sb = self.state.block
        hp, hs = _hadamard_popcount(fb), _hadamard_sylvester(sb)
        n_layers = max((int(t.name.split(".")[1]) for t in self.plan if t.name.startswith("blk.")), default=-1) + 1
        if layers is None:
            kinds = {}
            for t in self.plan:
                if t.name.startswith("blk.") and t.ternary:
                    li = int(t.name.split(".")[1])
                    kinds.setdefault(frozenset(x.name.split(".", 2)[2] for x in self.plan
                                               if x.name.startswith(f"blk.{li}.")), li)
            layers = sorted(set(kinds.values()) | {n_layers - 1})
        names = [t for t in self.plan if t.ternary and (not t.name.startswith("blk.")
                                                        or int(t.name.split(".")[1]) in layers)]
        worst = 0.0
        for t in names:
            in_f, out_f = t.ne
            idx = self.rows[t.name]
            k = min(rows, out_f)
            moved = np.nonzero(idx != np.arange(out_f))[0]
            if len(moved) and k > 1:
                pick = np.concatenate([rng.choice(moved, min(k // 2, len(moved)), replace=False),
                                       rng.choice(out_f, k - min(k // 2, len(moved)), replace=False)])
                pick = np.unique(pick)
            else:
                pick = np.sort(rng.choice(out_f, k, replace=False))
            gc, gs = ptq1_0.decode(np.concatenate([r.read(t.name, int(i), int(i) + 1) for i in pick]), in_f)
            wg = (gc.reshape(len(pick), -1, ptq1_0.QK).astype(np.float64)
                  * gs.astype(np.float64)[..., None]).reshape(len(pick), in_f)
            tt = self.state.tensors[t.source]
            src = torch.from_numpy(idx[pick].astype(np.int64))
            c = tt.codes.index_select(0, src).numpy().astype(np.float64)
            s = tt.scales.index_select(0, src).float().numpy().astype(np.float64)
            wq = (c.reshape(len(pick), -1, ptq1_0.QK) * s[..., None]).reshape(len(pick), in_f)
            s_st = self.state.signs[in_f].numpy().astype(np.float64)
            w_ref = np.concatenate([wq[:, b:b + sb] @ hs.T for b in range(0, in_f, sb)], 1) * s_st[None, :]
            s_kv = kv_signs[in_f]
            if t.role == "inverse":   # embedding lookup: e = s * (H z)
                got = np.concatenate([wg[:, b:b + fb] @ hp.T for b in range(0, in_f, fb)], 1) * s_kv[None, :]
                rel = np.linalg.norm(got - w_ref) / max(np.linalg.norm(w_ref), 1e-300)
            else:                    # runtime: W_gguf @ H(s * x)
                x = rng.standard_normal((in_f, vecs))
                xr = np.concatenate([hp @ (s_kv[b:b + fb, None] * x[b:b + fb]) for b in range(0, in_f, fb)], 0)
                y_ref = w_ref @ x
                rel = np.linalg.norm(wg @ xr - y_ref) / max(np.linalg.norm(y_ref), 1e-300)
            worst = max(worst, float(rel))
            if rel >= FUNC_TOL:
                self.check(False, f"function {t.name}", f"rel {rel:.2e} on {len(pick)} rows")
        self.check(worst < FUNC_TOL, "function check (FP64, runtime transform vs fake-quant source rows)",
                   f"{len(names)} tensors in layers {layers} + endpoints, worst rel {worst:.2e}")


def verify_gguf(path, adapter, plan, infos, kvs, state, producer, reference, *, chunk_rows=16384,
                func_layers=None, log=print) -> list[dict]:
    with GGUFReader(path) as r:
        v = Verifier(r, adapter, plan, kvs, state, producer, reference, log)
        log("verify: structure and KVs")
        v.structure(infos)
        if reference is not None:
            log("verify: against the reference file's header")
            v.against_reference()
        log("verify: content")
        if v.independent:
            v.layout_rule()
        else:
            log(f"  note: {adapter.name} states no independent row rule; content checks use {v.basis}")
        v.ptq_content(chunk_rows)
        v.f32_content()
        v.functional(func_layers)
    return v.results


# -- export -------------------------------------------------------------------

def export_gguf(adapter, source, model_dir, out, *, name: str, small: dict | None = None,
                size_label: str | None = None, verify: bool = True, chunk_rows: int = 16384,
                func_layers=None, log=print) -> dict:
    """Write ``out`` from a ternary state; returns the report (also ``<out>.export.json``).

    ``source``: a TernaryState (codes, scales, signs per width) or a research
    ``load_legacy_pt`` dict; ``small``: distilled high-precision tensors (FP32,
    names relative to the text model); ``model_dir``: the model's files (config,
    tokenizer, checkpoint tensors kept in F32)."""
    if adapter.gguf_arch is None:
        raise ExportError(f"the {adapter.name} adapter has no PTQ1_0 GGUF layout")
    t0 = time.time()
    out = Path(out)
    state = as_state(source)
    small = dict(small or {})
    plan = adapter.gguf_tensors()
    notes = check_inputs(plan, state, small)
    infos = [t.info() for t in plan]
    layout(infos)
    kvs, kv_notes = adapter.gguf_kvs(model_dir, name=name, signs=state.signs, block=state.block, tensors=plan,
                                     size_label=size_label)
    notes += kv_notes
    reference = adapter.gguf_reference()
    if reference is not None:
        got = [(t.name, t.gtype, list(t.ne), t.offset) for t in infos]
        want = [(h["name"], h["type"], list(h["ne"]), h["offset"]) for h in reference["tensors"]]
        if got != want:
            first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
            raise ExportError(f"the config is the reference model's but the tensor layout differs at {first}: "
                              f"{got[first] if first < len(got) else None} vs "
                              f"{want[first] if first < len(want) else None}")
        hard, soft = reference_kv_diff(kvs, reference, adapter.gguf_arch)
        if hard:
            raise ExportError(f"the config is the reference model's but runtime KVs differ from the reference "
                              f"file's: {hard[:4]}")
        notes.append("layout equals the reference file's (Ternary-Bonsai-2-9B): tensor names, order, types, ne, "
                     "offsets and the runtime KVs"
                     + (f"; other KVs differ (tokenizer or descriptive metadata): {soft}" if soft else ""))
    for n in notes:
        log(f"note: {n}")
    tmp = out.with_name(out.name + ".tmp")
    report = {"output": str(out), "name": name, "notes": notes, "reference": reference is not None}
    with Checkpoint(model_dir) as ckpt:
        check_coverage(adapter, plan, ckpt)
        producer = _Producer(adapter, plan, state, ckpt, small, chunk_rows)
        log(f"writing {tmp} ({len(infos)} tensors, {len(kvs)} KVs)")
        try:
            info = write_gguf(tmp, kvs, infos, producer, log=log)
        except BaseException:   # a refused tensor, a full disk, Ctrl-C: no partial file stays behind
            tmp.unlink(missing_ok=True)
            raise
        report["write"] = {**info, **producer.stats}
        log(f"wrote {info['file_size']:,} B in {info['seconds']:.0f}s; {producer.stats['ternary_weights']:,} ternary "
            f"weights, {producer.stats['neg_scale_groups']} negative-scale groups normalized, "
            f"{producer.stats['zero_scale_groups']} zero-scale groups "
            f"({producer.stats['zero_scale_codes_cleared']} had nonzero codes, cleared)")
        if verify:
            results = verify_gguf(tmp, adapter, plan, infos, kvs, state, producer, reference,
                                  chunk_rows=chunk_rows, func_layers=func_layers, log=log)
            report["verify"] = results
            failed = [x["check"] for x in results if not x["ok"]]
            log(f"verify: {len(results) - len(failed)} passed, {len(failed)} failed")
            if failed:
                atomic_write_json(out.with_name(out.name + ".export.json"), report)
                raise ExportError(f"verification failed ({failed[:3]}); the unverified file is {tmp}")
    os.replace(tmp, out)
    report["seconds"] = time.time() - t0
    atomic_write_json(out.with_name(out.name + ".export.json"), report)
    return report


def run(ctx) -> dict:
    from ..config import ConfigError, distill_uses_endpoints, upstream
    e = ctx.cfg["export"]
    if (e.get("format") or "ptq1_0-gguf") != "ptq1_0-gguf":
        raise ConfigError(f"export.format {e.get('format')!r}: only ptq1_0-gguf is supported")
    log = ctx.log("export")
    adapter = ctx.adapter()
    source = upstream(ctx.cfg, "export")[0]
    if source == "distill" and not distill_uses_endpoints(ctx.cfg):
        raise ConfigError(EXPORT_NEEDS_DISTILLED_ENDPOINTS)
    state = ctx.state_through(source)
    small = ctx.distill_small() if source == "distill" else {}
    name = e.get("name") or default_name(ctx.model_path())
    out = Path(e["out"]) if e.get("out") else ctx.run.stage_dir("export") / f"{name}.gguf"
    log(f"exporting the {source} state to {out}")
    report = export_gguf(adapter, state, ctx.model_dir(), out, name=name, small=small,
                         size_label=e.get("size_label"), verify=bool(e.get("verify", True)),
                         chunk_rows=int(e.get("chunk_rows") or 16384), log=log)
    checks = report.get("verify") or []
    return {"output": str(out), "source": source, "file_size": report["write"]["file_size"],
            "distilled_f32": report["write"]["distilled_f32"], "verified": bool(checks),
            "checks_passed": sum(x["ok"] for x in checks)}
