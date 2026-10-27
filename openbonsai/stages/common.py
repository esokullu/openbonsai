"""Shared stage plumbing: config access, model loading, signs, state install."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import torch

from ..adapters import ModelAdapter, load_adapter
from ..config import canonical_hash, stage_inputs, upstream
from ..core.geometry import Geometry, check_signs, derive_geometry, make_signs
from ..core.ternary import TernaryEmbedding, TernaryLinear, dense_weight, shared_hadamard
from ..formats.state import TernaryState, TernaryTensor, load_scales, load_signs, load_state, save_signs
from ..rundir import RunDir

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class StageContext:
    cfg: dict
    run: RunDir
    _adapter: ModelAdapter | None = field(default=None, repr=False)

    # -- config ----------------------------------------------------------
    @property
    def device(self) -> torch.device:
        dev = self.cfg["model"].get("device") or "auto"
        if dev == "auto":
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(dev)

    @property
    def dtype(self) -> torch.dtype:
        return _DTYPES[self.cfg["model"].get("dtype") or "bfloat16"]

    def stage_hash(self, stage: str) -> str:
        ups = {u: (self.run.done_info(u) or {}).get("config_hash") for u in upstream(self.cfg, stage)}
        return canonical_hash({"inputs": stage_inputs(self.cfg, stage), "upstream": ups})

    def log(self, stage: str):
        return self.run.logger(stage)

    # -- model -------------------------------------------------------------
    def model_path(self) -> str:
        path = self.cfg["model"].get("path")
        if not path:
            raise ValueError("model.path is not set (use --model or the config's model.path)")
        return str(path)

    def adapter(self) -> ModelAdapter:
        if self._adapter is None:
            m = self.cfg["model"]
            self._adapter = load_adapter(self.model_path(), revision=m.get("revision"),
                                         local_files_only=bool(m.get("local_files_only")),
                                         name=m.get("adapter"))
        return self._adapter

    def load_model(self):
        m = self.cfg["model"]
        return self.adapter().load(self.model_path(), revision=m.get("revision"), device=self.device,
                                   dtype=self.dtype, attn=m.get("attn") or "sdpa",
                                   local_files_only=bool(m.get("local_files_only")),
                                   keep_vision=bool(m.get("keep_vision")))

    def load_tokenizer(self):
        m = self.cfg["model"]
        return self.adapter().load_tokenizer(self.model_path(), revision=m.get("revision"),
                                             local_files_only=bool(m.get("local_files_only")))

    # -- rotation ----------------------------------------------------------
    def geometry(self, model) -> Geometry:
        a = self.adapter()
        r = self.cfg["rotation"]
        return derive_geometry(a.quantized_widths(model), a.hidden_size,
                               group=int(r["group"]), max_block=int(r["max_block"]))

    def signs(self, model) -> tuple[dict, Geometry]:
        """Per-width signs for this run: written once to run/signs.safetensors."""
        geo = self.geometry(model)
        if self.run.signs_path.exists():
            signs, block = load_signs(self.run.signs_path)
            if block != geo.block:
                raise ValueError(f"run signs were made for block {block}, model geometry needs {geo.block}")
        else:
            spec = self.cfg["rotation"]["signs"] or {}
            if spec.get("path"):
                signs, _ = load_signs(spec["path"])
                signs = {w: signs[w] for w in geo.widths if w in signs}
            else:
                signs = make_signs(geo, spec.get("mode", "seeded"), spec.get("seed", "width"))
            check_signs(signs, geo)
            save_signs(self.run.signs_path, signs, block=geo.block, meta={"geometry": geo.to_json()})
        check_signs(signs, geo)
        return signs, geo

    # -- stage outputs -----------------------------------------------------
    def calib_tokens(self) -> torch.Tensor:
        path = self.run.stage_dir("calibrate", create=False) / "tokens.npy"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run the calibrate stage first")
        return torch.from_numpy(np.load(path).astype(np.int64))

    def quantize_state(self) -> TernaryState:
        return load_state(self.run.stage_dir("quantize", create=False) / "state.safetensors")

    def endpoints_state(self) -> TernaryState | None:
        path = self.run.stage_dir("endpoints", create=False) / "state.safetensors"
        return load_state(path) if path.exists() else None

    def state_through(self, stage: str) -> TernaryState:
        """Decoder state, plus endpoints, plus distilled scales, up to ``stage``."""
        state = self.quantize_state()
        # Distillation trains (and so evaluates) the endpoints only when it installed them.
        if stage == "endpoints" or (stage == "distill" and self.cfg["distill"].get("endpoints", True)):
            ep = self.endpoints_state()
            if ep is not None and ep.tensors:
                state = state.merged(ep)
        if stage == "distill":
            scales = load_scales(self.run.stage_dir("distill", create=False) / "scales.safetensors")
            state = state.with_scales(scales)
        return state


def resolve_module(adapter: ModelAdapter, model, key: str):
    """(parent, attribute, module) for a relative key."""
    if key == "lm_head":
        return None, "lm_head", adapter.head(model)
    if key == "embed_tokens":
        return None, "embed_tokens", adapter.embedding(model)
    head, li, sub = key.split(".", 2)
    if head != "layers":
        raise KeyError(f"unknown module key {key!r}")
    layer = adapter.layers(model)[int(li)]
    parent_name, _, attr = sub.rpartition(".")
    parent = layer.get_submodule(parent_name) if parent_name else layer
    return parent, attr, getattr(parent, attr)


@torch.no_grad()
def apply_dense(adapter: ModelAdapter, model, records) -> int:
    """Write (C * s) R^T into ordinary Linear/Embedding weights (``apply_state``)."""
    n = 0
    for key, tensor, signs, block in records:
        _, _, module = resolve_module(adapter, model, key)
        w = dense_weight(tensor.codes, tensor.scales, signs, block, device=module.weight.device,
                         dtype=module.weight.dtype)
        if w.shape != module.weight.shape:
            raise ValueError(f"{key}: state shape {tuple(w.shape)} != module {tuple(module.weight.shape)}")
        module.weight.copy_(w)
        n += 1
    return n


def install_ternary(adapter: ModelAdapter, model, records, device) -> list:
    """Replace modules with TernaryLinear/TernaryEmbedding (``install_ternary``/``install_endpoints``)."""
    hs, installed = {}, []
    for key, tensor, signs, block in records:
        if block not in hs:
            hs[block] = shared_hadamard(block, device)
        parent, attr, module = resolve_module(adapter, model, key)
        args = (tensor.codes.to(device), tensor.scales.to(device), signs.to(device), block, hs[block])
        if key == "embed_tokens":
            new = TernaryEmbedding(*args)
            adapter.set_embedding(model, new)
        else:
            bias = getattr(module, "bias", None)
            new = TernaryLinear(*args, bias=None if bias is None else bias.to(device))
            if key == "lm_head":
                adapter.set_head(model, new)
            else:
                setattr(parent, attr, new)
        del module
        installed.append((key, new))
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return installed


def legacy_records(legacy: dict):
    """Records from ``load_legacy_pt``: each module keeps its own signs and block."""
    for key in sorted(legacy):
        t = legacy[key]
        yield key, TernaryTensor(t.codes, t.scales), t.signs, t.block


def free(device) -> None:
    if getattr(device, "type", device) == "cuda":
        torch.cuda.empty_cache()
