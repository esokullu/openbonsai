"""Run configuration: built-in defaults, a YAML file, then ``--set a.b=c`` overrides.

The resolved config is saved as ``run.yaml`` in the run directory. Each stage
hashes the config sections it depends on plus its upstream stages' hashes; a
stage whose output exists with the same hash is skipped.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

STAGES = ("traces", "calibrate", "quantize", "endpoints", "distill", "export", "eval")

DEFAULTS: dict = {
    "model": {
        "path": None,              # HF id or local directory (your fine-tune works the same way)
        "revision": None,          # pin a commit for hub ids
        "adapter": "auto",
        "dtype": "bfloat16",
        "device": "auto",          # auto = cuda if available, else cpu
        "attn": "sdpa",
        "keep_vision": False,
        "local_files_only": False,
    },
    "rotation": {
        "group": 128,
        "max_block": 4096,
        "signs": {"mode": "seeded", "seed": "width", "path": None},  # path: explicit signs.safetensors
        "impl": "dense",           # dense = the experiments' arithmetic; fwht = no dense R
    },
    "traces": {
        "enabled": True,
        "prompts": {"kind": "jsonl", "path": None, "n": None},
        "max_new_tokens": 512,
        "thinking": False,
        "batch": 8,
    },
    "calibrate": {
        "seq_len": 2048,
        "seed": 0,
        "sources": [
            {"kind": "wikitext2", "n": 128},
            {"kind": "ultrachat", "n": 96, "min_row": 6000},
            {"kind": "traces", "n": 32},
        ],
    },
    "quantize": {"damp": 0.01, "batch": 4},
    "endpoints": {"enabled": True, "head": "ternary", "embedding": "ternary", "damp": 0.01,
                  "chunk": 16384, "batch": 4},
    "distill": {
        "enabled": True,
        "teacher": {"n_seq": 2048, "seq_len": 2048, "seed": 1, "shard": 64, "topk": 64, "sources": None},
        "train": ["scales"],
        "endpoints": True,         # train endpoint scales too when the endpoints are ternary
        "lr": 2.0e-3,
        "epochs": 1,
        "accum": 8,
        "warmup": 10,
        "chunk": 512,
        "log_every": 5,
        "ckpt_every": 16,
        "max_steps": None,
    },
    "export": {"format": "ptq1_0-gguf"},
    "eval": {
        "from": "auto",            # auto = the last completed stage
        "suites": ["dev", "extended"],
        "generations": True,
        "compare": {"seq_len": 512, "n": 8, "seed": 7, "topk": 64, "sources": None},
    },
}


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def check_keys(cfg: dict) -> dict:
    """Reject unknown sections and unknown keys one level below them (typos)."""
    for section, value in cfg.items():
        if section not in DEFAULTS:
            raise ConfigError(f"unknown config section {section!r}; known: {sorted(DEFAULTS)}")
        if isinstance(value, dict):
            unknown = sorted(set(value) - set(DEFAULTS[section]))
            if unknown:
                raise ConfigError(f"unknown keys in {section!r}: {unknown}")
        elif value is not None:
            raise ConfigError(f"config section {section!r} must be a mapping")
    return cfg


def apply_set(cfg: dict, assignment: str) -> dict:
    if "=" not in assignment:
        raise ConfigError(f"--set expects key=value, got {assignment!r}")
    key, raw = assignment.split("=", 1)
    parts = [p for p in key.strip().split(".") if p]
    if len(parts) < 2:
        raise ConfigError(f"--set needs section.key=value, got {assignment!r}")
    value = yaml.safe_load(raw) if raw.strip() else None
    node = cfg
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value
    return check_keys(cfg)


def load_config(path=None, sets=(), model: str | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ConfigError(f"{path}: expected a mapping")
        cfg = check_keys(deep_merge(cfg, data))
    for assignment in sets:
        apply_set(cfg, assignment)
    if model is not None:
        cfg["model"]["path"] = model
    return cfg


def dump(cfg: dict) -> str:
    return yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
                          .encode("utf-8")).hexdigest()


_MODEL_IDENTITY = ("path", "revision", "adapter", "dtype", "attn")


def stage_inputs(cfg: dict, stage: str) -> dict:
    """The config sections a stage's output depends on (device excluded)."""
    model = {k: cfg["model"].get(k) for k in _MODEL_IDENTITY}
    sections = {
        "traces": ("traces",),
        "calibrate": ("calibrate",),
        "quantize": ("rotation", "quantize"),
        "endpoints": ("rotation", "endpoints"),
        "distill": ("rotation", "distill"),
        "export": ("export",),
        "eval": ("eval",),
    }[stage]
    return {"stage": stage, "model": model, **{s: cfg[s] for s in sections}}


def upstream(cfg: dict, stage: str) -> tuple[str, ...]:
    uses_traces = any(s.get("kind") == "traces" and not s.get("path") for s in cfg["calibrate"]["sources"])
    teacher_sources = cfg["distill"]["teacher"].get("sources")
    distill_traces = teacher_sources is None or any(
        s.get("kind") == "traces" and not s.get("path") for s in teacher_sources)
    return {
        "traces": (),
        "calibrate": ("traces",) if uses_traces else (),
        "quantize": ("calibrate",),
        "endpoints": ("quantize", "calibrate"),
        "distill": ("quantize", "endpoints") + (("traces",) if distill_traces else ()),
        "export": ("distill",),
        "eval": (),
    }[stage]
