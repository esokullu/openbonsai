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


# Keys a data source entry (calibrate.sources, distill.teacher.sources,
# eval.compare.sources) or a trace prompt spec may carry; which ones apply
# depends on its ``kind``.
SOURCE_KEYS = frozenset({"kind", "n", "path", "seed", "min_row", "repo", "revision", "file", "shuffle"})
PROMPT_KEYS = frozenset({"kind", "path", "n", "rows", "seed", "repo", "revision", "file"})
_SOURCE_LISTS = {("calibrate", "sources"), ("distill", "teacher", "sources"), ("eval", "compare", "sources")}
_PROMPTS = ("traces", "prompts")


def _check_sources(value, path: tuple) -> None:
    where = ".".join(path)
    if value is None and path != ("calibrate", "sources"):
        return
    if not isinstance(value, list) or not all(isinstance(s, dict) for s in value):
        raise ConfigError(f"{where} must be a list of mappings")
    for i, src in enumerate(value):
        unknown = sorted(set(src) - SOURCE_KEYS)
        if unknown:
            raise ConfigError(f"unknown keys in {where}[{i}]: {unknown}; known: {sorted(SOURCE_KEYS)}")
        if "kind" not in src:
            raise ConfigError(f"{where}[{i}] needs a kind")


def _check_mapping(value, schema: dict, path: tuple) -> None:
    where = ".".join(path)
    if not isinstance(value, dict):
        raise ConfigError(f"{where} must be a mapping")
    allowed = PROMPT_KEYS if path == _PROMPTS else frozenset(schema)
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConfigError(f"unknown keys in {where!r}: {unknown}; known: {sorted(allowed)}")
    if path == _PROMPTS:
        return
    for key, item in value.items():
        sub = path + (key,)
        if sub in _SOURCE_LISTS:
            _check_sources(item, sub)
        elif sub == _PROMPTS:
            if item is not None:
                _check_mapping(item, {}, sub)
        elif isinstance(schema[key], dict):
            if item is not None:
                _check_mapping(item, schema[key], sub)
        elif isinstance(item, dict):
            raise ConfigError(f"{'.'.join(sub)} must be a value, not a mapping")


def check_keys(cfg: dict) -> dict:
    """Reject unknown sections and keys at every level (typos), and values whose
    shape (mapping, list of sources, scalar) does not match the defaults."""
    for section, value in cfg.items():
        if section not in DEFAULTS:
            raise ConfigError(f"unknown config section {section!r}; known: {sorted(DEFAULTS)}")
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ConfigError(f"config section {section!r} must be a mapping")
        _check_mapping(value, DEFAULTS[section], (section,))
    return cfg


def _index(items: list, part: str, where: str) -> int:
    if not part.isdigit() or int(part) >= len(items):
        raise ConfigError(f"--set {where}: expected a list index below {len(items)}")
    return int(part)


def apply_set(cfg: dict, assignment: str) -> dict:
    """``a.b.c=value`` (YAML value). A numeric part indexes a list: ``calibrate.sources.2.path=x``."""
    if "=" not in assignment:
        raise ConfigError(f"--set expects key=value, got {assignment!r}")
    key, raw = assignment.split("=", 1)
    parts = [p for p in key.strip().split(".") if p]
    if len(parts) < 2:
        raise ConfigError(f"--set needs section.key=value, got {assignment!r}")
    value = yaml.safe_load(raw) if raw.strip() else None
    node = cfg
    for depth, part in enumerate(parts[:-1]):
        where = ".".join(parts[:depth + 1])
        if isinstance(node, list):
            node = node[_index(node, part, where)]
            continue
        if part not in node:
            node[part] = {}            # a new key: check_keys below rejects it unless the schema has it
        elif not isinstance(node[part], (dict, list)):
            raise ConfigError(f"--set {key}: {where} is {node[part]!r}, not a mapping or list")
        node = node[part]
    if isinstance(node, list):
        node[_index(node, parts[-1], key)] = value
    else:
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
