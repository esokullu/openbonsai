"""One run directory holds every stage's output.

    runs/<name>/
      run.yaml            resolved config
      signs.safetensors   per-width sign vectors (written once)
      traces/ calib/ quantize/ endpoints/ distill/ export/ eval/
      logs/<stage>.log

A stage is complete when ``<stage>/DONE.json`` exists; it records the stage's
config hash. Rerunning skips a stage whose hash is unchanged. Within a stage,
units of work (a decoder layer, a teacher shard, a distillation checkpoint)
are written atomically, so a crash loses at most the unit in progress.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import yaml

from .config import dump
from .formats.atomic import atomic_write_bytes, atomic_write_json

_STAGE_DIRS = {"calibrate": "calib"}


class RunDir:
    def __init__(self, root):
        self.root = Path(root)

    @property
    def config_path(self) -> Path:
        return self.root / "run.yaml"

    @property
    def signs_path(self) -> Path:
        return self.root / "signs.safetensors"

    def stage_dir(self, stage: str, create: bool = True) -> Path:
        path = self.root / _STAGE_DIRS.get(stage, stage)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    # -- config ----------------------------------------------------------
    def save_config(self, cfg: dict) -> None:
        atomic_write_bytes(self.config_path, dump(cfg).encode("utf-8"))

    def load_config(self) -> dict | None:
        if not self.config_path.exists():
            return None
        return yaml.safe_load(self.config_path.read_text(encoding="utf-8"))

    # -- completion --------------------------------------------------------
    def done_info(self, stage: str) -> dict | None:
        path = self.stage_dir(stage, create=False) / "DONE.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def is_done(self, stage: str, config_hash: str) -> bool:
        info = self.done_info(stage)
        return info is not None and info.get("config_hash") == config_hash

    def mark_done(self, stage: str, config_hash: str, **info) -> None:
        atomic_write_json(self.stage_dir(stage) / "DONE.json",
                          {"stage": stage, "config_hash": config_hash, **info})

    def clear_done(self, stage: str) -> None:
        path = self.stage_dir(stage, create=False) / "DONE.json"
        if path.exists():
            path.unlink()

    # -- logging -----------------------------------------------------------
    def logger(self, stage: str, echo: bool = True):
        path = self.root / "logs" / f"{stage}.log"
        path.parent.mkdir(parents=True, exist_ok=True)

        def log(msg: str) -> None:
            line = f"[{time.strftime('%H:%M:%S')}] {stage}: {msg}"
            if echo:
                print(line, flush=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        return log
