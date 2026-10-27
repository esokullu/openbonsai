"""Keep the reasoning suites' TEST problems out of calibration data and trace prompts.

The ``gsm8k`` and ``humaneval`` suites (openbonsai.eval.reasoning) score TEST
problems. If calibration blocks (calibrate, the distillation teacher's data)
or trace prompts contained them, the compressed model would be fitted on what
it is scored on. Three checks run on every build:

- ``check_source``: a hub source naming a reasoning dataset's test split, or a
  local file whose bytes are a pinned test parquet, is refused;
- ``check_texts``: trace prompts that contain a frozen test problem are refused;
- ``check_blocks``: a source's token blocks are decoded as one text (packed
  sources are consecutive cuts of one stream, so a problem split across two
  blocks is still found) and refused if it contains one.

A problem is contained when its whitespace-normalized text is a substring: a
GSM8K question, or a HumanEval prompt from ``def <entry_point>(`` on.

Future math or code calibration must come from TRAINING splits, e.g.
``openai/gsm8k`` ``main/train-00000-of-00001.parquet`` (HumanEval has no
training split; use another dataset's training split, such as MBPP's).
"""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path


class ContaminationError(ValueError):
    """Calibration or trace data contains evaluation test problems."""


ADVICE = ("these are evaluation TEST problems (openbonsai.eval.reasoning); math or code calibration data "
          "must come from training splits (e.g. openai/gsm8k main/train-00000-of-00001.parquet)")


def normalize(text: str) -> str:
    return " ".join(str(text).split())


@lru_cache(maxsize=None)
def needles() -> tuple[tuple[str, str], ...]:
    """(normalized text, problem id) of every frozen test problem."""
    from .reasoning import load_frozen
    out = [(normalize(p["question"]), p["id"]) for p in load_frozen("gsm8k")["problems"]]
    for p in load_frozen("humaneval")["problems"]:
        k = p["prompt"].find(f"def {p['entry_point']}(")
        out.append((normalize(p["prompt"][max(k, 0):]), p["id"]))
    return tuple(out)


def contained(texts) -> list[tuple[int, str]]:
    """(text index, problem id) for every text that contains a frozen test problem."""
    probes = needles()
    hits = []
    for i, text in enumerate(texts):
        flat = normalize(text)
        hits += [(i, pid) for needle, pid in probes if needle in flat]
    return hits


def check_texts(texts, where: str) -> None:
    hits = contained(texts)
    if hits:
        ids = sorted({pid for _, pid in hits})
        raise ContaminationError(f"{where}: {len({i for i, _ in hits})} item(s) contain {len(ids)} test problem(s) "
                                 f"({', '.join(ids[:5])}{', ...' if len(ids) > 5 else ''}); {ADVICE}")


def check_blocks(tokenizer, blocks, where: str, bos: int | None = None) -> None:
    """Decode one source's token blocks in order as one text (block-leading ``bos`` dropped) and
    refuse it if it contains a frozen test problem."""
    ids = [int(t) for b in blocks for t in (b[1:] if bos is not None and len(b) and int(b[0]) == bos else b)]
    check_texts([tokenizer.decode(ids, skip_special_tokens=False)], where)


def check_source(src: dict, where: str, path=None) -> None:
    """Refuse a data source that is a reasoning dataset's test split (by hub name or by file bytes)."""
    from ..digest import file_sha256
    from .reasoning import DATASETS
    repo, file = src.get("repo"), str(src.get("file") or "")
    for d in DATASETS.values():
        if repo == d["repo"] and (not file or d["split"] in Path(file).name.lower() or file == d["file"]):
            raise ContaminationError(f"{where}: {repo} {file or '(no file)'} is not a training split; {ADVICE}")
    if path is not None and Path(path).is_file():
        size = os.path.getsize(path)
        for d in DATASETS.values():
            if size == d["bytes"] and file_sha256(path) == d["sha256"]:
                raise ContaminationError(f"{where}: {path} is {d['repo']} {d['file']}; {ADVICE}")
