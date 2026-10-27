"""Byte parity of the package exporter with research/experiments/export_gguf.py on a 9B-shaped state.

Slow (two ~2 GB files, a ~9 GB state in memory, minutes on CPU) and dependent
on local data, so it is skipped unless OPENBONSAI_SLOW=1:

    OPENBONSAI_SLOW=1 CUDA_VISIBLE_DEVICES=-1 python -m pytest tests/test_export_parity.py -s

Inputs (environment variables, defaults are this machine's paths):
- OPENBONSAI_PARITY_STATE: a directory with research-layout ``ternary_state.pt``,
  ``endpoint_state.pt`` and optionally ``e2e_state.pt`` (make_synthetic_state.py);
- OPENBONSAI_PARITY_SNAPSHOT: the Qwen3.5-9B snapshot (config, tokenizer, F32 sources);
- OPENBONSAI_PARITY_OUT: where both files go (default: pytest's tmp dir);
- OPENBONSAI_PARITY_RESEARCH_GGUF: an existing research-exporter output to compare
  against; otherwise the research script runs in a subprocess (its source is
  executed with CUDA_VISIBLE_DEVICES forced to -1; nothing imports research/).
The result of the manual run is recorded in research/validation/p2-export-parity.md.
"""
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
STATE = Path(os.environ.get("OPENBONSAI_PARITY_STATE", "D:/openbonsai/runs/gguf-test/synthetic"))
SNAPSHOT = Path(os.environ.get("OPENBONSAI_PARITY_SNAPSHOT", "D:/openbonsai/runs/hf-cache/hub/models--Qwen--Qwen3.5-9B/"
                                                              "snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a"))
RESEARCH_SCRIPT = ROOT / "research" / "experiments" / "export_gguf.py"
NAME = "OpenBonsai-Qwen3.5-9B-PTQ1_0"

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(os.environ.get("OPENBONSAI_SLOW") != "1", reason="slow; set OPENBONSAI_SLOW=1"),
              pytest.mark.skipif(not (STATE / "ternary_state.pt").exists() or not SNAPSHOT.is_dir(),
                                 reason="parity inputs not on this machine")]

# Run the research exporter's own source with the GPU hidden: on Windows its
# CUDA_VISIBLE_DEVICES="" line would unset the variable instead.
_RUNNER = """
import sys
path = sys.argv[1]
src = open(path, encoding="utf-8").read()
line = 'os.environ["CUDA_VISIBLE_DEVICES"] = ""'
assert src.count(line) == 1
src = src.replace(line, 'os.environ["CUDA_VISIBLE_DEVICES"] = "-1"')
sys.argv = sys.argv[1:]
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
"""


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def first_difference(a, b, chunk=16 << 20):
    with open(a, "rb") as fa, open(b, "rb") as fb:
        off = 0
        while True:
            x, y = fa.read(chunk), fb.read(chunk)
            if x != y:
                n = min(len(x), len(y))
                i = next((k for k in range(n) if x[k] != y[k]), n)
                return off + i
            if not x:
                return None
            off += len(x)


def test_package_writer_is_byte_identical_to_the_research_exporter(tmp_path):
    from openbonsai.adapters import load_adapter
    from openbonsai.formats.state import legacy_to_state, load_legacy_pt
    from openbonsai.stages.export import export_gguf

    out = Path(os.environ.get("OPENBONSAI_PARITY_OUT", tmp_path))
    out.mkdir(parents=True, exist_ok=True)
    e2e = STATE / "e2e_state.pt"
    research = Path(os.environ.get("OPENBONSAI_PARITY_RESEARCH_GGUF", out / "research.gguf"))
    if not research.exists():
        cmd = [sys.executable, "-c", _RUNNER, str(RESEARCH_SCRIPT), "--state", str(STATE / "ternary_state.pt"),
               "--endpoints", str(STATE / "endpoint_state.pt"), "--snapshot", str(SNAPSHOT), "--out", str(research),
               "--name", NAME] + (["--e2e", str(e2e)] if e2e.exists() else [])
        subprocess.run(cmd, check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})

    legacy = load_legacy_pt(STATE / "ternary_state.pt")
    legacy.update(load_legacy_pt(STATE / "endpoint_state.pt"))
    state = legacy_to_state(legacy)
    if e2e.exists():
        trained = torch.load(e2e, map_location="cpu", weights_only=True)
        assert not trained.get("norms") and not trained.get("small"), "parity covers scales only"
        state = state.with_scales(trained["scales"])
    package = out / "package.gguf"
    report = export_gguf(load_adapter(SNAPSHOT, local_files_only=True), state, SNAPSHOT, package, name=NAME,
                         verify=True)
    assert report["reference"] and all(x["ok"] for x in report["verify"])
    diff = first_difference(research, package)
    assert research.stat().st_size == package.stat().st_size == 1_997_486_976
    assert diff is None, f"first differing byte at offset {diff}"
    assert sha256(research) == sha256(package)
    print(f"\nparity: {package} == {research}: sha256 {sha256(package)}; "
          f"write stats {({k: v for k, v in report['write'].items() if k != 'seconds'})}")
    np.testing.assert_equal(report["write"]["ternary_weights"], 8_946_450_432)
