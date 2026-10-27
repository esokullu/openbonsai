"""Config resolution and the ``openbonsai`` command line end to end on the tiny model."""
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from openbonsai import cli
from openbonsai.config import ConfigError, DEFAULTS, apply_set, load_config, stage_inputs

ROOT = Path(__file__).resolve().parents[1]


def test_config_layers_defaults_file_and_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"quantize": {"damp": 0.02}, "model": {"path": "x"}}))
    cfg = load_config(path, ["quantize.batch=8", "distill.teacher.n_seq=16", "eval.suites=[dev]"], model="y")
    assert cfg["quantize"] == {"damp": 0.02, "batch": 8} and cfg["model"]["path"] == "y"
    assert cfg["distill"]["teacher"]["n_seq"] == 16 and cfg["distill"]["teacher"]["topk"] == 64
    assert cfg["eval"]["suites"] == ["dev"] and DEFAULTS["quantize"]["damp"] == 0.01
    for bad in ("quantize.dmap=1", "nosuch.key=1", "quantize=1"):
        with pytest.raises(ConfigError):
            apply_set(load_config(), bad)
    path.write_text(yaml.safe_dump({"quantise": {"damp": 0.02}}))
    with pytest.raises(ConfigError, match="quantise"):
        load_config(path)


def test_set_indexes_lists_and_rejects_nested_typos_and_scalar_parents():
    cfg = load_config(None, ["calibrate.sources.2.path=old/ayot128.jsonl", "traces.prompts.rows=[0, 6000]",
                             "distill.teacher.sources=[{kind: jsonl, path: x.jsonl, n: 3}]",
                             "distill.teacher.sources.0.n=5"])
    assert isinstance(cfg["calibrate"]["sources"], list) and len(cfg["calibrate"]["sources"]) == 3
    assert cfg["calibrate"]["sources"][2] == {"kind": "traces", "n": 32, "path": "old/ayot128.jsonl"}
    assert cfg["traces"]["prompts"]["rows"] == [0, 6000] and cfg["distill"]["teacher"]["sources"][0]["n"] == 5
    bad = {"calibrate.sources.7.path=x": "list index", "calibrate.sources.x.path=x": "list index",
           "quantize.damp.x=1": "not a mapping", "distill.teacher.nseq=4": "nseq",
           "rotation.signs.sed=3": "sed", "eval.compare.sources=[{kind: jsonl, pth: x}]": "pth",
           "calibrate.sources=[{n: 3}]": "kind", "rotation.signs=3": "mapping", "quantize.damp={a: 1}": "value",
           "traces.prompts.kinds=x": "kinds"}
    for assignment, match in bad.items():
        with pytest.raises(ConfigError, match=match):
            apply_set(load_config(), assignment)


def test_stage_hash_ignores_device_but_not_numerics():
    a, b = load_config(), load_config(None, ["model.device=cuda"])
    assert stage_inputs(a, "quantize") == stage_inputs(b, "quantize")
    c = load_config(None, ["rotation.impl=fwht"])
    assert stage_inputs(a, "quantize") != stage_inputs(c, "quantize")
    assert stage_inputs(a, "calibrate") == stage_inputs(c, "calibrate")


def test_examples_are_valid_configs():
    for name in ("qwen3_5-9b.yaml", "tiny-qwen3_5.yaml"):
        cfg = load_config(ROOT / "examples" / name)
        assert cfg["rotation"]["group"] == 128
    big = load_config(ROOT / "examples" / "qwen3_5-9b.yaml")
    assert [(s["kind"], s["n"]) for s in big["calibrate"]["sources"]] == [("wikitext2", 128), ("ultrachat", 96), ("traces", 32)]
    assert big["calibrate"]["seq_len"] == 2048 and big["quantize"]["damp"] == 0.01 and big["rotation"]["max_block"] == 4096
    assert big["distill"]["train"] == ["scales"] and "refit" not in json.dumps(big)


def test_end_to_end_run_on_cpu_then_skip_and_rerun(tiny_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    out = tmp_path / "run"
    t0 = time.time()
    assert cli.main(["run", "--model", str(tiny_dir), "--config", "examples/tiny-qwen3_5.yaml", "--out", str(out)]) == 0
    elapsed = time.time() - t0
    assert elapsed < 60, f"tiny end-to-end run took {elapsed:.1f}s"
    for f in ("run.yaml", "signs.safetensors", "traces/traces.jsonl", "calib/tokens.npy", "quantize/state.safetensors",
              "endpoints/state.safetensors", "distill/scales.safetensors", "eval/distill.compare.json"):
        assert (out / f).exists(), f
    result = json.loads((out / "eval" / "distill.compare.json").read_text())
    assert result["tokens"] == 126 and 0 <= result["top1_agreement"] <= 1
    saved = yaml.safe_load((out / "run.yaml").read_text())
    assert saved["model"]["path"] == str(tiny_dir)
    # Second run: everything up to date.
    capsys.readouterr()
    assert cli.main(["run", "--out", str(out), "--until", "distill"]) == 0
    assert "\"quantize\": \"skipped\"" in capsys.readouterr().out
    for stage in ("traces", "calibrate", "quantize", "endpoints", "distill"):
        assert (out / "logs" / f"{stage}.log").read_text().rstrip().endswith("skipped")
    # A numeric override reruns quantize and everything after it, not the data stages.
    before = (out / "quantize" / "state.safetensors").read_bytes()
    assert cli.main(["run", "--out", str(out), "--set", "quantize.damp=0.05", "--until", "endpoints"]) == 0
    assert (out / "quantize" / "state.safetensors").read_bytes() != before
    assert (out / "calib" / "DONE.json").exists()
    assert "skipped" in (out / "logs" / "calibrate.log").read_text().splitlines()[-1]
    # Per-stage commands, inspect and the P2 export placeholder.
    assert cli.main(["eval", "--run", str(out), "--from", "quantize", "--suite", "compare"]) == 0
    assert (out / "eval" / "quantize.compare.json").exists()
    assert cli.main(["inspect", str(out / "quantize" / "state.safetensors")]) == 0
    assert '"block": 128' in capsys.readouterr().out
    assert cli.main(["export", "--run", str(out)]) == 2


def test_module_entry_point_and_missing_model(tmp_path):
    r = subprocess.run([sys.executable, "-m", "openbonsai", "--help"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0 and "quantize" in r.stdout and "distill" in r.stdout
    assert cli.main(["calibrate", "--run", str(tmp_path / "empty")]) == 2
