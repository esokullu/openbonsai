import json
from pathlib import Path

import pytest
import torch
import transformers

from openbonsai import adapter
from openbonsai.cli import main, parser, train_command
from openbonsai.export import load_compressed


def test_train_entry_point_with_local_fixture_and_revision(monkeypatch, tmp_path):
    torch.set_num_threads(2)
    data = tmp_path / "calibration.jsonl"
    data.write_text(json.dumps({"input_ids": [4, 5, 6, 7, 8, 9], "model_id": "fixture",
                               "resolved_revision": "test-commit"}) + "\n")
    calls = []

    def load(model_id, **kwargs):
        calls.append((model_id, kwargs))
        model = adapter.tiny_model(seed=5)
        model.config._commit_hash = "test-commit"
        return model

    class Tokenizer:
        pad_token_id = 0

        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append((model_id, kwargs))
            return cls()

        def save_pretrained(self, path):
            Path(path).mkdir()
            (Path(path) / "test-tokenizer.json").write_text("{}")

    monkeypatch.setattr(adapter, "load_base_model", load)
    # Patch the resolved class: later lazy imports can replace the package's
    # AutoTokenizer attribute when this test runs independently of the suite.
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", Tokenizer.from_pretrained)
    output = tmp_path / "artifact"
    args = parser().parse_args(["train", "--model", "fixture", "--revision", "test-commit",
        "--calibration", str(data), "--output", str(output), "--local-files-only",
        "--epochs", "3", "--batch-size", "1", "--sequence-length", "8", "--stride", "4",
        "--cache-dir", str(tmp_path / "cache")])
    result = train_command(args)
    assert calls[0][1]["local_files_only"] and calls[1][1]["local_files_only"]
    assert calls[0][1]["revision"] == calls[1][1]["revision"] == "test-commit"
    assert result["storage"]["artifact_bytes"] > 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["metadata"]["resolved_revision"] == "test-commit"
    assert manifest["metadata"]["calibration"]["expected_revision"] == "test-commit"
    assert (output / "tokenizer/test-tokenizer.json").is_file()
    loaded = load_compressed(output)
    with torch.no_grad():
        assert torch.isfinite(loaded(input_ids=torch.tensor([[4, 5, 6]]), use_cache=False).logits).all()


def test_train_refuses_existing_output_before_loading_model(tmp_path, monkeypatch):
    (tmp_path / "keep.txt").write_text("keep")
    def forbid(*args, **kwargs):
        raise AssertionError("must not load model")
    monkeypatch.setattr(adapter, "load_base_model", forbid)
    args = parser().parse_args(["train", "--calibration", "absent.jsonl", "--output", str(tmp_path)])
    with pytest.raises(FileExistsError):
        train_command(args)
    assert (tmp_path / "keep.txt").read_text() == "keep"


def test_repack_cli_preserves_assets_and_reports_actual_savings(tmp_path, monkeypatch, capsys):
    from openbonsai import export

    torch.set_num_threads(2)
    model = adapter.tiny_model(seed=7)
    adapter.replace_projections(model)
    source, destination = tmp_path / "original", tmp_path / "smaller"
    before = export.save_compressed(model, source, packing="2bit")
    (source / "tokenizer").mkdir()
    (source / "tokenizer/tokenizer.json").write_text('{"fixture": true}')

    def forbid(*args, **kwargs):
        raise AssertionError("repack must not instantiate or download a model")

    monkeypatch.setattr(export, "model_from_config", forbid)
    monkeypatch.setattr(adapter, "load_base_model", forbid)
    main(["repack", "--input", str(source), "--output", str(destination)])
    result = json.loads(capsys.readouterr().out)
    after = json.loads((destination / "manifest.json").read_text())
    assert result["packing"] == "auto" and result["retrained"] is False
    assert result["artifact_bytes_saved"] == before["storage"]["artifact_bytes"] - after["storage"]["artifact_bytes"]
    assert result["artifact_bytes_saved"] > 0
    assert result["before"]["packed_tensor_bytes"] > result["after"]["packed_tensor_bytes"]
    assert (destination / "tokenizer/tokenizer.json").read_bytes() == (source / "tokenizer/tokenizer.json").read_bytes()


@pytest.mark.parametrize("command,required", [
    ("smoke", ["--output", "unused"]),
    ("train", ["--calibration", "unused.jsonl", "--output", "unused"]),
    ("repack", ["--input", "unused", "--output", "unused-new"]),
])
def test_packing_choices_and_defaults(command, required):
    assert parser().parse_args([command, *required]).packing == "auto"
    for packing in ("auto", "bitcos", "2bit"):
        assert parser().parse_args([command, *required, "--packing", packing]).packing == packing
    with pytest.raises(SystemExit):
        parser().parse_args([command, *required, "--packing", "unknown"])
