import json
from pathlib import Path

import pytest
import torch
import transformers

from openbonsai import adapter
from openbonsai.cli import main, parser, train_command
from openbonsai.export import load_compressed


@pytest.mark.parametrize("lora_rank", [0, 4])
@pytest.mark.parametrize("factor_mapping,reconstruction_loss", [("exp", "mse"), ("sigmoid", "huber")])
def test_train_entry_point_with_local_fixture_and_revision(
    monkeypatch, tmp_path, lora_rank, factor_mapping, reconstruction_loss,
):
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
        "--cache-dir", str(tmp_path / "cache"), "--lora-rank", str(lora_rank),
        "--lora-learning-rate", "0.002", "--factor-mapping", factor_mapping,
        "--reconstruction-loss", reconstruction_loss, "--huber-delta", "0.25"])
    result = train_command(args)
    assert calls[0][1]["local_files_only"] and calls[1][1]["local_files_only"]
    assert calls[0][1]["revision"] == calls[1][1]["revision"] == "test-commit"
    assert result["storage"]["artifact_bytes"] > 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["metadata"]["resolved_revision"] == "test-commit"
    assert manifest["metadata"]["calibration"]["expected_revision"] == "test-commit"
    assert manifest["metadata"]["lora_rank"] == lora_rank
    assert manifest["metadata"]["lora_initialization_seed"] == args.seed
    assert manifest["metadata"]["factor_mapping"] == factor_mapping
    training = manifest["metadata"]["training"]
    assert training["training_config"]["lora_learning_rate"] == 0.002
    assert training["quantization"]["lora_ranks"] == [lora_rank]
    assert training["quantization"]["factor_mappings"] == [factor_mapping]
    assert training["training_config"]["reconstruction_loss"] == reconstruction_loss
    assert training["training_config"]["huber_delta"] == 0.25
    assert training["reconstruction_objective"]["name"] == reconstruction_loss
    assert (training["quantization"]["lora_parameters"] > 0) == (lora_rank > 0)
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


@pytest.mark.parametrize("option,value", [
    ("--lora-rank", "-1"), ("--lora-learning-rate", "0"),
    ("--lora-learning-rate", "nan"), ("--lora-learning-rate", "inf"),
    ("--huber-delta", "0"), ("--huber-delta", "-0.1"),
    ("--huber-delta", "nan"), ("--huber-delta", "inf"),
])
def test_invalid_training_config_rejected_before_model_loading(option, value, monkeypatch, tmp_path):
    def forbid(*args, **kwargs):
        raise AssertionError("must not load model for invalid correction settings")

    monkeypatch.setattr(adapter, "load_base_model", forbid)
    args = parser().parse_args(["train", "--calibration", "absent.jsonl",
                              "--output", str(tmp_path / "unused"), option, value])
    with pytest.raises(ValueError):
        train_command(args)


def test_default_lora_recipe():
    args = parser().parse_args(["train", "--calibration", "unused", "--output", "unused"])
    assert args.lora_rank == 64
    assert args.lora_learning_rate == 0.001
    assert args.learning_rate == 0.001
    assert args.factor_mapping == "exp"
    assert args.reconstruction_loss == "mse"
    assert args.huber_delta == 1.0


@pytest.mark.parametrize("command,required", [
    ("train", ["--calibration", "unused", "--output", "unused"]),
    ("smoke", ["--output", "unused"]),
])
def test_reconstruction_options_are_independent_and_reject_unknown_names(command, required):
    for mapping in ("exp", "sigmoid"):
        for loss in ("mse", "huber"):
            args = parser().parse_args([command, *required, "--factor-mapping", mapping,
                                       "--reconstruction-loss", loss, "--huber-delta", "0.25"])
            assert (args.factor_mapping, args.reconstruction_loss, args.huber_delta) == (mapping, loss, 0.25)
    for option in ("--factor-mapping", "--reconstruction-loss"):
        with pytest.raises(SystemExit):
            parser().parse_args([command, *required, option, "unknown"])


def test_offline_smoke_accepts_sigmoid_and_huber(tmp_path, capsys):
    output = tmp_path / "sigmoid-huber"
    main(["smoke", "--output", str(output), "--factor-mapping", "sigmoid",
          "--reconstruction-loss", "huber", "--huber-delta", "0.25"])
    result = json.loads(capsys.readouterr().out)
    assert result["real_9b_weights_loaded"] is False
    assert result["packed_roundtrip_max_logit_error"] == 0
    assert result["training"]["quantization"]["factor_mappings"] == ["sigmoid"]
    assert result["training"]["reconstruction_objective"]["name"] == "huber"
    assert result["training"]["reconstruction_objective"]["huber_delta"] == 0.25


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
