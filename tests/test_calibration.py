import argparse
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from openbonsai.calibration import load_calibration
from openbonsai.prepare import add_arguments, generate_record, read_prompts, run


def write_jsonl(tmp_path, records):
    path = tmp_path / "calibration.jsonl"
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path


def test_calibration_padding_truncation_and_metadata(tmp_path):
    path = write_jsonl(tmp_path, [
        {"model_id": "target", "input_ids": [4, 5], "attention_mask": [1, 0]},
        {"model": "target", "input_ids": [6, 7, 8, 9, 10], "hit_generation_limit": True},
    ])
    samples, manifest = load_calibration(path, "target", 4, pad_token_id=99)
    assert samples[0]["input_ids"].tolist() == [[4, 5, 99, 99]]
    assert samples[0]["attention_mask"].tolist() == [[1, 0, 0, 0]]
    assert samples[1]["input_ids"].tolist() == [[6, 7, 8, 9]]
    assert samples[1]["attention_mask"].tolist() == [[1, 1, 1, 1]]
    assert samples[0]["input_ids"].dtype == torch.long
    assert manifest["source_tokens"] == 7
    assert manifest["retained_tokens"] == 6
    assert manifest["active_tokens"] == 5
    assert manifest["padding_tokens"] == 2
    assert manifest["truncated_tokens"] == 1
    assert manifest["truncated_records"] == 1
    assert manifest["prepared_records_hitting_generation_limit"] == 1
    assert manifest["records_without_model_id"] == 0
    json.dumps(manifest)


@pytest.mark.parametrize("bad", [[], [-1], [True], [1.5], ["1"], [None], [2**63], None])
def test_reject_invalid_ids(tmp_path, bad):
    with pytest.raises(ValueError, match="input_ids"):
        load_calibration(write_jsonl(tmp_path, [{"input_ids": bad}]), "target", 4)


@pytest.mark.parametrize("record", [
    {"input_ids": [1], "model_id": "other"},
    {"input_ids": [1], "model": "other"},
    {"input_ids": [1], "model_id": "target", "model": "other"},
    {"input_ids": [1], "model_id": None},
])
def test_reject_wrong_model_provenance(tmp_path, record):
    with pytest.raises(ValueError, match="does not match target"):
        load_calibration(write_jsonl(tmp_path, [record]), "target", 4)


@pytest.mark.parametrize("mask", [[1], [1, True], [1, 2], [0, 0], "11"])
def test_reject_bad_masks(tmp_path, mask):
    with pytest.raises(ValueError):
        load_calibration(write_jsonl(tmp_path, [{"input_ids": [1, 2], "attention_mask": mask}]), "target", 4)


def test_max_samples_and_missing_provenance(tmp_path):
    path = write_jsonl(tmp_path, [{"input_ids": [1, 2]}, {"input_ids": [-1]}])
    samples, metadata = load_calibration(path, "target", 2, max_samples=1)
    assert len(samples) == 1
    assert metadata["records_without_model_id"] == 1


def test_pinned_revision_and_unknown_provenance(tmp_path):
    path = write_jsonl(tmp_path, [
        {"input_ids": [1], "resolved_revision": "exact-commit"},
        {"input_ids": [2], "revision": "exact-commit"},
        {"input_ids": [3]},
    ])
    samples, metadata = load_calibration(path, "target", 2, expected_revision="exact-commit")
    assert len(samples) == 3
    assert metadata["expected_revision"] == "exact-commit"
    assert metadata["revisions"] == ["exact-commit"]
    assert metadata["records_without_revision"] == 1


@pytest.mark.parametrize("provenance", [
    {"revision": "main"},
    {"resolved_revision": "different-commit"},
    {"resolved_revision": "exact-commit", "revision": "different-commit"},
])
def test_reject_mismatched_revision(tmp_path, provenance):
    path = write_jsonl(tmp_path, [{"input_ids": [1], **provenance}])
    with pytest.raises(ValueError, match="does not match expected revision"):
        load_calibration(path, "target", 2, expected_revision="exact-commit")


def test_collect_revisions_when_no_pinned_revision_requested(tmp_path):
    path = write_jsonl(tmp_path, [
        {"input_ids": [1], "revision": "z-commit"},
        {"input_ids": [2], "resolved_revision": "a-commit"},
    ])
    _, metadata = load_calibration(path, "target", 2)
    assert metadata["revisions"] == ["a-commit", "z-commit"]
    assert metadata["records_without_revision"] == 0


@pytest.mark.parametrize("revision", [None, 123, "", "   "])
def test_reject_malformed_record_revision(tmp_path, revision):
    path = write_jsonl(tmp_path, [{"input_ids": [1], "resolved_revision": revision}])
    with pytest.raises(ValueError, match="revision string"):
        load_calibration(path, "target", 2)


@pytest.mark.parametrize("sequence_length", [0, -1, True, 1.5])
def test_invalid_sequence_length(tmp_path, sequence_length):
    with pytest.raises(ValueError, match="sequence_length"):
        load_calibration(write_jsonl(tmp_path, [{"input_ids": [1]}]), "target", sequence_length)


def test_no_active_tokens_after_truncation(tmp_path):
    with pytest.raises(ValueError, match="no active tokens"):
        load_calibration(write_jsonl(tmp_path, [{"input_ids": [1, 2], "attention_mask": [0, 1]}]), "target", 1)


def test_empty_or_invalid_json(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n")
    with pytest.raises(ValueError, match="no calibration records"):
        load_calibration(path, "target", 2)
    path.write_text('{"input_ids": [1]}\nnot json\n')
    with pytest.raises(ValueError, match=r":2: invalid JSON"):
        load_calibration(path, "target", 2)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{"role": "user", "content": "question"}]
        assert kwargs == {"tokenize": True, "return_dict": False,
                          "add_generation_prompt": True, "enable_thinking": True}
        return [1, 2, 3]

    def decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return "".join({1: "<user>", 2: "question", 3: "<assistant><think>", 4: "reasoning", 5: "</think>answer", 9: "<eos>"}[i] for i in ids)


class FakeModel:
    generation_config = SimpleNamespace(eos_token_id=9)

    def __init__(self, continuation):
        self.continuation = continuation
        self.kwargs = None

    def generate(self, input_ids, attention_mask, **kwargs):
        self.kwargs = kwargs
        assert input_ids.tolist() == [[1, 2, 3]]
        assert attention_mask.tolist() == [[1, 1, 1]]
        assert not torch.is_grad_enabled()
        return torch.cat([input_ids, torch.tensor([self.continuation])], dim=1)


def test_generated_trace_preserves_prompt_reasoning_answer_and_eos(tmp_path):
    model = FakeModel([4, 5, 9])
    record = generate_record(FakeTokenizer(), model, "question", model_id="target", device="cpu", max_new_tokens=3)
    assert record["input_ids"] == [1, 2, 3, 4, 5, 9]
    assert record["prompt_token_count"] == 3
    assert record["generated_token_count"] == 3
    assert record["formatted_prompt"].endswith("<assistant><think>")
    assert record["generated_text"] == "reasoning</think>answer<eos>"
    assert not record["hit_generation_limit"]
    assert "temperature" not in model.kwargs
    assert model.kwargs["do_sample"] is False
    samples, _ = load_calibration(write_jsonl(tmp_path, [record]), "target", 6)
    assert samples[0]["input_ids"].tolist() == [[1, 2, 3, 4, 5, 9]]


def test_prepare_with_real_fast_tokenizer_preserves_tokens_and_provenance(tmp_path, monkeypatch):
    """Exercise Transformers' real BatchEncoding default without any downloads."""
    import transformers
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import BatchEncoding, PreTrainedTokenizerFast

    vocab = {token: index for index, token in enumerate([
        "[UNK]", "<user>", "question", "<assistant>", "<think>",
        "reasoning", "</think>", "answer", "<eos>", "<pad>",
    ])}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="<pad>", eos_token="<eos>",
        additional_special_tokens=["<user>", "<assistant>", "<think>", "</think>"],
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ '<user> ' + message['content'] + ' ' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ '<assistant> ' }}"
        "{% if enable_thinking %}{{ '<think>' }}{% endif %}{% endif %}"
    )
    default_output = tokenizer.apply_chat_template(
        [{"role": "user", "content": "question"}],
        tokenize=True, add_generation_prompt=True, enable_thinking=True,
    )
    assert isinstance(default_output, BatchEncoding)
    assert default_output["input_ids"] == [1, 2, 3, 4]

    class OfflineModel:
        dtype = torch.float32
        generation_config = SimpleNamespace(eos_token_id=8)

        def to(self, device):
            assert device == "cpu"
            return self

        def eval(self):
            return self

        def generate(self, input_ids, attention_mask, **kwargs):
            assert input_ids.tolist() == [[1, 2, 3, 4]]
            assert attention_mask.tolist() == [[1, 1, 1, 1]]
            assert not torch.is_grad_enabled()
            assert kwargs["max_new_tokens"] == 4
            return torch.cat((input_ids, torch.tensor([[5, 6, 7, 8]])), dim=1)

    calls = []

    def loader(kind, result):
        def from_pretrained(model_id, **kwargs):
            calls.append((kind, model_id, kwargs))
            return result
        return SimpleNamespace(__name__=f"Offline{kind}Loader", from_pretrained=from_pretrained)

    config = SimpleNamespace(model_type="qwen3_5", _commit_hash="offline-pinned-sha",
                             quantization_config=None)
    monkeypatch.setattr(transformers, "AutoConfig", loader("config", config))
    monkeypatch.setattr(transformers, "AutoTokenizer", loader("tokenizer", tokenizer))
    monkeypatch.setattr(transformers, "AutoModelForImageTextToText", loader("model", OfflineModel()))
    prompt_path = write_jsonl(tmp_path, [{"question": "question"}])
    output_path = tmp_path / "prepared.jsonl"
    args = add_arguments(argparse.ArgumentParser()).parse_args([
        "--model", "offline-target", "--revision", "requested-revision",
        "--prompts", str(prompt_path), "--output", str(output_path),
        "--device", "cpu", "--local-files-only", "--max-new-tokens", "4",
    ])
    manifest = run(args)
    record = json.loads(output_path.read_text())
    assert record["model_id"] == "offline-target"
    assert record["resolved_revision"] == manifest["resolved_revision"] == "offline-pinned-sha"
    assert record["input_ids"] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert record["attention_mask"] == [1] * 8
    assert record["prompt_token_count"] == record["generated_token_count"] == 4
    assert record["formatted_prompt"] == tokenizer.decode([1, 2, 3, 4], skip_special_tokens=False)
    assert record["generated_text"] == tokenizer.decode([5, 6, 7, 8], skip_special_tokens=False)
    assert not record["hit_generation_limit"]
    assert manifest["total_tokens"] == 8
    assert manifest["generated_tokens"] == 4
    assert [kind for kind, _, _ in calls] == ["config", "tokenizer", "model"]
    for kind, model_id, kwargs in calls:
        assert model_id == "offline-target"
        assert kwargs["revision"] == ("requested-revision" if kind == "config" else "offline-pinned-sha")
        assert kwargs["local_files_only"] is True
    samples, _ = load_calibration(output_path, "offline-target", 8, expected_revision="offline-pinned-sha")
    assert samples[0]["input_ids"].tolist() == [[1, 2, 3, 4, 5, 6, 7, 8]]


def test_generation_limit_and_explicit_sampling():
    model = FakeModel([4, 5])
    record = generate_record(FakeTokenizer(), model, "question", model_id="target", device="cpu", max_new_tokens=2, do_sample=True)
    assert record["hit_generation_limit"]
    assert model.kwargs["temperature"] == 0.6
    assert model.kwargs["top_p"] == 0.95


def test_prompt_limit_rejects_instead_of_truncating():
    model = FakeModel([4, 5])
    with pytest.raises(ValueError, match="exceeding max_prompt_tokens"):
        generate_record(FakeTokenizer(), model, "question", model_id="target", device="cpu", max_prompt_tokens=2)
    assert model.kwargs is None


def test_local_prompt_reader(tmp_path):
    path = write_jsonl(tmp_path, [{"question": " q1 "}, {"prompt": "q2"}])
    assert read_prompts(path) == [" q1 ", "q2"]
    assert read_prompts(path, max_samples=1) == [" q1 "]
    path = write_jsonl(tmp_path, [{"question": "q1", "prompt": "q2"}])
    with pytest.raises(ValueError, match="conflicting"):
        read_prompts(path)


@pytest.mark.parametrize("model_type, expected_loader", [("qwen3_5", "vision"), ("qwen3", "causal")])
def test_prepare_runtime_pins_target_revision_and_writes_manifest(tmp_path, monkeypatch, model_type, expected_loader):
    """Exercise the prepare workflow without importing HF or fetching weights."""
    prompt_path = write_jsonl(tmp_path, [{"question": "question"}])
    output_path = tmp_path / "prepared.jsonl"
    calls = []
    config = SimpleNamespace(model_type=model_type, _commit_hash="resolved-sha", quantization_config=None)
    model = FakeModel([4, 5, 9])
    model.dtype = torch.bfloat16
    model.to = lambda device: model
    model.eval = lambda: model

    class VisionLoader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("vision", model_id, kwargs))
            return model

    class CausalLoader:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            calls.append(("causal", model_id, kwargs))
            return model

    def tokenizer_loader(model_id, **kwargs):
        calls.append(("tokenizer", model_id, kwargs))
        return FakeTokenizer()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **kw: config),
        AutoModelForCausalLM=CausalLoader,
        AutoModelForImageTextToText=VisionLoader,
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
        set_seed=lambda seed: None,
    ))
    args = add_arguments(argparse.ArgumentParser()).parse_args([
        "--model", "target", "--prompts", str(prompt_path), "--output", str(output_path),
        "--device", "cpu", "--local-files-only", "--max-new-tokens", "3",
    ])
    manifest = run(args)
    assert [call[0] for call in calls] == ["tokenizer", expected_loader]
    for _, model_id, kwargs in calls:
        assert model_id == "target"
        assert kwargs["revision"] == "resolved-sha"
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
    record = json.loads(output_path.read_text())
    assert record["resolved_revision"] == "resolved-sha"
    assert record["input_ids"] == [1, 2, 3, 4, 5, 9]
    assert manifest["total_tokens"] == 6
    assert manifest["generated_tokens"] == 3
    assert manifest["truncated_tokens"] == 0
    assert json.loads(output_path.with_suffix(".jsonl.manifest.json").read_text()) == manifest


def test_prepare_does_not_overwrite_prompt_via_manifest_path(tmp_path):
    prompt_path = tmp_path / "output.jsonl.manifest.json"
    prompt_path.write_text('{"question": "question"}\n')
    args = add_arguments(argparse.ArgumentParser()).parse_args([
        "--prompts", str(prompt_path), "--output", str(tmp_path / "output.jsonl"), "--overwrite",
    ])
    with pytest.raises(ValueError, match="must differ"):
        run(args)
    assert prompt_path.read_text() == '{"question": "question"}\n'
