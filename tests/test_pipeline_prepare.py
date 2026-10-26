"""Preparation provenance and failure behavior; no pretrained quality claim."""
import copy
import json

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from openbonsai.adapter import tiny_model
from openbonsai.eval_backends import _byte_decoder, _identity, local_directory_identity
from openbonsai.pipeline_prepare import prepare_bound, validate_generation
from openbonsai.pipeline_sources import tokenizer_identity


def inputs(tmp_path):
    torch.set_num_threads(2)
    source, tokenizer = tmp_path / "source", tmp_path / "tokenizer"
    tiny_model(seed=88, multimodal=True).bfloat16().eval().save_pretrained(source)
    backend = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(_byte_decoder())}, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=backend)
    tok.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    tok.save_pretrained(tokenizer)
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(json.dumps({"prompt": "ABC"}) + "\n")
    return dict(model_directory=str(source), expected_model_sha256=local_directory_identity(source)["sha256"],
        tokenizer_directory=str(tokenizer), expected_tokenizer_sha256=tokenizer_identity(tokenizer)["sha256"],
        prompts=str(prompts), expected_prompts_sha256=_identity(prompts)["sha256"],
        generation=dict(seed=5, max_new_tokens=2, max_prompt_tokens=8, max_samples=1,
                        enable_thinking=False, do_sample=False, temperature=.6, top_p=.95),
        role="calibration", output=tmp_path / "prepared")


def test_actual_original_preparation_binds_source_and_exact_generated_records(tmp_path):
    kwargs = inputs(tmp_path)
    before = local_directory_identity(kwargs["model_directory"])
    events = []
    receipt = prepare_bound(**kwargs, progress_callback=events.append)
    row = json.loads((kwargs["output"] / "traces.jsonl").read_bytes())
    assert receipt["status"] == "complete" and receipt["counts"]["records"] == 1
    assert row["source_content_sha256"] == before["sha256"]
    assert row["prompt_token_count"] == 3 and 1 <= row["generated_token_count"] <= 2
    assert len(row["input_ids"]) == row["prompt_token_count"] + row["generated_token_count"]
    assert receipt["source_jsonl_sha256"] == _identity(kwargs["output"] / "traces.jsonl")["sha256"]
    assert receipt["tokenizer_content_sha256"] == tokenizer_identity(kwargs["tokenizer_directory"])["sha256"]
    assert local_directory_identity(kwargs["model_directory"]) == before
    assert events[-1]["counts"]["records"] == 1 and receipt["quality_claim"] is False


@pytest.mark.parametrize("field", ["expected_model_sha256", "expected_tokenizer_sha256", "expected_prompts_sha256"])
def test_changed_input_rejected_before_model_load_or_output(tmp_path, monkeypatch, field):
    kwargs = inputs(tmp_path)
    kwargs[field] = "0" * 64
    monkeypatch.setattr("openbonsai.pipeline_prepare.load_original_source", lambda *_a, **_k: pytest.fail("must not load"))
    with pytest.raises(ValueError):
        prepare_bound(**kwargs)
    assert not kwargs["output"].exists()


def test_impossible_context_limit_rejected_before_loading_or_output(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path)
    kwargs["generation"]["max_prompt_tokens"] = 10**9
    monkeypatch.setattr("openbonsai.pipeline_prepare.load_original_source", lambda *_a, **_k: pytest.fail("must not load"))
    with pytest.raises(ValueError, match="context"):
        prepare_bound(**kwargs)
    assert not kwargs["output"].exists()


def test_interrupted_preparation_has_no_complete_producer(tmp_path):
    kwargs = inputs(tmp_path)
    def stop(event):
        raise TimeoutError("bounded attempt")
    with pytest.raises(TimeoutError):
        prepare_bound(**kwargs, progress_callback=stop)
    assert (kwargs["output"] / "traces.jsonl").is_file()
    assert not (kwargs["output"] / "producer.json").exists()


@pytest.mark.parametrize("change", [{"seed": True}, {"max_new_tokens": 0}, {"temperature": float("nan")},
                                   {"top_p": 2}, {"do_sample": 1}, {"max_samples": None}, {"extra": 5}])
def test_generation_configuration_has_no_implicit_recipe(change):
    generation = dict(seed=5, max_new_tokens=2, max_prompt_tokens=8, max_samples=1,
                      enable_thinking=False, do_sample=False, temperature=.6, top_p=.95)
    generation.update(change)
    with pytest.raises(ValueError):
        validate_generation(generation)
