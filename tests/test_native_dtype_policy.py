"""Original native dtype selection: tiny CPU sources, never pretrained quality."""
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from openbonsai.adapter import tiny_model
from openbonsai.eval_backends import _byte_decoder, local_directory_identity


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "native_dtype_runner", ROOT / "tools/evaluate_transformers_manifest.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture(params=[False, True], ids=["causal", "conditional"])
def native_source(tmp_path, request):
    torch.set_num_threads(2)
    model = tiny_model(seed=981, multimodal=request.param).bfloat16().eval().requires_grad_(False)
    prefix = "model.language_model" if request.param else "model"
    name = prefix + ".layers.0.linear_attn.A_log"
    parent = model.get_submodule(prefix + ".layers.0.linear_attn")
    parent.A_log.data = torch.full_like(parent.A_log, .1234567, dtype=torch.float32)
    assert not torch.equal(parent.A_log, parent.A_log.bfloat16().float())
    model_path, tokens_path = tmp_path / "model", tmp_path / "tokenizer"
    model.save_pretrained(model_path)
    tokens = Tokenizer(models.BPE(vocab={char: i for i, char in enumerate(_byte_decoder())}, merges=[]))
    tokens.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokens.decoder = decoders.ByteLevel()
    PreTrainedTokenizerFast(tokenizer_object=tokens).save_pretrained(tokens_path)
    identities = {"model": local_directory_identity(model_path), "tokenizer": local_directory_identity(tokens_path)}
    return model, model_path, tokens_path, name, identities


def arguments(source, policy):
    _, model, tokenizer, _, identities = source
    result = SimpleNamespace(kind="original", device="cpu", cache_policy="cold",
        model=model, tokenizer=tokenizer, expected_model_sha256=identities["model"]["sha256"],
        expected_tokenizer_sha256=identities["tokenizer"]["sha256"])
    if policy is not None:
        result.original_dtype_policy = policy
    return result


def load(source, policy, output):
    output.mkdir()
    plan = {"threads": 2, "context_length": 32, "vocab_size": 256,
            "logits_chunk_size": 2, "minimum_free_cuda_bytes": 0}
    return runner.load_backend(arguments(source, policy), plan, output, time.monotonic() + 120)


def direct_probabilities(backend, ids, positions):
    with torch.no_grad(), backend._precision():
        logits = backend.model(input_ids=torch.tensor([ids]),
            attention_mask=torch.ones(1, len(ids), dtype=torch.long), use_cache=False).logits[0].float().double()
    values = []
    for position in positions:
        row = logits[position - 1]
        centered = row - row.max()
        values.append(float(centered[ids[position]] - centered.exp().sum().log()))
    return values


def test_preserved_policy_matches_exact_saved_source_legacy_keeps_cast(native_source, tmp_path):
    original, model_path, tokens_path, name, before = native_source
    preserved, execution = load(native_source, "preserve-checkpoint", tmp_path / "preserved")
    legacy, legacy_execution = load(native_source, None, tmp_path / "legacy")
    try:
        expected = original.state_dict()[name]
        actual = preserved.model.state_dict()[name]
        cast = legacy.model.state_dict()[name]
        assert actual.dtype == torch.float32 and cast.dtype == torch.bfloat16
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(cast, expected.bfloat16(), rtol=0, atol=0)
        assert not torch.equal(actual, cast.float())
        for tensor_name, tensor in original.state_dict().items():
            result = preserved.model.state_dict()[tensor_name]
            assert result.dtype == tensor.dtype
            torch.testing.assert_close(result, tensor, rtol=0, atol=0)
        assert execution["original_dtype_policy"] == "preserve-checkpoint"
        assert execution["original_load_dtype"] == "checkpoint-preserved"
        receipt = execution["original_source_load_receipt"]
        assert receipt["source_content_sha256"] == before["model"]["sha256"]
        assert receipt["restored_saved_dtype_tensors"][name] == {"dtype": "float32", "bytes": expected.numel() * 4}
        assert receipt["unused_mtp_tensors"] == {} and receipt["unused_mtp_is_not_compression"] is True
        assert legacy_execution["original_dtype_policy"] == "bf16-cast"
        assert legacy_execution["original_load_dtype"] == "bfloat16"
        assert legacy_execution["original_source_load_receipt"] is None
        stored = json.loads((tmp_path / "preserved" / "loaded-source.json").read_text())
        assert stored["execution"] == execution
        assert stored["source"] == before["model"]
        ids, positions = [10, 20, 30, 40, 50], [1, 4]
        for backend in (preserved, legacy):
            observed = backend.score_tokens(ids, positions)
            assert observed["target_logprobs"] == pytest.approx(direct_probabilities(backend, ids, positions), abs=1e-7)
            assert observed["score_positions"] == positions and observed["count"] == 2
        # No claim that rounding these particular tiny values causes a measurable
        # final-score difference; source state identity is the controlled property.
        assert local_directory_identity(model_path) == before["model"]
        assert local_directory_identity(tokens_path) == before["tokenizer"]
    finally:
        preserved.close()
        legacy.close()


def test_explicit_legacy_option_is_same_as_default(native_source, tmp_path):
    default, default_execution = load(native_source, None, tmp_path / "default")
    explicit, explicit_execution = load(native_source, "bf16-cast", tmp_path / "explicit")
    try:
        assert default_execution == explicit_execution
        for name, value in default.model.state_dict().items():
            torch.testing.assert_close(value, explicit.model.state_dict()[name], rtol=0, atol=0)
    finally:
        default.close()
        explicit.close()


@pytest.mark.parametrize("kind,policy", [("original", "auto"), ("original", None), ("mixed", "preserve-checkpoint")])
def test_unsupported_or_inapplicable_policy_fails_before_file_access(kind, policy):
    args = SimpleNamespace(kind=kind, device="cpu", cache_policy="cold", original_dtype_policy=policy)
    with pytest.raises(ValueError, match="dtype policy|only to original"):
        runner.check_source_inputs(args)


def test_mixed_default_policy_does_not_change_stored_dtype_contract():
    assert runner.original_dtype_policy(SimpleNamespace(kind="mixed")) is None
    assert runner.original_dtype_policy(SimpleNamespace(kind="mixed", original_dtype_policy="bf16-cast")) is None
