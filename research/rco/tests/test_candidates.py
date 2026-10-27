import hashlib

import pytest
import torch
from torch import nn

from openbonsai.adapter import tiny_model
from openbonsai.candidates import (
    InputHessian, collect_input_hessians, eligible_mixed_modules,
    fit_embedding_rtn_candidates, fit_gptq_candidates,
    hessian_from_activations, source_weight_sha256,
)
from openbonsai._vendor.rco.quant.fast_obq import FastOBQ
from openbonsai._vendor.rco.quant import linalg_utils
from openbonsai.mixed_formats import candidate_weight, encode_candidate, decode_candidate


def fixture():
    generator = torch.Generator().manual_seed(15)
    weight = torch.randn((5, 16), generator=generator).to(torch.bfloat16)
    inputs = torch.randn((30, 16), generator=generator)
    inputs[:, 1:] += 1.8 * inputs[:, :1]
    return weight, 2 * inputs.T @ inputs / inputs.shape[0]


def test_gptq_matches_pinned_upstream_and_preserves_sources():
    weight, hessian = fixture()
    original, old_h = weight.clone(), hessian.clone()
    fitted = fit_gptq_candidates(weight, hessian, group_size=8, row_chunk=2)
    layer = nn.Linear(16, 5, bias=False)
    layer.weight = nn.Parameter(weight.float(), requires_grad=False)
    upstream = FastOBQ(layer, [4, 6], group_size=8, block_size=8, sym=False)
    upstream.H = hessian.clone()
    codes, scales, zeros, perm = upstream.quantize([4, 6])
    assert perm is None
    for bit, candidate in fitted.items():
        expected = ((codes[bit].float().reshape(5, 2, 8) - zeros[bit][..., None])
                    * scales[bit][..., None]).reshape(5, 16)
        assert torch.equal(candidate_weight(candidate), expected)
        assert torch.equal(candidate_weight(decode_candidate(encode_candidate(candidate))), expected)
        assert candidate.tensors["scales"].dtype == torch.float32
        assert candidate.metadata["source"] == "gptq"
        assert candidate.metadata["provenance"]["source_weight_sha256"] == source_weight_sha256(weight)
    assert torch.equal(weight, original)
    assert torch.equal(hessian, old_h)


def test_correlation_changes_gptq_choices():
    weight, hessian = fixture()
    correlated = fit_gptq_candidates(weight, hessian, bits=(4,), group_size=8)[4]
    diagonal = fit_gptq_candidates(weight, torch.eye(16), bits=(4,), group_size=8)[4]
    assert not torch.equal(candidate_weight(correlated), candidate_weight(diagonal))
    assert correlated.metadata["provenance"]["weight_reconstruction_mse"] >= 0


def test_cholesky_is_shared_across_row_chunks(monkeypatch):
    weight, hessian = fixture()
    original = linalg_utils.inv_sym
    calls = []
    def count(matrix):
        calls.append(matrix.shape)
        return original(matrix)
    monkeypatch.setattr(linalg_utils, "inv_sym", count)
    fit_gptq_candidates(weight, hessian, group_size=8, row_chunk=1)
    assert calls == [(16, 16)]


def test_embedding_rtn_preserves_negative_zero_points_and_labels():
    weight = torch.linspace(1, 2, 16).reshape(2, 8).to(torch.bfloat16)
    original = weight.clone()
    candidate = fit_embedding_rtn_candidates(weight, group_size=8, bits=(4,), row_chunk=1)[4]
    assert candidate.metadata["source"] == "embedding-rtn"
    assert candidate.module_kind == "embedding"
    assert bool((candidate.tensors["zeros"] < 0).all())
    assert torch.equal(weight, original)
    assert torch.equal(candidate_weight(decode_candidate(encode_candidate(candidate))), candidate_weight(candidate))
    assert candidate.metadata["provenance"]["calibration"] == "none-identity-weight-error-baseline"


def test_hash_is_original_row_major_bf16_bytes():
    weight, _ = fixture()
    expected = hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest()
    assert source_weight_sha256(weight, row_chunk=1) == expected
    with pytest.raises(ValueError, match="does not match"):
        fit_embedding_rtn_candidates(weight, group_size=8, provenance={"source_weight_sha256": "bad"})


def test_explicit_activations_and_mask():
    inputs = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    mask = torch.tensor([[1, 0, 1], [0, 1, 1]])
    stats = hessian_from_activations(inputs, attention_mask=mask, max_tokens=3)
    chosen = inputs.reshape(6, 4)[[0, 2, 4]]
    assert stats.token_count == 3
    torch.testing.assert_close(stats.matrix, 2 * chosen.T @ chosen / 3, rtol=1e-6, atol=1e-5)


class OriginalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
    def forward(self, input_ids, attention_mask=None):
        return self.linear(input_ids)


def test_capture_is_bounded_masked_and_restores_modes():
    model = OriginalModel()
    model.train()
    model.linear.eval()
    weight = model.linear.weight.detach().clone()
    x = torch.arange(24).reshape(2, 3, 4).to(torch.bfloat16)
    mask = torch.tensor([[1, 0, 1], [0, 1, 1]])
    stats = collect_input_hessians(model, [{"input_ids": x, "attention_mask": mask}],
                                   ["linear"], max_tokens=3)["linear"]
    chosen = x.reshape(6, 4)[[0, 2, 4]].float()
    assert stats.token_count == 3
    torch.testing.assert_close(stats.matrix, 2 * chosen.T @ chosen / 3, rtol=1e-6, atol=1e-5)
    assert model.training and not model.linear.training
    assert torch.equal(model.linear.weight, weight)
    assert not model.linear._forward_pre_hooks


def test_capture_failure_removes_hooks():
    model = OriginalModel()
    with pytest.raises(ValueError, match="mask does not match"):
        collect_input_hessians(model, [{"input_ids": torch.ones(1, 2, 4, dtype=torch.bfloat16),
                                       "attention_mask": torch.ones(1, 3)}], ["linear"])
    assert not model.linear._forward_pre_hooks
    assert model.training


@pytest.mark.parametrize("bad", [torch.ones(4, 16), torch.ones(4, 15, dtype=torch.bfloat16)])
def test_reject_wrong_source_dtype_or_group_shape(bad):
    with pytest.raises(ValueError):
        fit_gptq_candidates(bad, torch.eye(bad.shape[1]), group_size=8)


def test_reject_uninformative_or_nonfinite_hessian():
    weight, _ = fixture()
    with pytest.raises(ValueError, match="informative"):
        fit_gptq_candidates(weight, torch.zeros(16, 16), group_size=8)
    with pytest.raises(ValueError, match="informative"):
        fit_gptq_candidates(weight, torch.full((16, 16), float("nan")), group_size=8)


@pytest.mark.parametrize("multimodal", [False, True])
def test_native_eligibility_excludes_vision_and_protects_recurrence(multimodal):
    model = tiny_model(multimodal=multimodal).to(torch.bfloat16)
    choices = eligible_mixed_modules(model)
    assert sum(kind == "embedding" for kind in choices.values()) == 1
    assert "lm_head" in choices
    assert not any("visual" in name or "vision" in name or "in_proj_a" in name or "in_proj_b" in name
                   for name in choices)
    assert len(eligible_mixed_modules(model, aggressive_gates=True)) == len(choices) + 6
