"""Complete table fitting, gradient normalization, and export eligibility."""
import copy
import math

import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from openbonsai import embedding_calibration as ec
from openbonsai.embedding_calibration import (
    EmbeddingTrainingConfig, calibration_state_sha256, evaluate_embedding_table,
    fit_embedding_table, require_complete_calibration,
)
from openbonsai.embedding_quantizer import BoundedTernaryEmbedding
from openbonsai.hadamard import inverse_rotate
from openbonsai.quantizer import TernaryLinear
from openbonsai.rotations import RotatedTernaryLinear, fixed_signs


SOURCE = {"source_checkpoint": "original-BF16-fixture", "source_revision": "fixture-seed-730"}


def fixture(*, rotated=True, rank=2, mapping="sigmoid", chunk=3, rows=7, width=8):
    torch.manual_seed(730)
    source = nn.Embedding(rows, width, padding_idx=1, dtype=torch.bfloat16)
    source.requires_grad_(False)
    with torch.no_grad():
        source.weight.normal_(std=.3)  # A nonzero padding row must also be fitted.
    embedding = BoundedTernaryEmbedding(source, group_size=4, row_chunk_size=chunk,
        signs=fixed_signs(width, seed=12) if rotated else None, block_size=4,
        lora_rank=rank, factor_mapping=mapping)
    with torch.no_grad():
        for parameter in embedding.factor_parameters():
            parameter.normal_(std=.03)
        if rank:
            embedding.quantizer.lora_B.normal_(std=.02)
    return source, embedding


def dense_reference(source, embedding):
    linear = nn.Linear(source.embedding_dim, source.num_embeddings, bias=False, device="meta")
    linear.weight = nn.Parameter(source.weight.detach(), requires_grad=False)
    kwargs = dict(group_size=4, lora_rank=embedding.lora_rank, factor_mapping=embedding.factor_mapping)
    dense = (TernaryLinear(linear, **kwargs) if embedding.rotation_signs is None else
             RotatedTernaryLinear(linear, signs=embedding.rotation_signs, block_size=4, **kwargs))
    with torch.no_grad():
        for wanted, current in zip(dense.trainable_parameters(), embedding.trainable_parameters()):
            wanted.copy_(current)
    return dense


@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("mapping", ["exp", "sigmoid"])
@pytest.mark.parametrize("rank,loss_name", [(0, "mse"), (2, "mse"), (2, "huber")])
def test_complete_streamed_gradients_and_adam_updates_match_dense(monkeypatch, rotated, mapping, rank, loss_name):
    torch.set_num_threads(2)
    source, embedding = fixture(rotated=rotated, rank=rank, mapping=mapping)
    dense = dense_reference(source, embedding)
    config = EmbeddingTrainingConfig(steps=3, factor_lr=.002, lora_lr=.001,
                                    weight_decay=.02, grad_clip=.2, loss=loss_name, huber_delta=.08)
    groups = [{"params": list(dense.factor_parameters()), "lr": config.factor_lr}]
    if rank:
        groups.append({"params": list(dense.lora_parameters()), "lr": config.lora_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay, foreach=False)
    expected = []
    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        dense.set_progress(step / config.steps)
        weight = dense.quantized_weight()
        if rotated:
            weight = inverse_rotate(weight, dense.rotation_signs, dense.block_size)
        target = source.weight.float()
        loss = ((weight.float() - target).square().mean() if loss_name == "mse" else
                F.huber_loss(weight.float(), target, delta=config.huber_delta))
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(dense.trainable_parameters(), config.grad_clip, foreach=False)
        gradients = [p.grad.clone() for p in dense.trainable_parameters()]
        optimizer.step()
        expected.append((float(loss.detach()), float(norm), gradients, [p.detach().clone() for p in dense.trainable_parameters()]))
    original_step = torch.optim.AdamW.step
    observations = []
    def observed_step(optimizer, *args, **kwargs):
        index = len(observations)
        params = [p for group in optimizer.param_groups for p in group["params"]]
        for actual, wanted in zip(params, expected[index][2]):
            assert actual.grad.dtype == torch.float32
            torch.testing.assert_close(actual.grad, wanted, atol=3e-7, rtol=3e-4)
        result = original_step(optimizer, *args, **kwargs)
        for actual, wanted in zip(params, expected[index][3]):
            torch.testing.assert_close(actual, wanted, atol=3e-6, rtol=3e-4)
        observations.append(index)
        return result
    monkeypatch.setattr(torch.optim.AdamW, "step", observed_step)
    result = fit_embedding_table(embedding, config, source_provenance=SOURCE)
    assert len(observations) == config.steps
    for event, (loss, norm, _, _) in zip(result.report["steps"], expected):
        assert event["loss"] == pytest.approx(loss, rel=3e-6, abs=1e-8)
        assert event["grad_norm_before_clip"] == pytest.approx(norm, rel=3e-5, abs=1e-7)
    assert result.report["objective"]["denominator"] == 56


def test_each_update_covers_all_rows_snapshot_captured_once_and_padding_error_reported(monkeypatch):
    source, embedding = fixture()
    original = source.weight.clone()
    begins, rows_by_step, saved = [], {}, {}
    begin = embedding.begin_step
    reconstruct = embedding.reconstruct_rows
    def tracked_begin(progress):
        begins.append(progress)
        begin(progress)
        if progress <= embedding.soft_fraction:
            saved.clear()
            saved.update({name: None if getattr(embedding.quantizer, name) is None else getattr(embedding.quantizer, name).clone()
                          for name in embedding.quantizer._snapshot_names()})
    def tracked_rows(ids):
        if torch.is_grad_enabled():
            rows_by_step.setdefault(len(begins), []).extend(ids.tolist())
            for name, expected in saved.items():
                actual = getattr(embedding.quantizer, name)
                assert actual is None if expected is None else torch.equal(actual, expected)
        return reconstruct(ids)
    monkeypatch.setattr(embedding, "begin_step", tracked_begin)
    monkeypatch.setattr(embedding, "reconstruct_rows", tracked_rows)
    events = []
    result = fit_embedding_table(embedding, EmbeddingTrainingConfig(steps=5), source_provenance=SOURCE,
                                progress_callback=events.append)
    assert begins == [.2, .4, .6, .8, 1.]
    assert all(row_ids == list(range(7)) for row_ids in rows_by_step.values())
    assert len(rows_by_step) == 5 and len(events) == 5
    assert source.weight.grad is None and torch.equal(source.weight, original)
    assert all(p.dtype == torch.float32 and p.grad is None for p in embedding.trainable_parameters())
    report = result.report
    assert report["coverage"] == {**report["coverage"], "soft_rows": 7, "hard_rows": 7,
        "soft_visits_min": 4, "soft_visits_max": 4, "hard_visits_min": 1, "hard_visits_max": 1,
        "last_soft_step": 4, "final_hard_step": 5, "last_soft_rows": 7, "final_hard_rows": 7}
    assert not report["quality_claim"] and report["final_hard_reconstruction"]["role"] == "training-reconstruction"
    with torch.no_grad():
        pad_error = (embedding.reconstruct_rows(torch.tensor([1])).float() - source.weight[1].float()).square().mean()
    assert report["final_hard_reconstruction"]["padding_row_mse"] == pytest.approx(float(pad_error))
    assert report["timing_seconds"]["total"] == pytest.approx(sum(v for k, v in report["timing_seconds"].items() if k != "total"))
    assert all(math.isfinite(value) and value >= 0 for value in report["timing_seconds"].values())
    assert report["certificate"] == require_complete_calibration(embedding)
    assert len(ec._json_bytes(report["certificate"])) < 8192


@pytest.mark.parametrize("chunk", [1, 2, 4, 7])
def test_chunk_size_does_not_change_loss_weighting_or_optimizer_schedule(chunk):
    _, dense_chunk = fixture(chunk=7)
    _, streamed = fixture(chunk=chunk)
    config = EmbeddingTrainingConfig(steps=3)
    expected = fit_embedding_table(dense_chunk, config, source_provenance=SOURCE)
    actual = fit_embedding_table(streamed, config, source_provenance=SOURCE)
    for a, b in zip(dense_chunk.trainable_parameters(), streamed.trainable_parameters()):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-4)
    for a, b in zip(expected.report["steps"], actual.report["steps"]):
        assert a["loss"] == pytest.approx(b["loss"], rel=2e-6, abs=1e-8)
        assert a["rows"] == b["rows"] == 7
    assert expected.report["coverage"] == actual.report["coverage"]


@pytest.mark.parametrize("mutation", ["factor", "factor_data", "lora", "stats", "sign", "block", "mapping", "padding", "progress", "source_data"])
def test_certificate_rejects_stale_source_configuration_and_compact_state(mutation):
    source, embedding = fixture(rank=0 if mutation == "stats" else 2)
    fit_embedding_table(embedding, EmbeddingTrainingConfig(steps=2), source_provenance=SOURCE)
    certificate = require_complete_calibration(embedding)
    certificate["source_provenance"]["source_revision"] = "tampered-copy"
    assert require_complete_calibration(embedding)["source_provenance"] == SOURCE
    with torch.no_grad():
        if mutation == "factor": embedding.quantizer.raw_scale[0].add_(.1)
        elif mutation == "factor_data": embedding.quantizer.raw_mu.data[0].add_(.1)
        elif mutation == "lora": embedding.quantizer.lora_A[0, 0].add_(.1)
        elif mutation == "stats": embedding.quantizer.alpha0[0].add_(.1)
        elif mutation == "sign": embedding.quantizer.rotation_signs[0].neg_()
        elif mutation == "block": embedding.quantizer.block_size = 8
        elif mutation == "mapping": embedding.quantizer.factor_mapping = "exp"
        elif mutation == "padding": embedding.padding_idx = 0
        elif mutation == "progress": embedding.progress = .5
        elif mutation == "source_data": source.weight.data[0, 0].add_(1)
    with pytest.raises(RuntimeError, match="stale|incomplete|source hash changed"):
        require_complete_calibration(embedding)


def test_fingerprint_is_bounded_chunk_independent_and_binds_stats_without_source(monkeypatch):
    _, embedding = fixture(rows=11, width=4096, chunk=2, rank=2)
    expected = calibration_state_sha256(embedding)
    cpu = torch.Tensor.cpu
    def bounded_cpu(value, *args, **kwargs):
        assert value.numel() <= 4096
        return cpu(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "cpu", bounded_cpu)
    assert calibration_state_sha256(embedding, chunk_elements=13) == expected
    with torch.no_grad():
        embedding.quantizer.alpha0[0].add_(.01)
    assert calibration_state_sha256(embedding, chunk_elements=13) != expected


@pytest.mark.parametrize("failure", ["missing_row", "duplicate_row", "nonfinite_output", "nonfinite_gradient", "callback_param", "callback_snapshot", "callback_source", "callback_exception"])
def test_failed_or_incomplete_fit_never_leaves_prior_certificate(monkeypatch, failure):
    source, embedding = fixture()
    fit_embedding_table(embedding, EmbeddingTrainingConfig(steps=2), source_provenance=SOURCE)
    spans = ec._row_spans
    reconstruct = embedding.reconstruct_rows
    if failure in {"missing_row", "duplicate_row"}:
        def incorrect_spans(module):
            values = list(spans(module))
            if torch.is_grad_enabled():
                values = values[:-1] if failure == "missing_row" else values + [values[0]]
            yield from values
        monkeypatch.setattr(ec, "_row_spans", incorrect_spans)
    elif failure == "nonfinite_output":
        monkeypatch.setattr(embedding, "reconstruct_rows", lambda ids: reconstruct(ids) * float("nan"))
    elif failure == "nonfinite_gradient":
        embedding.quantizer.raw_scale.register_hook(lambda value: value * float("nan"))
    def callback(event):
        if failure == "callback_param": embedding.quantizer.raw_scale.data[0].add_(.1)
        elif failure == "callback_snapshot": embedding.quantizer._hard_snapshot_scale.data[0].add_(.1)
        elif failure == "callback_source": source.weight.data[0, 0].add_(1)
        elif failure == "callback_exception": raise RuntimeError("caller interrupted")
    with pytest.raises((RuntimeError, FloatingPointError, ValueError)):
        fit_embedding_table(embedding, EmbeddingTrainingConfig(steps=3), source_provenance=SOURCE,
                            progress_callback=callback)
    assert getattr(embedding, ec._CERTIFICATE_ATTRIBUTE) is None
    assert all(p.grad is None for p in embedding.trainable_parameters())
    with pytest.raises(RuntimeError, match="missing|source"):
        require_complete_calibration(embedding)


@pytest.mark.parametrize("provenance", [{}, {"source_checkpoint": "x"}, {"source_checkpoint": "x", "source_revision": "main"},
    {**SOURCE, "source_weight_sha256": "0" * 64}, {**SOURCE, "source_content_sha256": "wrong"}])
def test_missing_or_conflicting_source_identity_rejected_and_certificate_cleared(provenance):
    _, embedding = fixture()
    fit_embedding_table(embedding, EmbeddingTrainingConfig(steps=2), source_provenance=SOURCE)
    with pytest.raises(ValueError):
        fit_embedding_table(embedding, EmbeddingTrainingConfig(steps=2), source_provenance=provenance)
    assert getattr(embedding, ec._CERTIFICATE_ATTRIBUTE) is None


@pytest.mark.parametrize("kwargs", [{"steps": 0}, {"steps": 1}, {"steps": True}, {"steps": 1.5},
    {"factor_lr": 0}, {"lora_lr": float("nan")}, {"grad_clip": -1}, {"weight_decay": -1},
    {"loss": "latent"}, {"huber_delta": 0}])
def test_invalid_training_budget_and_objective_rejected(kwargs):
    _, embedding = fixture()
    with pytest.raises(ValueError):
        fit_embedding_table(embedding, EmbeddingTrainingConfig(**kwargs), source_provenance=SOURCE)
    assert getattr(embedding, ec._CERTIFICATE_ATTRIBUTE) is None


def test_evaluation_does_not_capture_or_mutate_snapshot_and_content_hash_identity_accepted():
    _, embedding = fixture()
    embedding.begin_step(.4)
    before = calibration_state_sha256(embedding, include_snapshots=True)
    evaluation = evaluate_embedding_table(embedding, loss="huber", huber_delta=.1)
    assert calibration_state_sha256(embedding, include_snapshots=True) == before
    assert evaluation["rows"] == 7 and evaluation["loss_name"] == "huber"
    result = fit_embedding_table(embedding, EmbeddingTrainingConfig(steps=2),
        source_provenance={"source_checkpoint": "local-finetune", "source_content_sha256": "a" * 64})
    assert result.report["source_provenance"]["source_content_sha256"] == "a" * 64


def test_callback_cannot_change_the_declared_training_schedule_or_loss():
    _, expected = fixture()
    _, actual = fixture()
    configuration = EmbeddingTrainingConfig(steps=3)
    reference = fit_embedding_table(expected, copy.deepcopy(configuration), source_provenance=SOURCE)
    def change_callers_config(event):
        configuration.steps = 99
        configuration.loss = "huber"
        configuration.huber_delta = .0001
        configuration.factor_lr = .2
    result = fit_embedding_table(actual, configuration, source_provenance=SOURCE,
                                 progress_callback=change_callers_config)
    assert result.report["config"] == reference.report["config"]
    assert result.report["certificate"]["training_config"] == reference.report["config"]
    assert len(result.report["steps"]) == 3
    for first, second in zip(expected.parameters(), actual.parameters()):
        assert torch.equal(first, second)


class NoFullFP32Table(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for value in tree_flatten(result)[0]:
            if isinstance(value, Tensor) and value.dtype == torch.float32 and value.device.type != "meta":
                assert value.numel() != 11 * 4096, f"Full FP32 table in {func}"
        return result


def test_complete_training_at_native_width_never_allocates_full_float_table():
    with NoFullFP32Table():
        _, embedding = fixture(rows=11, width=4096, chunk=2, rank=2)
        result = fit_embedding_table(embedding, EmbeddingTrainingConfig(steps=2), source_provenance=SOURCE)
        assert require_complete_calibration(embedding) == result.report["certificate"]


def test_tiny_native_embedding_only_fit_preserves_backbone_head_source_and_inference():
    from openbonsai.adapter import tiny_model
    model = tiny_model(seed=8).bfloat16()
    original = model.get_input_embeddings()
    original_values = original.weight.clone()
    embedding = BoundedTernaryEmbedding(original, group_size=32, row_chunk_size=64,
        signs=fixed_signs(128, seed=8), block_size=128, lora_rank=2)
    retained = {name: p.clone() for name, p in model.named_parameters() if p is not original.weight}
    model.set_input_embeddings(embedding)
    result = fit_embedding_table(embedding, EmbeddingTrainingConfig(steps=2),
        source_provenance={"source_checkpoint": "tiny-native", "source_revision": "seed8"})
    assert torch.equal(original.weight, original_values)
    for name, expected in retained.items():
        assert torch.equal(model.get_parameter(name), expected)
        assert model.get_parameter(name).grad is None
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([[3, 7, 9, 3]]), use_cache=False).logits
    assert torch.isfinite(logits).all()
    assert result.report["coverage"]["vocabulary_rows"] == original.num_embeddings
    assert require_complete_calibration(embedding)
