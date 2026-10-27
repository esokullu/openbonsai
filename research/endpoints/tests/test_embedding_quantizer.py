"""Embedding operator invariants on CPU; no pretrained quality claims."""
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

import openbonsai.endpoint_quantizer as eq
from openbonsai.embedding_quantizer import BoundedTernaryEmbedding
from openbonsai.hadamard import inverse_rotate
from openbonsai.quantizer import TernaryLinear
from openbonsai.rotations import RotatedTernaryLinear, fixed_signs


class DenseEmbeddingOracle(nn.Module):
    def __init__(self, source, *, rank, mapping, rotated, hard_phase='frozen_jacobian'):
        super().__init__()
        linear = nn.Linear(source.embedding_dim, source.num_embeddings, bias=False, device='meta')
        linear.weight = nn.Parameter(source.weight.detach(), requires_grad=False)
        kwargs = dict(group_size=4, lora_rank=rank, factor_mapping=mapping, hard_phase=hard_phase)
        self.core = (RotatedTernaryLinear(linear, signs=fixed_signs(16, seed=5), block_size=8, **kwargs)
                     if rotated else TernaryLinear(linear, **kwargs))
        self.rotated, self.padding_idx = rotated, source.padding_idx

    def forward(self, ids):
        c = self.core
        if not c.quant_enabled or c.progress == 0:
            return F.embedding(ids, c.weight, self.padding_idx)
        result = F.embedding(ids, c.quantized_weight(), self.padding_idx)
        return inverse_rotate(result, c.rotation_signs, c.block_size) if self.rotated else result


def paired(*, rank=2, mapping='sigmoid', rotated=True, padding_idx=3, hard_phase='frozen_jacobian'):
    torch.manual_seed(921)
    source = nn.Embedding(11, 16, padding_idx=padding_idx, dtype=torch.bfloat16)
    source.requires_grad_(False)
    # Native padding_idx suppresses gradients; it need not store an all-zero row.
    with torch.no_grad():
        source.weight.normal_(std=.2)
    dense = DenseEmbeddingOracle(source, rank=rank, mapping=mapping, rotated=rotated, hard_phase=hard_phase)
    bounded = BoundedTernaryEmbedding(source, group_size=4, row_chunk_size=2,
        signs=fixed_signs(16, seed=5) if rotated else None, block_size=8,
        lora_rank=rank, factor_mapping=mapping, hard_phase=hard_phase)
    with torch.no_grad():
        for p in dense.core.factor_parameters():
            p.normal_(std=.06)
        if rank:
            dense.core.lora_B.normal_(std=.05)
        for a, b in zip(dense.core.trainable_parameters(), bounded.trainable_parameters()):
            b.copy_(a)
    ids = torch.tensor([[10, 2, 3, 10, 0, 7, 2], [4, 7, 10, 3, 4, 0, 2]])
    return source, dense, bounded, ids


@pytest.mark.parametrize('mapping', ['exp', 'sigmoid'])
@pytest.mark.parametrize('rank', [0, 2])
@pytest.mark.parametrize('rotated', [False, True])
@pytest.mark.parametrize('phase', ['soft', 'hard'])
@pytest.mark.parametrize('padding_idx', [None, 3])
def test_sparse_values_all_gradients_and_adam_update_match_dense(mapping, rank, rotated, phase, padding_idx):
    source, dense, bounded, ids = paired(rank=rank, mapping=mapping, rotated=rotated, padding_idx=padding_idx)
    original = source.weight.clone()
    dense.core.set_progress(.23)
    bounded.begin_step(.23)
    dense(ids)
    snapshots = {n: None if getattr(bounded.quantizer, n) is None else getattr(bounded.quantizer, n).clone()
                 for n in bounded.quantizer._snapshot_names()}
    if phase == 'hard':
        with torch.no_grad():
            for i, (a, b) in enumerate(zip(dense.core.trainable_parameters(), bounded.trainable_parameters())):
                a.add_(.013 * (i + 1)); b.copy_(a)
        dense.core.set_progress(.93)
        bounded.begin_step(.93)
    expected, actual = dense(ids), bounded(ids)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(actual)
    (expected.float() * upstream.float()).sum().backward()
    (actual.float() * upstream.float()).sum().backward()
    for a, b in zip(dense.core.trainable_parameters(), bounded.trainable_parameters()):
        assert a.grad is not None and b.grad is not None and b.grad.count_nonzero() > 0
        torch.testing.assert_close(b.grad, a.grad, atol=3e-6, rtol=3e-4)
    unseen = torch.tensor([1, 5, 6, 8, 9] + ([] if padding_idx is None else [padding_idx]))
    for p in bounded.factor_parameters():
        assert p.grad.reshape(11, -1)[unseen].count_nonzero() == 0
    if rank:
        assert bounded.quantizer.lora_B.grad[unseen].count_nonzero() == 0
    for module in (dense.core, bounded):
        torch.optim.Adam(module.trainable_parameters(), lr=.002, eps=1e-4).step()
    for a, b in zip(dense.core.trainable_parameters(), bounded.trainable_parameters()):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    if phase == 'hard':
        for name, saved in snapshots.items():
            actual_snapshot = getattr(bounded.quantizer, name)
            assert actual_snapshot is None if saved is None else torch.equal(actual_snapshot, saved)
    assert source.weight.grad is None and torch.equal(source.weight, original)
    assert bounded.weight.data_ptr() == source.weight.data_ptr()
    bounded.assert_source_unchanged(verify_hash=True)


@pytest.mark.parametrize('rotated', [False, True])
def test_current_soft_surrogate_and_hard_export_use_current_parameters(rotated):
    _, dense, bounded, ids = paired(rotated=rotated, hard_phase='soft_surrogate')
    dense.core.set_progress(1)
    bounded.begin_step(1)
    (dense(ids).float().square().sum()).backward()
    (bounded(ids).float().square().sum()).backward()
    for a, b in zip(dense.core.trainable_parameters(), bounded.trainable_parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=3e-4)
    codes, scales = dense.core._hard_codes_and_scales()
    for start in range(0, 11, 2):
        end = min(start + 2, 11)
        actual_codes, actual_scales = bounded.hard_codes_rows(start, end)
        assert torch.equal(actual_codes, codes.reshape(11, 16)[start:end].to(torch.int8))
        torch.testing.assert_close(actual_scales, scales.reshape(11, 4)[start:end], atol=1e-8, rtol=1e-6)


def test_reconstruct_rows_fits_padding_and_token_forward_suppresses_its_gradient():
    source, _, bounded, _ = paired()
    bounded.begin_step(.3)
    ids = torch.tensor([3, 10])
    expected = bounded(ids)
    rows = bounded.reconstruct_rows(ids)
    assert torch.equal(rows, expected)
    rows.float().square().sum().backward()
    assert bounded.quantizer.raw_scale.grad.reshape(11, -1)[3].count_nonzero() > 0
    bounded.zero_grad(set_to_none=True)
    bounded(torch.tensor([3, 3])).float().square().sum().backward()
    assert all(p.grad is not None and p.grad.count_nonzero() == 0 for p in bounded.trainable_parameters())
    assert torch.equal(bounded.teacher_rows(ids), source(ids))


@pytest.mark.parametrize('rotated', [False, True])
def test_exact_original_bypass_ids_shapes_dtype_and_empty_gradient(rotated):
    source, _, bounded, _ = paired(rotated=rotated)
    for ids in (torch.tensor(3), torch.tensor([[[3, 10], [3, 2]]], dtype=torch.int32),
                torch.empty((2, 0, 3), dtype=torch.int64)):
        assert torch.equal(bounded(ids), source(ids))
        bounded.begin_step(.2)
        bounded.quant_enabled = False
        assert torch.equal(bounded(ids), source(ids))
        bounded.quant_enabled = True
        bounded.begin_step(0)
    bounded.begin_step(.2)
    for ids in (torch.tensor(2, dtype=torch.int32), torch.tensor([[[2, 7], [2, 4]]], dtype=torch.int32)):
        assert bounded(ids).shape == (*ids.shape, 16)
        assert torch.equal(bounded(ids), bounded(ids.long()))
    empty = bounded(torch.empty((2, 0, 3), dtype=torch.int64))
    assert empty.shape == (2, 0, 3, 16) and empty.dtype == torch.bfloat16 and empty.requires_grad
    empty.sum().backward()
    assert all(p.grad is not None and p.grad.count_nonzero() == 0 for p in bounded.trainable_parameters())
    assert bounded.reconstruct_rows(torch.empty(0, dtype=torch.int32)).shape == (0, 16)
    assert bounded.teacher_rows(torch.empty(0, dtype=torch.int64)).shape == (0, 16)


def test_snapshot_only_changes_in_begin_step_and_missing_hard_reference_rejected():
    _, _, bounded, ids = paired()
    bounded.begin_step(.4)
    before = {n: getattr(bounded.quantizer, n).clone() for n in bounded.quantizer._snapshot_names()}
    marker = bounded.quantizer._snapshot_step
    with torch.no_grad():
        bounded.quantizer.lora_B.add_(.1)
        bounded(ids)
    bounded.reconstruct_rows(torch.tensor([10, 0]))
    assert bounded.quantizer._snapshot_step == marker
    assert all(torch.equal(getattr(bounded.quantizer, n), value) for n, value in before.items())
    bounded.begin_step(.95)
    assert all(torch.equal(getattr(bounded.quantizer, n), value) for n, value in before.items())
    bounded.clear_soft_snapshot()
    with pytest.raises(RuntimeError, match='preceding soft begin_step'):
        bounded(ids)
    with torch.no_grad():
        assert torch.isfinite(bounded(ids)).all()


def test_dtype_conversion_preserves_original_and_fp32_state_and_gradients():
    source, _, bounded, ids = paired()
    bounded.begin_step(.3)
    bounded(ids).float().square().sum().backward()
    before = {n: p.detach().clone() for n, p in bounded.named_parameters()}
    before.update({n: b.clone() for n, b in bounded.named_buffers()
                   if b is not None and b.dtype == torch.float32 and not n.endswith('_device_probe')})
    original = source.weight.clone()
    bounded.half().float().bfloat16()
    for name, expected in before.items():
        actual = bounded.get_parameter(name) if name in dict(bounded.named_parameters()) else bounded.get_buffer(name)
        assert actual.dtype == torch.float32 and torch.equal(actual, expected)
    assert all(p.grad.dtype == torch.float32 for p in bounded.trainable_parameters())
    assert bounded.weight.dtype == torch.bfloat16 and torch.equal(bounded.weight, original)
    assert torch.equal(source.weight, original)
    bounded.assert_source_unchanged(verify_hash=True)


def test_source_mutation_validation_and_unsupported_source_options():
    source, _, bounded, ids = paired()
    with torch.no_grad():
        source.weight[0, 0].add_(1)
    with pytest.raises(RuntimeError, match='source was modified'):
        bounded(ids)
    source, _, bounded, _ = paired()
    source.weight.data[0, 0].add_(1)
    with pytest.raises(RuntimeError, match='source hash changed'):
        bounded.assert_source_unchanged(verify_hash=True)
    for kwargs in ({'max_norm': 1.}, {'sparse': True}, {'scale_grad_by_freq': True}):
        with pytest.raises(ValueError, match='plain embeddings'):
            BoundedTernaryEmbedding(nn.Embedding(11, 16, dtype=torch.bfloat16, **kwargs), group_size=4)
    with pytest.raises(TypeError, match='nn.Embedding'):
        BoundedTernaryEmbedding(nn.Linear(16, 11, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match='BF16'):
        BoundedTernaryEmbedding(nn.Embedding(11, 16), group_size=4)


def test_id_and_row_contracts_nonfinite_factors_and_bounds():
    _, _, bounded, _ = paired()
    for ids in (torch.tensor([1.]), torch.tensor([True]), [1]):
        with pytest.raises(ValueError, match='int32 or int64'):
            bounded(ids)
    for ids in (torch.tensor([-1]), torch.tensor([11])):
        with pytest.raises(IndexError, match='outside vocabulary'):
            bounded(ids)
    for ids in (torch.tensor([1, 1]), torch.tensor([[1]]), torch.tensor([1, 2, 3])):
        with pytest.raises(ValueError, match='row IDs'):
            bounded.reconstruct_rows(ids)
        with pytest.raises(ValueError, match='row IDs'):
            bounded.teacher_rows(ids)
    with pytest.raises(ValueError, match='row_chunk_size'):
        bounded.hard_codes_rows(0, 3)
    with torch.no_grad():
        bounded.quantizer.raw_scale[0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        bounded.validate_parameters()
    with pytest.raises(ValueError, match='nonfinite'):
        bounded.hard_codes_rows(0, 1)


class NoFullFloatTable(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        for value in tree_flatten(result)[0]:
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32 and value.device.type != 'meta':
                assert value.numel() != 11 * 4096, f'Full FP32 table in {func}'
        return result


def test_native_width_selected_rows_are_bounded_in_init_forward_backward_and_codes(monkeypatch):
    source = nn.Embedding(11, 4096, dtype=torch.bfloat16)
    observations = []
    original_rotate = eq.rotate
    def checked_rotate(value, signs, block):
        assert value.ndim == 2 and value.shape[0] <= 2 and value.shape[-1] == 4096
        observations.append(tuple(value.shape))
        return original_rotate(value, signs, block)
    monkeypatch.setattr(eq, 'rotate', checked_rotate)
    with NoFullFloatTable():
        bounded = BoundedTernaryEmbedding(source, row_chunk_size=2, lora_rank=2,
            signs=fixed_signs(4096, seed=8))
        original_components = bounded.quantizer._components
        def components(start, end, parameters, **kwargs):
            assert end - start <= 2 and parameters[-1].shape[0] <= 2
            return original_components(start, end, parameters, **kwargs)
        monkeypatch.setattr(bounded.quantizer, '_components', components)
        for progress in (.4, .95):
            bounded.begin_step(progress)
            bounded(torch.tensor([10, 2, 10, 7, 3, 0])).float().square().mean().backward()
            bounded.zero_grad(set_to_none=True)
        for start in range(0, 11, 2):
            bounded.hard_codes_rows(start, min(start + 2, 11))
    assert observations and bounded.weight.data_ptr() == source.weight.data_ptr()


def test_original_float_rotation_diagnostic_and_native_hybrid_gradient_path():
    from openbonsai.adapter import tiny_model
    torch.set_num_threads(2)
    model = tiny_model(seed=44).to(torch.bfloat16)
    source = model.get_input_embeddings()
    ids = torch.tensor([[3, 7, 9, 3, 14, 9]])
    with torch.no_grad():
        original = model(input_ids=ids, use_cache=False).logits
    replacement = BoundedTernaryEmbedding(source, group_size=32, row_chunk_size=2,
        signs=fixed_signs(128, seed=44), block_size=128, lora_rank=2)
    model.set_input_embeddings(replacement)
    with torch.no_grad():
        assert torch.equal(model(input_ids=ids, use_cache=False).logits, original)
        replacement.begin_step(.2)
        replacement.quant_enabled = False
        assert torch.equal(model(input_ids=ids, use_cache=False).logits, original)
        replacement.quant_enabled = True
        replacement.rotation_only = True
        torch.testing.assert_close(replacement(ids), source(ids), atol=2e-7, rtol=1e-5)
        replacement.rotation_only = False
        replacement.quantizer.lora_B.normal_(std=.02)
    logits = model(input_ids=ids, use_cache=False).logits
    F.cross_entropy(logits[:, :-1].float().flatten(0, 1), ids[:, 1:].flatten()).backward()
    assert all(p.grad is not None and p.grad.count_nonzero() > 0 for p in replacement.trainable_parameters())
    owned = {id(p) for p in replacement.parameters()}
    assert all(p.grad is None and not p.requires_grad for p in model.parameters() if id(p) not in owned)
    replacement.assert_source_unchanged(verify_hash=True)
    with torch.no_grad():
        assert not model(input_ids=ids, use_cache=False).logits.requires_grad
