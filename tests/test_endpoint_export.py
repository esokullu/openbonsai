"""Codec equivalence, bounded work, and native head-only artifact round trips."""
import copy
import hashlib

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from openbonsai.candidates import source_weight_sha256
from openbonsai.endpoint_export import export_head_candidate
from openbonsai.endpoint_quantizer import BoundedTernaryHead
from openbonsai.hadamard import rotate
from openbonsai.mixed_formats import (
    MixedLinear, candidate_nbytes, candidate_weight, decode_candidate, encode_candidate,
)


class FixedRows:
    """Protocol fixture with explicit row patterns and real source hash guards."""
    def __init__(self, codes, *, group_size=1, chunk=2, scales=None, bias=False, signs=None):
        self.codes = codes.to(torch.int8).clone()
        self.out_features, self.in_features = codes.shape
        self.group_size, self.row_chunk_size = group_size, chunk
        self.weight = torch.arange(codes.numel(), dtype=torch.float32).reshape(codes.shape).to(torch.bfloat16)
        self.bias = (torch.arange(self.out_features).to(torch.bfloat16) / 16) if bias else None
        self.scales = (torch.linspace(.10001, .90001, codes.numel() // group_size).reshape(
            self.out_features, self.in_features // group_size) if scales is None else scales)
        self.rotation_signs, self.block_size = signs, 4
        self.source_weight_sha256 = source_weight_sha256(self.weight)
        self._bias_hash = self._hash_bias()
        self.calls, self.guards = [], []

    def _hash_bias(self):
        return None if self.bias is None else hashlib.sha256(self.bias.view(torch.uint8).numpy().tobytes()).hexdigest()

    def assert_source_unchanged(self, *, verify_hash=False):
        self.guards.append(verify_hash)
        if verify_hash and (source_weight_sha256(self.weight) != self.source_weight_sha256
                            or self._hash_bias() != self._bias_hash):
            raise RuntimeError('original source hash changed')

    def validate_parameters(self):
        pass

    def hard_codes_rows(self, start, end):
        assert 0 < end - start <= self.row_chunk_size
        self.calls.append((start, end))
        return self.codes[start:end], self.scales[start:end]


PATTERN = torch.tensor([
    [0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [-1, 1, 0, 0, 0],
    [1, -1, 1, 0, 0], [-1, 1, -1, 1, 0], [1, -1, 1, -1, 1],
    [0, 0, 0, 0, -1],
], dtype=torch.int8)


def dense_reference(head):
    return (head.codes.reshape(head.out_features, -1, head.group_size).float()
            * head.scales.half().float().unsqueeze(-1)).reshape(head.codes.shape)


@pytest.mark.parametrize('packing', ['2bit', 'bitcos', 'base3', 'auto'])
@pytest.mark.parametrize('chunk', [1, 2, 3, 4])
def test_codecs_cross_partial_bytes_tail_rows_and_fp16_scales(packing, chunk):
    head = FixedRows(PATTERN, chunk=chunk, bias=True)
    candidate = export_head_candidate(head, packing=packing, provenance={'source_checkpoint': 'fixture'})
    torch.testing.assert_close(candidate_weight(candidate), dense_reference(head), rtol=0, atol=0)
    torch.testing.assert_close(candidate.tensors['bias'], head.bias.half(), rtol=0, atol=0)
    blob = encode_candidate(candidate)
    assert len(blob) == candidate_nbytes(candidate)
    torch.testing.assert_close(candidate_weight(decode_candidate(blob)), dense_reference(head), rtol=0, atol=0)
    spans = [(i, min(i + chunk, head.out_features)) for i in range(0, head.out_features, chunk)]
    assert head.calls == spans * 2
    assert head.guards == [True, True]
    assert not {'weight', 'lora_A', 'lora_B'} & candidate.tensors.keys()
    assert candidate.metadata['provenance']['source_weight_sha256'] == head.source_weight_sha256
    if candidate.encoding == 'ternary-bitcos':
        assert candidate.tensors['row_sign_offsets'].tolist() == [0, 0, 1, 3, 6, 10, 15, 16]


@pytest.mark.parametrize('density', ['zero', 'nonzero'])
def test_bitcos_all_zero_and_all_nonzero_rows(density):
    codes = torch.zeros(5, 7, dtype=torch.int8) if density == 'zero' else torch.full((5, 7), -1, dtype=torch.int8)
    head = FixedRows(codes, chunk=2)
    candidate = export_head_candidate(head, packing='bitcos')
    torch.testing.assert_close(candidate_weight(candidate), dense_reference(head), rtol=0, atol=0)
    if density == 'zero':
        assert candidate.tensors['signs'].numel() == 0
        assert candidate.tensors['row_sign_offsets'].count_nonzero() == 0


@pytest.mark.parametrize('rows,width,nonzero,expected', [
    (1, 16, False, 'ternary-2bit'),  # Base3 has equal payload bytes but a larger header.
    (17, 2048, False, 'ternary-bitcos'),
    (17, 2048, True, 'ternary-base3'),
])
def test_auto_chooses_actual_complete_serialized_cost(rows, width, nonzero, expected):
    codes = torch.full((rows, width), int(nonzero), dtype=torch.int8)
    head = FixedRows(codes, group_size=16, chunk=3)
    candidates = {codec: export_head_candidate(head, packing=codec)
                  for codec in ('2bit', 'bitcos', 'base3', 'auto')}
    actual = {codec: len(encode_candidate(candidate)) for codec, candidate in candidates.items()}
    assert actual['auto'] == min(actual[codec] for codec in ('2bit', 'bitcos', 'base3'))
    assert candidates['auto'].encoding == expected


@pytest.mark.parametrize('packing', ['2bit', 'bitcos', 'base3'])
def test_rotation_signs_remain_explicit_and_operator_reloads_identically(packing):
    codes = torch.tensor([[1, 0, -1, 0, 1, 0, -1, 1]] * 9, dtype=torch.int8)
    signs = torch.tensor([1, -1, 1, 1, -1, 1, -1, -1], dtype=torch.int8)
    head = FixedRows(codes, group_size=2, chunk=4, signs=signs, bias=True)
    candidate = export_head_candidate(head, packing=packing, provenance={'rotation_seed': 23})
    assert candidate.metadata['rotation'] == {'block_size': 4, 'convention': 'row-DH', 'normalization': 'orthonormal'}
    torch.testing.assert_close(candidate.tensors['rotation_signs'], signs)
    hidden = torch.randn(3, 8)
    expected = F.linear(rotate(hidden, signs, 4), dense_reference(head), head.bias.float())
    restored = MixedLinear(decode_candidate(encode_candidate(candidate)), dtype=torch.float32, row_chunk_size=4)
    torch.testing.assert_close(restored(hidden), expected, rtol=1e-6, atol=1e-6)


def test_export_does_not_materialize_complete_codes_or_float_matrix():
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    head = FixedRows(torch.randint(-1, 2, (13, 33), dtype=torch.int8), group_size=3, chunk=2)

    class RejectDenseMatrix(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            for value in tree_flatten(result)[0]:
                if isinstance(value, torch.Tensor) and tuple(value.shape) == (13, 33):
                    pytest.fail(f'export allocated or materialized a full head through {func}')
            return result

    # The protocol fixture source hash helper defaults to 256 rows. Supply its
    # bounded equivalent here so this checks the exporter and source guard too.
    def bounded_guard(*, verify_hash=False):
        head.guards.append(verify_hash)
        digest = hashlib.sha256()
        for start in range(0, head.out_features, head.row_chunk_size):
            digest.update(head.weight[start:start + head.row_chunk_size].view(torch.uint8).numpy().tobytes())
        assert digest.hexdigest() == head.source_weight_sha256

    head.assert_source_unchanged = bounded_guard
    with RejectDenseMatrix():
        candidate = export_head_candidate(head)
    torch.testing.assert_close(candidate_weight(candidate), dense_reference(head), rtol=0, atol=0)


@pytest.mark.parametrize('problem', ['bad-code', 'bad-code-type', 'scale-zero', 'scale-inf',
                                    'scale-overflow', 'scale-underflow', 'bad-scale-shape'])
def test_rejects_malformed_hard_rows_and_nonrepresentable_scales(problem):
    head = FixedRows(PATTERN)
    if problem == 'bad-code':
        head.codes[2, 1] = 2
    elif problem == 'bad-code-type':
        head.codes = head.codes.float()
    elif problem == 'bad-scale-shape':
        head.scales = head.scales[:, :1]
    else:
        head.scales[2, 1] = {'scale-zero': 0., 'scale-inf': float('inf'),
                             'scale-overflow': 1e6, 'scale-underflow': 1e-9}[problem]
    with pytest.raises(ValueError):
        export_head_candidate(head)


@pytest.mark.parametrize('problem', ['source', 'bias', 'wrong-provenance', 'bad-json', 'invalid-rotation', 'chunk'])
def test_rejects_stale_source_and_invalid_export_options(problem):
    head = FixedRows(PATTERN, bias=True)
    kwargs = {}
    if problem == 'source':
        head.weight[0, 0] = -1
    elif problem == 'bias':
        head.bias[0] = -1
    elif problem == 'wrong-provenance':
        kwargs['provenance'] = {'source_weight_sha256': '0' * 64}
    elif problem == 'bad-json':
        kwargs['provenance'] = {'invalid': float('nan')}
    elif problem == 'invalid-rotation':
        head.rotation_signs = torch.ones(5, dtype=torch.int8)
    else:
        kwargs['row_chunk_size'] = head.row_chunk_size + 1
    with pytest.raises((ValueError, RuntimeError)):
        export_head_candidate(head, **kwargs)


@pytest.mark.parametrize('change', ['code', 'scale'])
def test_rejects_changed_result_between_count_and_packing_passes(change):
    head = FixedRows(PATTERN)
    original = head.hard_codes_rows
    number_chunks = (head.out_features + head.row_chunk_size - 1) // head.row_chunk_size

    def mutate_second_pass(start, end):
        if len(head.calls) == number_chunks:
            if change == 'code':
                # Preserve nonzero count so the planned sign allocation fits.
                head.codes[1, 0] *= -1
            else:
                head.scales[1, 0] += .01
        return original(start, end)

    head.hard_codes_rows = mutate_second_pass
    with pytest.raises(ValueError, match='changed between export passes'):
        export_head_candidate(head, packing='bitcos')


@pytest.mark.parametrize('packing', ['2bit', 'bitcos', 'base3'])
@pytest.mark.parametrize('rotated', [False, True])
def test_actual_calibrated_head_exports_lora_into_codes(packing, rotated):
    torch.manual_seed(15)
    source = nn.Linear(16, 11, bias=True).to(torch.bfloat16)
    signs = torch.tensor([1, -1] * 8, dtype=torch.int8) if rotated else None
    head = BoundedTernaryHead(source, group_size=4, row_chunk_size=3, signs=signs, block_size=4, lora_rank=2)
    with torch.no_grad():
        head.lora_B.normal_(std=.15)
        head.raw_scale.add_(.2)
    original_weight = source.weight.detach().clone()
    candidate = export_head_candidate(head, packing=packing)
    expected = []
    for start in range(0, 11, 3):
        codes, scales = head.hard_codes_rows(start, min(start + 3, 11))
        expected.append((codes.reshape(-1, 4).float() * scales.half().float().reshape(-1, 1)).reshape(-1, 16))
    torch.testing.assert_close(candidate_weight(candidate), torch.cat(expected), rtol=0, atol=0)
    torch.testing.assert_close(source.weight, original_weight, rtol=0, atol=0)
    assert not {'weight', 'lora_A', 'lora_B'} & candidate.tensors.keys()


@pytest.mark.parametrize('rotated', [False, True])
@pytest.mark.parametrize('packing', ['base3', 'auto'])
def test_native_head_only_mixed_export_and_reload(tmp_path, rotated, packing):
    from openbonsai.adapter import tiny_model
    from openbonsai.mixed_export import prepare_mixed_export, save_mixed, load_mixed

    torch.set_num_threads(2)
    model = tiny_model(seed=41).to(torch.bfloat16)
    signs = torch.ones(model.lm_head.in_features, dtype=torch.int8) if rotated else None
    if signs is not None:
        signs[::3] = -1
    head = BoundedTernaryHead(model.lm_head, row_chunk_size=13, signs=signs, block_size=128, lora_rank=2)
    candidate = export_head_candidate(head, packing=packing, provenance={'calibration': 'fixture'})
    output = tmp_path / 'head-only'
    with prepare_mixed_export(model, ['lm_head'], metadata={'source_checkpoint': 'native-fixture'}) as plan:
        cost = plan.fixed_bytes + candidate_nbytes(candidate)
        report = save_mixed(plan, {'lm_head': candidate}, output, budget_bytes=cost)
    assert int(report['storage']['artifact_bytes']) == cost
    assert sum(p.stat().st_size for p in output.rglob('*') if p.is_file()) == cost
    expected = copy.deepcopy(model)
    expected.lm_head = MixedLinear(candidate)
    restored = load_mixed(output)
    ids = torch.tensor([[4, 5, 6, 7, 8]])
    with torch.no_grad():
        torch.testing.assert_close(restored(input_ids=ids).logits, expected(input_ids=ids).logits, rtol=0, atol=0)
    assert list(report['modules']) == ['lm_head']


def test_actual_head_source_data_mutation_is_caught_by_export_hash_guard():
    source = nn.Linear(8, 5, bias=False).to(torch.bfloat16)
    head = BoundedTernaryHead(source, group_size=4, row_chunk_size=2, lora_rank=0)
    source.weight.data[0, 0] = 10  # .data evades a version-only guard.
    with pytest.raises(RuntimeError, match='source hash'):
        export_head_candidate(head)


@pytest.mark.parametrize('parameter', ['raw_scale', 'lora_A'])
def test_export_rejects_nonfinite_trainable_head_state(parameter):
    source = nn.Linear(8, 5, bias=False).to(torch.bfloat16)
    head = BoundedTernaryHead(source, group_size=4, row_chunk_size=2, lora_rank=2)
    with torch.no_grad():
        getattr(head, parameter).reshape(-1)[0] = float('nan')
    with pytest.raises(ValueError, match='nonfinite'):
        export_head_candidate(head)


@pytest.mark.parametrize('change', ['source', 'rotation'])
def test_source_or_rotation_changes_during_packing_are_rejected(change):
    head = FixedRows(torch.ones(7, 8, dtype=torch.int8), signs=torch.ones(8, dtype=torch.int8))
    original = head.hard_codes_rows

    def mutate_at_last_row(start, end):
        result = original(start, end)
        if len(head.calls) == 8:
            if change == 'source':
                head.weight[0, 0] = -1
            else:
                head.rotation_signs[0] = -1
        return result

    head.hard_codes_rows = mutate_at_last_row
    with pytest.raises((ValueError, RuntimeError), match='[Rr]otation changed|source hash changed'):
        export_head_candidate(head)


@pytest.mark.parametrize('packing', ['2bit', 'bitcos', 'base3', 'auto'])
def test_smaller_requested_chunk_preserves_identical_serialized_candidate(packing):
    head = FixedRows(PATTERN, chunk=4)
    default = export_head_candidate(head, packing=packing)
    head.calls.clear()
    smaller = export_head_candidate(head, packing=packing, row_chunk_size=1)
    assert all(end - start == 1 for start, end in head.calls)
    assert encode_candidate(smaller) == encode_candidate(default)
