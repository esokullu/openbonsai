"""Continuous base3 endpoint streams, bounded scratch and native reload parity."""
import copy
import hashlib

import pytest
import torch

from openbonsai.candidates import source_weight_sha256
from openbonsai.embedding_calibration import EmbeddingTrainingConfig, fit_embedding_table
from openbonsai.embedding_quantizer import BoundedTernaryEmbedding
from openbonsai.endpoint_export import _TritWriter, export_embedding_candidate, export_head_candidate
from openbonsai.mixed_formats import (
    MixedEmbedding, candidate_nbytes, candidate_weight, decode_candidate, encode_candidate,
)
from openbonsai.rotations import fixed_signs
from test_endpoint_export import FixedRows, dense_reference


def independent_bytes(codes):
    """Scalar base-3 oracle, independent of the tensor packing implementation."""
    flat = codes.flatten().tolist()
    digits = {0: 0, 1: 1, -1: 2}
    return bytes(sum(digits[code] * 3 ** lane for lane, code in enumerate(flat[start:start + 5]))
                 for start in range(0, len(flat), 5))


@pytest.mark.parametrize('width', [1, 4, 5, 7, 4096])
@pytest.mark.parametrize('chunk', [1, 2, 3, 6])
def test_flat_base3_stream_carries_trits_across_rows_and_chunk_boundaries(width, chunk):
    codes = ((torch.arange(7 * width) * 7 + torch.arange(7 * width) // 11) % 3 - 1).to(torch.int8).reshape(7, width)
    head = FixedRows(codes, chunk=chunk, group_size=128 if width == 4096 else 1)
    candidate = export_head_candidate(head, packing='base3')
    assert candidate.encoding == 'ternary-base3'
    assert set(candidate.tensors) == {'codes', 'scales'}
    assert bytes(candidate.tensors['codes'].tolist()) == independent_bytes(codes)
    assert candidate.tensors['codes'].numel() == (codes.numel() + 4) // 5
    if codes.numel() % 5:
        assert int(candidate.tensors['codes'][-1]) < 3 ** (codes.numel() % 5)
    torch.testing.assert_close(candidate_weight(candidate), dense_reference(head), rtol=0, atol=0)
    round_trip = decode_candidate(encode_candidate(candidate))
    assert torch.equal(round_trip.tensors['codes'], candidate.tensors['codes'])


@pytest.mark.parametrize('value', [-1, 0, 1])
def test_base3_zero_and_nonzero_rows_have_no_row_padding(value):
    codes = torch.full((7, 7), value, dtype=torch.int8)
    candidate = export_head_candidate(FixedRows(codes, chunk=2), packing='base3')
    assert bytes(candidate.tensors['codes'].tolist()) == independent_bytes(codes)
    assert candidate.tensors['codes'].numel() == 10  # ceil(49/5), not 7*ceil(7/5).


def test_writer_retains_only_four_pending_trits_and_rejects_wrong_length():
    writer = _TritWriter(14)
    codes = torch.tensor([0, 1, -1, 1, 0, -1, 0, 0, 1, 1, -1, 0, 1, -1], dtype=torch.int8)
    writer.append(codes[:4])
    assert writer.pending.numel() == 4
    assert writer.pending.untyped_storage().nbytes() == 4
    with pytest.raises(ValueError, match='length changed'):
        writer.finish()
    writer.append(codes[4:11])
    assert writer.pending.numel() == 1
    assert writer.pending.untyped_storage().nbytes() == 1
    writer.append(codes[11:])
    assert bytes(writer.finish().tolist()) == independent_bytes(codes)
    with pytest.raises(ValueError, match='does not fit'):
        writer.append(codes[:1])
    with pytest.raises(ValueError, match='only -1'):
        _TritWriter(1).append(torch.tensor([2], dtype=torch.int8))


def test_base3_export_avoids_a_full_flat_or_matrix_code_allocation():
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    rows, width, chunk = 13, 4096, 2
    head = FixedRows(torch.randint(-1, 2, (rows, width), dtype=torch.int8), group_size=128, chunk=chunk)

    def bounded_guard(*, verify_hash=False):
        assert verify_hash
        digest = hashlib.sha256()
        for start in range(0, rows, chunk):
            digest.update(head.weight[start:start + chunk].view(torch.uint8).numpy().tobytes())
        assert digest.hexdigest() == head.source_weight_sha256

    head.assert_source_unchanged = bounded_guard

    class BoundScratch(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            for value in tree_flatten(result)[0]:
                if isinstance(value, torch.Tensor) and value.dtype in (torch.int8, torch.float32):
                    assert value.numel() <= chunk * width + 4, f'unbounded endpoint scratch through {func}'
            return result

    with BoundScratch():
        candidate = export_head_candidate(head, packing='base3')
    assert bytes(candidate.tensors['codes'].tolist()) == independent_bytes(head.codes)


@pytest.mark.parametrize('change', ['code', 'scale'])
def test_base3_export_keeps_two_pass_identity_guard(change):
    head = FixedRows(torch.ones(7, 7, dtype=torch.int8), chunk=2)
    original = head.hard_codes_rows

    def changed_second_pass(start, end):
        if len(head.calls) == 4:
            if change == 'code':
                head.codes[-1, -1] = -1
            else:
                head.scales[-1, -1] += .01
        return original(start, end)

    head.hard_codes_rows = changed_second_pass
    with pytest.raises(ValueError, match='changed between export passes'):
        export_head_candidate(head, packing='base3')


@pytest.mark.parametrize('rotated', [False, True])
def test_native_embedding_only_base3_artifact_matches_two_bit_lookup_and_logits(tmp_path, rotated):
    from openbonsai.adapter import tiny_model
    from openbonsai.mixed_export import load_mixed, prepare_mixed_export, save_mixed

    torch.set_num_threads(2)
    model = tiny_model(seed=101, multimodal=True).to(torch.bfloat16)
    source = model.get_input_embeddings()
    module_name = next(name for name, module in model.named_modules() if module is source)
    signs = fixed_signs(source.embedding_dim, seed=101) if rotated else None
    embedding = BoundedTernaryEmbedding(source, group_size=32, row_chunk_size=13,
        signs=signs, block_size=64, lora_rank=0)
    provenance = {'source_checkpoint': 'random-native-base3-embedding-fixture',
                  'source_content_sha256': source_weight_sha256(source.weight)}
    fit_embedding_table(embedding, EmbeddingTrainingConfig(steps=2), source_provenance=provenance)
    two_bit = export_embedding_candidate(embedding, packing='2bit')
    base3 = export_embedding_candidate(embedding, packing='base3')
    assert base3.metadata['provenance']['calibration_certificate'] == two_bit.metadata['provenance']['calibration_certificate']
    assert torch.equal(candidate_weight(base3), candidate_weight(two_bit))
    output = tmp_path / 'embedding-only'
    with prepare_mixed_export(model, [module_name], metadata=provenance) as plan:
        budget = plan.fixed_bytes + candidate_nbytes(base3)
        report = save_mixed(plan, {module_name: base3}, output, budget_bytes=budget)
    assert int(report['storage']['artifact_bytes']) == budget
    assert sum(path.stat().st_size for path in output.rglob('*') if path.is_file()) == budget
    expected = copy.deepcopy(model)
    expected.set_submodule(module_name, MixedEmbedding(two_bit, padding_idx=source.padding_idx))
    restored = load_mixed(output)
    ids = torch.tensor([[4, 7, 7, 80, 2, 100, 255, 5]])
    with torch.no_grad():
        assert torch.equal(restored.get_input_embeddings()(ids), expected.get_input_embeddings()(ids))
        assert torch.equal(restored(input_ids=ids).logits, expected(input_ids=ids).logits)
    assert list(report['modules']) == [module_name]
