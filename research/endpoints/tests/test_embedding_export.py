"""Certified table coverage, exact mixed-format bytes and embedding semantics."""
import math

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from openbonsai.candidates import source_weight_sha256
from openbonsai.embedding_calibration import EmbeddingTrainingConfig, fit_embedding_table
from openbonsai.embedding_quantizer import BoundedTernaryEmbedding
from openbonsai.endpoint_export import export_embedding_candidate, export_head_candidate
from openbonsai.hadamard import inverse_rotate
from openbonsai.mixed_formats import (MixedEmbedding, candidate_nbytes, candidate_weight,
                                      decode_candidate, encode_candidate)
from openbonsai.rotations import fixed_signs


def fitted(*, rows=7, width=8, group=2, rotated=False, pattern='random', padding_idx=2):
    torch.manual_seed(172)
    source = nn.Embedding(rows, width, padding_idx=padding_idx, dtype=torch.bfloat16)
    with torch.no_grad():
        if pattern == 'zero':
            source.weight.zero_()
        elif pattern == 'nonzero':
            source.weight.copy_((torch.arange(width) % 2 * 2 - 1).expand(rows, -1))
        else:
            source.weight.normal_(std=.1)  # Deliberately nonzero padding row.
    embedding = BoundedTernaryEmbedding(source, group_size=group, row_chunk_size=3,
        lora_rank=0, signs=fixed_signs(width, seed=5090) if rotated else None, block_size=4)
    provenance = {'source_checkpoint': 'synthetic-embedding-table',
                  'source_content_sha256': source_weight_sha256(source.weight)}
    fit_embedding_table(embedding, EmbeddingTrainingConfig(steps=2), source_provenance=provenance)
    return embedding, provenance


@pytest.mark.parametrize('rotated', [False, True])
@pytest.mark.parametrize('packing', ['2bit', 'bitcos', 'base3', 'auto'])
def test_complete_candidate_cost_kind_and_inverse_lookup(rotated, packing):
    embedding, provenance = fitted(rotated=rotated)
    candidate = export_embedding_candidate(embedding, packing=packing, provenance=provenance)
    assert candidate.module_kind == 'embedding'
    assert candidate.metadata['provenance']['export_method'] == 'bounded-embedding-hard-ternary-v1'
    coverage = candidate.metadata['provenance']['calibration_certificate']['coverage']
    assert coverage['soft_rows'] == coverage['hard_rows'] == embedding.num_embeddings
    assert not {'weight', 'bias', 'lora_A', 'lora_B'} & candidate.tensors.keys()
    blob = encode_candidate(candidate)
    assert len(blob) == candidate_nbytes(candidate)
    restored = MixedEmbedding(decode_candidate(blob), padding_idx=embedding.padding_idx, row_chunk_size=2)
    latent = candidate_weight(candidate, dtype=torch.bfloat16)
    ids = torch.tensor([[0, 6, 2, 6], [1, 2, 0, 5]], dtype=torch.int32)
    expected = F.embedding(ids, latent)
    if rotated:
        expected = inverse_rotate(expected, embedding.rotation_signs, embedding.block_size)
    assert torch.equal(restored(ids), expected)
    assert restored(torch.empty((2, 0), dtype=torch.int64)).shape == (2, 0, embedding.embedding_dim)
    # Native padding_idx does not zero a nonzero stored padding row.
    assert torch.count_nonzero(restored(torch.tensor([embedding.padding_idx]))) > 0


@pytest.mark.parametrize('pattern', ['zero', 'nonzero', 'random'])
def test_partial_bytes_and_auto_compares_final_embedding_metadata(pattern):
    embedding, _ = fitted(rows=5, width=7, group=7, pattern=pattern)
    candidates = {codec: export_embedding_candidate(embedding, packing=codec)
                  for codec in ('2bit', 'bitcos', 'base3', 'auto')}
    costs = {codec: len(encode_candidate(candidate)) for codec, candidate in candidates.items()}
    assert costs['auto'] == min(costs[codec] for codec in ('2bit', 'bitcos', 'base3'))
    assert all(candidate.module_kind == 'embedding' for candidate in candidates.values())
    for candidate in candidates.values():
        assert torch.equal(candidate_weight(candidate), candidate_weight(candidates['2bit']))
    if pattern == 'zero':
        assert candidates['bitcos'].tensors['signs'].numel() == 0
    if pattern == 'nonzero':
        assert torch.count_nonzero(candidate_weight(candidates['bitcos'])) == 35


def test_missing_coverage_rejected_before_hard_row_traversal(monkeypatch):
    embedding = BoundedTernaryEmbedding(nn.Embedding(7, 8, dtype=torch.bfloat16),
                                        group_size=2, row_chunk_size=3, lora_rank=0)
    def forbidden(*args):
        raise AssertionError('uncalibrated table must not reach packing')
    monkeypatch.setattr(embedding.quantizer, 'hard_codes_rows', forbidden)
    with pytest.raises(RuntimeError, match='certificate'):
        export_embedding_candidate(embedding)
    with pytest.raises(ValueError, match='export_embedding_candidate'):
        export_head_candidate(embedding)


@pytest.mark.parametrize('mutation', ['scale', 'signs', 'source', 'padding'])
def test_stale_state_rejected_even_if_changed_through_data(mutation):
    embedding, _ = fitted(rotated=True)
    if mutation == 'scale':
        embedding.quantizer.raw_scale.data.add_(.01)
    elif mutation == 'signs':
        embedding.rotation_signs.data[0] *= -1
    elif mutation == 'source':
        embedding.weight.data[0, 0] += 1
    else:
        embedding.padding_idx = None
    with pytest.raises(RuntimeError):
        export_embedding_candidate(embedding)


@pytest.mark.parametrize('packing', ['bitcos', 'base3'])
def test_state_change_after_last_code_chunk_is_not_silently_certified(monkeypatch, packing):
    embedding, _ = fitted()
    original = embedding.quantizer.hard_codes_rows
    calls = 0
    final_call = 2 * math.ceil(embedding.num_embeddings / embedding.row_chunk_size)
    def changed_after_rows(start, end):
        nonlocal calls
        result = original(start, end)
        calls += 1
        if calls == final_call:
            embedding.quantizer.raw_scale.data.add_(.125)
        return result
    monkeypatch.setattr(embedding.quantizer, 'hard_codes_rows', changed_after_rows)
    with pytest.raises(RuntimeError, match='stale'):
        export_embedding_candidate(embedding, packing=packing)


def test_provenance_and_bias_cannot_change_embedding_semantics():
    embedding, _ = fitted()
    with pytest.raises(ValueError, match='contradicts'):
        export_embedding_candidate(embedding, provenance={'source_checkpoint': 'different-checkpoint'})
    with pytest.raises(ValueError, match='reserved'):
        export_embedding_candidate(embedding, provenance={'calibration_certificate': {}})
    # Bias is unsupported even if an advanced caller directly alters the core.
    embedding.quantizer.bias = torch.zeros(embedding.num_embeddings, dtype=torch.bfloat16)
    with pytest.raises((ValueError, RuntimeError)):
        export_embedding_candidate(embedding)


def test_revision_alias_cannot_contradict_the_fitting_checkpoint():
    source = nn.Embedding(5, 8, dtype=torch.bfloat16)
    embedding = BoundedTernaryEmbedding(source, group_size=2, row_chunk_size=3, lora_rank=0)
    provenance = {'source_checkpoint': 'fixture', 'resolved_revision': 'a' * 40}
    fit_embedding_table(embedding, EmbeddingTrainingConfig(steps=2), source_provenance=provenance)
    candidate = export_embedding_candidate(embedding, provenance=provenance)
    assert candidate.metadata['provenance']['source_revision'] == 'a' * 40
    assert 'resolved_revision' not in candidate.metadata['provenance']
    with pytest.raises(ValueError, match='contradicts'):
        export_embedding_candidate(embedding, provenance={'resolved_revision': 'b' * 40})
    with pytest.raises(ValueError, match='conflicting'):
        export_embedding_candidate(embedding, provenance={'source_revision': 'a' * 40,
                                                          'resolved_revision': 'b' * 40})


@pytest.mark.parametrize('packing', ['bitcos', 'base3'])
def test_export_guards_and_packing_do_not_promote_the_full_source_table(packing):
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
    embedding, _ = fitted(rows=13, width=32, group=4)
    class NoFullTable(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            for value in tree_flatten(result)[0]:
                if (isinstance(value, torch.Tensor) and tuple(value.shape) == (13, 32)
                        and value.dtype in (torch.float32, torch.int8)):
                    pytest.fail(f'Full table materialized by {func}')
            return result
    with NoFullTable():
        candidate = export_embedding_candidate(embedding, packing=packing)
    assert candidate.shape == (13, 32)
