"""Read-only, bounded cache KL for an exported head and its original teacher.

This explicitly scores the caller-supplied cache role. It does not select data,
fit parameters, warm decoded weights, or infer a staged job's authenticity.
"""
from __future__ import annotations

import copy
import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from .head_calibration import (
    _cache, _check_source, _positive_int, _provenance, chunked_full_vocab_kl,
)
from .export import _digest
from .mixed_formats import MixedLinear, _segments, validate_candidate


def _serialized_identity(candidate):
    """Exact save_candidate file SHA/length, including its trailing checksum.

    _segments transfers at most 4 MiB or one tensor row at a time. No encoded
    candidate blob or reconstructed full matrix is retained by this function.
    """
    prefix_digest, file_digest, size = hashlib.sha256(), hashlib.sha256(), 0
    for segment in _segments(candidate):
        prefix_digest.update(segment)
        file_digest.update(segment)
        size += len(segment)
    checksum = prefix_digest.digest()
    file_digest.update(checksum)
    return {'sha256': file_digest.hexdigest(), 'bytes': size + len(checksum),
            'scope': 'complete canonical candidate serialization including trailing checksum'}


def _check_candidate_source(candidate, cache, source_provenance):
    details = candidate.metadata['provenance']
    if (details.get('source_weight_sha256') != cache.manifest['source_weight_sha256']
            or details.get('source_dtype') != 'bfloat16'):
        raise ValueError('Packed head original BF16 source weight identity is missing or mismatched')
    if (candidate.metadata['source'] != 'openbonsai-endpoint-calibration'
            or details.get('export_method') != 'bounded-head-hard-ternary-v1'):
        raise ValueError('Expected the bounded head endpoint export contract')
    expected = _provenance(source_provenance)
    checked = ['source_weight_sha256', 'source_dtype']
    if 'source_provenance' in details:
        if _provenance(details['source_provenance']) != expected:
            raise ValueError('Packed head nested original source provenance mismatch')
        checked.append('source_provenance')
    identity_fields = ('source_checkpoint', 'source_revision', 'resolved_revision', 'source_content_sha256')
    for name in identity_fields:
        if name in details:
            target = 'source_revision' if name == 'resolved_revision' else name
            if details[name] != expected.get(target):
                raise ValueError(f'Packed head original source provenance mismatch: {name}')
            checked.append(name)
    if 'source_bias_sha256' in details:
        if details['source_bias_sha256'] != cache.manifest['source_bias_sha256']:
            raise ValueError('Packed head original bias identity mismatch')
        checked.append('source_bias_sha256')
    if 'scale_storage' in details and details['scale_storage'] != 'float16':
        raise ValueError('Packed head scale storage provenance mismatch')
    return checked


@torch.no_grad()
def evaluate_packed_head_from_cache(packed_head, original_head, cache, *, source_provenance,
                                    token_microbatch=64, vocabulary_chunk=2048):
    """Score every valid cached position against the unchanged BF16 head.

    The input must be a cold MixedLinear with ternary payloads, not a trainable
    head. Existing decoded caches are rejected without clearing them. Teacher
    and student use native BF16 logits; normalization and reduction are FP64
    over the entire vocabulary. Stored FP16 scales/bias are used verbatim before
    native BF16 computation, so this is distinct from pre-export hard-head KL.

    Source/cache hashes and any supplied candidate source claims are checked.
    Candidate job/config fields are preserved as claims, not authenticated as a
    complete pipeline receipt by this low-level function. No other cache role
    is discovered or scored implicitly. The caller owns semantic data selection.
    """
    _positive_int(token_microbatch, 'token_microbatch')
    _positive_int(vocabulary_chunk, 'vocabulary_chunk')
    if type(packed_head) is not MixedLinear or type(original_head) is not nn.Linear:
        raise ValueError('Expected a MixedLinear student and original nn.Linear teacher')
    if packed_head._decoded_weight_cache is not None or packed_head._decoded_weight_cache_signature is not None:
        raise ValueError('Packed head scoring requires a cold decoded cache; no cache is cleared or warmed')
    if any(p.requires_grad for p in original_head.parameters()):
        raise ValueError('Original teacher must be frozen')
    device = original_head.weight.device
    if (original_head.weight.dtype != torch.bfloat16
            or (original_head.bias is not None and (original_head.bias.dtype != torch.bfloat16
                                                    or original_head.bias.device != device))
            or packed_head._compute_reference.dtype != torch.bfloat16
            or packed_head._compute_reference.device != device):
        raise ValueError('Teacher and packed compute must use BF16 on the same device')
    candidate = packed_head._candidate()
    validate_candidate(candidate)
    if (candidate.module_kind != 'linear' or candidate.encoding not in
            {'ternary-2bit', 'ternary-bitcos', 'ternary-base3'}):
        raise ValueError('Expected a validated ternary linear candidate')
    if (tuple(candidate.shape) != tuple(original_head.weight.shape)
            or (packed_head.out_features, packed_head.in_features) != tuple(candidate.shape)
            or (original_head.out_features, original_head.in_features) != tuple(candidate.shape)
            or any(t.device != device or t.requires_grad for t in candidate.tensors.values())):
        raise ValueError('Packed dimensions/device or immutable payload contract mismatch')
    if (packed_head.bias is None) != (original_head.bias is None):
        raise ValueError('Packed head bias presence does not match the original teacher')
    if original_head.bias is not None and (packed_head.bias.dtype != torch.float16
            or not torch.equal(packed_head.bias, original_head.bias.detach().to(torch.float16))):
        raise ValueError('Packed head bias does not match the original bias converted to export FP16')
    _positive_int(packed_head.row_chunk_size, 'packed_head.row_chunk_size')
    cache = _cache(cache)
    _check_source(original_head, cache, source_provenance)
    checked = _check_candidate_source(candidate, cache, source_provenance)
    identity = _serialized_identity(candidate)
    cache_sha = _digest(cache.directory / 'manifest.json')
    chunk = min(vocabulary_chunk, packed_head.row_chunk_size)
    total, positions, microbatches = 0., 0, 0
    for index in range(len(cache)):
        for record in cache.iter_record(index, token_microbatch=token_microbatch):
            hidden = record['hidden'].to(device)
            mask = torch.ones(hidden.shape[0], dtype=torch.bool, device=device)
            def teacher(start, end):
                bias = None if original_head.bias is None else original_head.bias[start:end]
                return F.linear(hidden, original_head.weight[start:end], bias)
            total += float(chunked_full_vocab_kl(
                lambda start, end: packed_head.forward_rows(hidden, start, end), teacher,
                vocabulary=packed_head.out_features, row_chunk=chunk, mask=mask,
                denominator=cache.positions, backward=False))
            positions += hidden.shape[0]
            microbatches += 1
    if positions != cache.positions:
        raise ValueError('Scored cache position count changed')
    _check_source(original_head, cache, source_provenance)
    if _serialized_identity(packed_head._candidate()) != identity:
        raise ValueError('Packed head candidate changed during scoring')
    if _digest(cache.directory / 'manifest.json') != cache_sha:
        raise ValueError('Cache manifest changed during scoring')
    if packed_head._decoded_weight_cache is not None or packed_head._decoded_weight_cache_signature is not None:
        raise ValueError('Decoded cache state changed during scoring')
    return {
        'kl': total, 'positions': positions, 'records': len(cache), 'microbatches': microbatches,
        'role': cache.manifest['role'], 'source_provenance': copy.deepcopy(cache.manifest['source_provenance']),
        'source_weight_sha256': cache.manifest['source_weight_sha256'],
        'source_bias_sha256': cache.manifest['source_bias_sha256'],
        'cache_manifest_sha256': cache_sha, 'cache_config_sha256': cache.manifest['config_sha256'],
        'position_policy': cache.manifest['position_policy'],
        'candidate_serialization': identity, 'candidate_format': candidate.metadata['format'],
        'candidate_encoding': candidate.encoding, 'candidate_shape': list(candidate.shape),
        'candidate_provenance': copy.deepcopy(candidate.metadata['provenance']),
        'checked_candidate_source_fields': checked, 'pipeline_receipt_verified': False,
        'rotation': copy.deepcopy(candidate.metadata.get('rotation')),
        'scale_storage_dtype': str(candidate.tensors['scales'].dtype).removeprefix('torch.'),
        'student_bias_storage_dtype': None if packed_head.bias is None else str(packed_head.bias.dtype).removeprefix('torch.'),
        'teacher_bias_dtype': None if original_head.bias is None else 'bfloat16',
        'compute_dtype': 'bfloat16', 'device': str(device), 'normalization': 'FP64',
        'objective': 'actual full-vocabulary KL(original_teacher || stored_student), temperature 1',
        'representation': 'exported codes and stored scales/bias; not pre-export FP32-scale hard head',
        'token_microbatch': token_microbatch, 'effective_vocabulary_chunk': chunk,
        'runtime_row_chunk_size': packed_head.row_chunk_size,
        'cache_policy': 'cold bounded row reconstruction', 'decoded_weight_cache_bytes': 0,
        'quality_claim': False,
    }
