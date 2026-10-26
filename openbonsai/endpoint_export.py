"""Row-bounded export of a calibrated ternary vocabulary head.

Only one row chunk of hard codes and FP32 scales is requested at a time.
Compact payloads reside on CPU. A count pass determines the exact complete
serialized cost of both codecs without allocating either code stream; a second
pass packs only the selected codec. No original weight or low-rank factors are
included in the candidate. This is a reference exporter, not a training stage.
"""
from __future__ import annotations

import copy
import hashlib

import torch
from torch import Tensor

from .hadamard import validate_rotation
from .mixed_formats import (
    FORMAT, MixedWeightCandidate, _DIGEST, _PREFIX, _header,
    candidate_nbytes, validate_candidate,
)
from .packing import pack_ternary_codes


def _positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _sha256(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest")
    return value


def _pack_bits(bits: Tensor) -> Tensor:
    bits = bits.detach().to(device='cpu', dtype=torch.uint8).reshape(-1)
    if bits.numel() % 8:
        bits = torch.cat((bits, bits.new_zeros(8 - bits.numel() % 8)))
    lanes = bits.reshape(-1, 8)
    packed = torch.zeros(lanes.shape[0], dtype=torch.uint8)
    for shift in range(8):
        packed |= lanes[:, shift] << shift
    return packed


class _BitWriter:
    """Append packed little-endian bits, carrying partial bytes between rows."""
    def __init__(self, bit_count: int):
        self.bit_count = bit_count
        self.position = 0
        self.buffer = torch.zeros((bit_count + 7) // 8, dtype=torch.uint8)

    def append(self, packed: Tensor, count: int):
        if (packed.dtype != torch.uint8 or packed.ndim != 1 or packed.device.type != 'cpu'
                or packed.numel() != (count + 7) // 8 or count < 0
                or self.position + count > self.bit_count):
            raise ValueError('Packed chunk does not fit the planned bit stream')
        if count == 0:
            return
        if count % 8 and int(packed[-1]) >> (count % 8):
            raise ValueError('Packed chunk has nonzero unused bits')
        start, offset = divmod(self.position, 8)
        if offset == 0:
            self.buffer[start:start + packed.numel()] |= packed
        else:
            # OR is necessary: the first destination byte already contains the
            # preceding chunk's low bits. Last source-byte padding remains zero.
            low = (packed.to(torch.int16) << offset).to(torch.uint8)
            amount = min(low.numel(), self.buffer.numel() - start)
            self.buffer[start:start + amount] |= low[:amount]
            carry = packed >> (8 - offset)
            amount = min(carry.numel(), self.buffer.numel() - start - 1)
            if amount:
                self.buffer[start + 1:start + 1 + amount] |= carry[:amount]
        self.position += count

    def finish(self):
        if self.position != self.bit_count:
            raise ValueError('Packed stream length changed between count and export passes')
        return self.buffer


def _checked_rows(head, start: int, end: int):
    codes, scales = head.hard_codes_rows(start, end)
    shape = (end - start, head.in_features)
    scale_shape = (end - start, head.in_features // head.group_size)
    if not isinstance(codes, Tensor) or codes.dtype != torch.int8 or tuple(codes.shape) != shape:
        raise ValueError('hard_codes_rows must return an int8 matrix matching the requested rows')
    if not bool(((codes == -1) | (codes == 0) | (codes == 1)).all()):
        raise ValueError('Hard codes must contain only -1, 0 or +1')
    if (not isinstance(scales, Tensor) or scales.dtype != torch.float32 or tuple(scales.shape) != scale_shape
            or not bool(torch.isfinite(scales).all()) or not bool((scales > 0).all())):
        raise ValueError('Hard scales must be finite positive FP32 values per row group')
    codes = codes.detach().to(device='cpu').contiguous()
    scales = scales.detach().to(device='cpu').contiguous()
    half = scales.to(torch.float16)
    if not bool(torch.isfinite(half).all()) or not bool((half > 0).all()):
        raise ValueError('Head scales are not representable as positive finite FP16 values')
    return codes, scales, half


def _update_digest(digest, codes, scales):
    # Fixed traversal and dtype/shape contracts make this a streaming digest of
    # the actual hard result. Include FP32 scales to detect pre-rounding changes.
    digest.update(codes.view(torch.uint8).numpy().tobytes())
    digest.update(scales.view(torch.uint8).numpy().tobytes())


def _prototype(encoding, rows, width, group_size, nonzero, provenance, bias, rotation):
    metadata = {'format': FORMAT, 'encoding': encoding, 'shape': [rows, width],
                'module_kind': 'linear', 'source': 'openbonsai-endpoint-calibration',
                'provenance': copy.deepcopy(provenance), 'group_size': group_size,
                'group_axis': 'flattened'}
    tensors = {'scales': torch.empty(rows * (width // group_size), dtype=torch.float16, device='meta')}
    if encoding == 'ternary-2bit':
        tensors['codes'] = torch.empty((rows * width + 3) // 4, dtype=torch.uint8, device='meta')
    else:
        tensors.update(presence=torch.empty((rows * width + 7) // 8, dtype=torch.uint8, device='meta'),
                       signs=torch.empty((nonzero + 7) // 8, dtype=torch.uint8, device='meta'),
                       row_sign_offsets=torch.empty(rows + 1, dtype=torch.int64, device='meta'))
    if bias is not None:
        tensors['bias'] = torch.empty(rows, dtype=torch.float16, device='meta')
    if rotation is not None:
        block_size, _ = rotation
        metadata['rotation'] = {'block_size': block_size, 'convention': 'row-DH',
                                'normalization': 'orthonormal'}
        tensors['rotation_signs'] = torch.empty(width, dtype=torch.int8, device='meta')
    candidate = MixedWeightCandidate(metadata, tensors)
    # Reuse the serializer's exact canonical layout; meta tensors carry only
    # shape/dtype and do not reserve payload memory. Validation runs on real
    # data after packing, never against these intentionally valueless tensors.
    header, payload_bytes = _header(candidate)
    return candidate, _PREFIX + len(header) + payload_bytes + _DIGEST


@torch.no_grad()
def export_head_candidate(head, *, packing='auto', provenance=None, row_chunk_size=None) -> MixedWeightCandidate:
    """Export a BoundedTernaryHead, retaining only one compact codec payload.

    Source identity is checked before and after export by the head's bounded
    hash guard. A supplied ``source_weight_sha256`` must match that original
    BF16 source. Both passes must produce identical hard codes/FP32 scales.
    Positive scales that underflow to zero in FP16 are rejected explicitly.
    ``row_chunk_size`` can reduce, but never exceed, the head's chunk limit.
    """
    if packing not in ('auto', '2bit', 'bitcos'):
        raise ValueError('packing must be auto, 2bit or bitcos')
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError('provenance must be a dictionary')
    rows = _positive_integer(head.out_features, 'out_features')
    width = _positive_integer(head.in_features, 'in_features')
    group_size = _positive_integer(head.group_size, 'group_size')
    maximum_chunk = _positive_integer(head.row_chunk_size, 'head.row_chunk_size')
    chunk = maximum_chunk if row_chunk_size is None else _positive_integer(row_chunk_size, 'row_chunk_size')
    if width % group_size or chunk > maximum_chunk:
        raise ValueError('Groups must fit complete rows and export chunks must fit the head chunk limit')
    if (not isinstance(head.weight, Tensor) or head.weight.dtype != torch.bfloat16
            or tuple(head.weight.shape) != (rows, width)):
        raise ValueError('Head must retain its original BF16 source matrix')
    head.assert_source_unchanged(verify_hash=True)
    head.validate_parameters()
    source_hash = _sha256(head.source_weight_sha256, 'source_weight_sha256')
    details = copy.deepcopy(provenance or {})
    if 'source_weight_sha256' in details and details['source_weight_sha256'] != source_hash:
        raise ValueError('source_weight_sha256 does not match the original BF16 head')
    details.update(source_weight_sha256=source_hash, source_dtype='bfloat16',
                   scale_storage='float16', export_method='bounded-head-hard-ternary-v1')
    # Canonical JSON/type validation happens in _prototype before any payload
    # allocation. Preserve caller provenance; never infer the model revision.
    bias = None
    if head.bias is not None:
        if (not isinstance(head.bias, Tensor) or tuple(head.bias.shape) != (rows,)
                or not head.bias.is_floating_point() or not bool(torch.isfinite(head.bias).all())):
            raise ValueError('Head bias must be a finite vector matching the vocabulary')
        bias = head.bias.detach().to(device='cpu', dtype=torch.float16).contiguous().clone()
        if not bool(torch.isfinite(bias).all()):
            raise ValueError('Head bias is not representable in FP16')
    signs = getattr(head, 'rotation_signs', None)
    rotation = None
    if signs is not None:
        block_size = head.block_size
        validate_rotation(signs, block_size, width)
        rotation = (block_size, signs.detach().to(device='cpu', dtype=torch.int8).contiguous().clone())
    count_digest = hashlib.sha256()
    nonzero = 0
    for start in range(0, rows, chunk):
        end = min(start + chunk, rows)
        codes, scales, half = _checked_rows(head, start, end)
        _update_digest(count_digest, codes, scales)
        nonzero += int((codes != 0).sum())
        del codes, scales, half
    choices = ('2bit', 'bitcos') if packing == 'auto' else (packing,)
    prototypes = [_prototype('ternary-' + codec, rows, width, group_size, nonzero,
                             details, bias, rotation) for codec in choices]
    prototype, planned_bytes = min(prototypes, key=lambda pair: pair[1])
    codec = prototype.encoding.removeprefix('ternary-')
    groups_per_row = width // group_size
    stored_scales = torch.empty(rows * groups_per_row, dtype=torch.float16)
    tensors = {'scales': stored_scales}
    if bias is not None:
        tensors['bias'] = bias
    if rotation is not None:
        tensors['rotation_signs'] = rotation[1]
    if codec == '2bit':
        codes_writer = _BitWriter(rows * width * 2)
    else:
        presence_writer, signs_writer = _BitWriter(rows * width), _BitWriter(nonzero)
        offsets = torch.empty(rows + 1, dtype=torch.int64)
        offsets[0] = 0
    export_digest = hashlib.sha256()
    count = 0
    for start in range(0, rows, chunk):
        end = min(start + chunk, rows)
        codes, scales, half = _checked_rows(head, start, end)
        _update_digest(export_digest, codes, scales)
        stored_scales[start * groups_per_row:end * groups_per_row] = half.reshape(-1)
        if codec == '2bit':
            codes_writer.append(pack_ternary_codes(codes), codes.numel() * 2)
        else:
            present = codes != 0
            counts = present.sum(dim=1, dtype=torch.int64)
            offsets[start + 1:end + 1] = count + counts.cumsum(0)
            chunk_count = int(counts.sum())
            presence_writer.append(_pack_bits(present), codes.numel())
            signs_writer.append(_pack_bits(codes[present] < 0), chunk_count)
            count += chunk_count
        del codes, scales, half
    if count_digest.digest() != export_digest.digest():
        raise ValueError('Hard codes or scales changed between export passes')
    if codec == '2bit':
        tensors['codes'] = codes_writer.finish()
    else:
        tensors.update(presence=presence_writer.finish(), signs=signs_writer.finish(), row_sign_offsets=offsets)
    head.assert_source_unchanged(verify_hash=True)
    head.validate_parameters()
    if source_hash != head.source_weight_sha256:
        raise ValueError('Original source identity changed during export')
    current_signs = getattr(head, 'rotation_signs', None)
    if ((rotation is None) != (current_signs is None) or (rotation is not None and (
            head.block_size != rotation[0] or not torch.equal(current_signs.detach().cpu(), rotation[1])))):
        raise ValueError('Rotation changed during export')
    candidate = MixedWeightCandidate(prototype.metadata, tensors)
    validate_candidate(candidate)
    if candidate_nbytes(candidate) != planned_bytes:
        raise RuntimeError('Actual candidate cost disagrees with the metadata-only size plan')
    return candidate
