"""Deterministic candidate storage and slow, bounded-row reference operators.

Imports learned ternary or externally fitted GPTQ/GSQ integer codes. It does
not fit a quantizer. Affine reconstruction is (q - zero) * scale with per-row
input groups. Ternary grouping preserves the existing flattened convention.
Storage bytes include the canonical JSON header, all tensors and checksum.
No low-bit inference acceleration is supplied; ``weight`` reconstructs dense
floats, while forwards decode row chunks (only requested embedding rows).
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .packing import (BITCOS_FORMAT, FORMAT as TERNARY_FORMAT,
                      repack_grouped_weight, unpack_grouped_codes)
from .hadamard import inverse_rotate, rotate, validate_rotation

FORMAT = "openbonsai.mixed-candidate.v1"
MAGIC = b"OBMIXED1\x00"
_PREFIX = len(MAGIC) + 8
_DIGEST = 32
_MAX_HEADER = 1024 * 1024
_DTYPES = {"uint8": torch.uint8, "int8": torch.int8, "int16": torch.int16,
           "int32": torch.int32, "int64": torch.int64, "float16": torch.float16,
           "bfloat16": torch.bfloat16, "float32": torch.float32}
_FLOATS = (torch.float16, torch.bfloat16, torch.float32)
_CHUNK_ROWS = 64


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _shape(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("shape must contain two positive dimensions")
    return tuple(_positive_int(x, "shape dimension") for x in value)


def _json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False, ensure_ascii=True).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("metadata must be finite JSON data") from error


def _metadata(encoding, shape, module_kind, source, provenance, **kwargs):
    if module_kind not in ("linear", "embedding"):
        raise ValueError("module_kind must be linear or embedding")
    if not isinstance(source, str) or not source or len(source) > 1024:
        raise ValueError("source must be a nonempty string of at most 1024 characters")
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError("provenance must be a dictionary")
    return {"format": FORMAT, "encoding": encoding, "shape": list(_shape(shape)),
            "module_kind": module_kind, "source": source,
            "provenance": copy.deepcopy(provenance or {}), **kwargs}


@dataclass
class MixedWeightCandidate:
    metadata: dict[str, Any]
    tensors: dict[str, Tensor]

    @property
    def shape(self):
        return tuple(self.metadata["shape"])

    @property
    def module_kind(self):
        return self.metadata["module_kind"]

    @property
    def encoding(self):
        return self.metadata["encoding"]

    def dequantize(self, *, device=None, dtype=torch.float32):
        return candidate_weight(self, device=device, dtype=dtype)


def _pack_unsigned(values, bits):
    values = values.detach().reshape(-1).cpu().to(torch.int64)
    lanes = 8 // math.gcd(bits, 8)
    if values.numel() % lanes:
        values = F.pad(values, (0, lanes - values.numel() % lanes))
    groups = values.reshape(-1, lanes)
    words = torch.zeros(groups.shape[0], dtype=torch.int64)
    for lane in range(lanes):
        words |= groups[:, lane] << (bits * lane)
    byte_count = lanes * bits // 8
    return torch.stack([(words >> (8 * lane)).to(torch.uint8)
                        for lane in range(byte_count)], dim=1).flatten()


def _unsigned_at(stream, positions, bits):
    bit_positions = positions * bits
    byte_positions = bit_positions // 8
    offsets = bit_positions % 8
    low = stream[byte_positions].to(torch.int64)
    high = stream[(byte_positions + 1).clamp(max=stream.numel() - 1)].to(torch.int64)
    return ((low | (high << 8)) >> offsets) & ((1 << bits) - 1)


def _padding(stream, count, bits, name):
    expected = (count * bits + 7) // 8
    if stream.dtype != torch.uint8 or stream.ndim != 1 or stream.numel() != expected:
        raise ValueError(f"invalid {name} stream size or dtype")
    used = (count * bits) % 8
    if used and int(stream[-1].item()) >> used:
        raise ValueError(f"nonzero {name} padding")


def _bias(bias, rows, module_kind, *, require_bf16=False):
    if bias is None:
        return {}
    if module_kind == "embedding":
        raise ValueError("embedding candidates cannot have bias")
    if (not isinstance(bias, Tensor) or bias.shape != (rows,)
            or bias.dtype not in _FLOATS or not bool(torch.isfinite(bias).all())):
        raise ValueError("bias must be a finite floating vector matching rows")
    if require_bf16 and bias.dtype != torch.bfloat16:
        raise ValueError("BF16 fallback requires an original BF16 bias")
    return {"bias": bias.detach().cpu().contiguous().clone()}


def bf16_candidate(weight, bias=None, *, module_kind="linear", source="original-bf16", provenance=None):
    """Keep original BF16 values; reject implicit rounding from other dtypes."""
    if (not isinstance(weight, Tensor) or weight.dtype != torch.bfloat16
            or weight.ndim != 2 or not bool(torch.isfinite(weight).all())):
        raise ValueError("BF16 fallback requires a finite original BF16 matrix")
    metadata = _metadata("bf16", weight.shape, module_kind, source, provenance)
    tensors = {"weight": weight.detach().cpu().contiguous().clone(),
               **_bias(bias, weight.shape[0], module_kind, require_bf16=True)}
    candidate = MixedWeightCandidate(metadata, tensors)
    validate_candidate(candidate)
    return candidate


def affine_candidate(qcodes, scales, zeros, *, bits, group_size, bias=None,
                     module_kind="linear", source="gptq", provenance=None):
    """Import fitted unsigned codes and zero points; preserve scale precision.

    qcodes: [rows,width] integer values in [0,2**bits-1]. scales and zeros:
    [rows,ceil(width/group_size)]. Integer zero points may lie outside the
    unsigned code range (as in asymmetric positive-only GPTQ groups). No
    act-order permutation is inferred; the
    producer must undo permutations or supply ordinary input-order codes.
    """
    if type(bits) is not int or bits not in (4, 6):
        raise ValueError("affine bits must be 4 or 6")
    group_size = _positive_int(group_size, "group_size")
    if not isinstance(qcodes, Tensor) or qcodes.ndim != 2 or qcodes.dtype not in (
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("qcodes must be a two-dimensional integer tensor")
    rows, width = _shape(qcodes.shape)
    groups = (width + group_size - 1) // group_size
    if not bool(((qcodes >= 0) & (qcodes < (1 << bits))).all()):
        raise ValueError("qcodes outside representable range")
    if (not isinstance(scales, Tensor) or scales.shape != (rows, groups)
            or scales.dtype not in _FLOATS or not bool(torch.isfinite(scales).all())
            or not bool((scales > 0).all())):
        raise ValueError("scales must be finite positive floating values per row group")
    if (not isinstance(zeros, Tensor) or zeros.shape != (rows, groups)
            or zeros.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
        raise ValueError("zeros must be integer zero points per row group")
    low, high = int(zeros.min()), int(zeros.max())
    zero_dtype = torch.int64
    for choice in (torch.uint8, torch.int8, torch.int16, torch.int32):
        limits = torch.iinfo(choice)
        if limits.min <= low and high <= limits.max:
            zero_dtype = choice
            break
    # _pack_unsigned pads whole lanes; trim to exact byte count for partial rows.
    packed = _pack_unsigned(qcodes, bits)[:(rows * width * bits + 7) // 8]
    candidate = MixedWeightCandidate(
        _metadata("affine", (rows, width), module_kind, source, provenance,
                  bits=bits, group_size=group_size, group_axis="row-input"),
        {"codes": packed, "scales": scales.detach().cpu().contiguous().clone(),
         "zeros": zeros.detach().cpu().to(zero_dtype).contiguous(),
         **_bias(bias, rows, module_kind)})
    validate_candidate(candidate)
    return candidate


def ternary_candidate(state, *, module_kind="linear", packing="auto", source="scaleq", provenance=None):
    """Import learned existing packed ternary values without refitting/scaling.

    Auto compares the complete serialized candidates, including BITCOS row
    indexes and differing JSON metadata, rather than nominal code bits.
    """
    if packing not in ("auto", "2bit", "bitcos"):
        raise ValueError("packing must be auto, 2bit or bitcos")
    choices = []
    for codec in (("2bit", "bitcos") if packing == "auto" else (packing,)):
        packed = repack_grouped_weight(state, packing=codec)
        rows, width = packed["shape"]
        tensors = {key: value.clone() for key, value in packed.items()
                   if isinstance(value, Tensor) and key != "bias"}
        tensors.update(_bias(packed.get("bias"), rows, module_kind))
        if codec == "bitcos":
            codes = unpack_grouped_codes(packed)[:rows * width].reshape(rows, width)
            counts = (codes != 0).sum(dim=1, dtype=torch.int64)
            tensors["row_sign_offsets"] = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
        candidate = MixedWeightCandidate(
            _metadata("ternary-" + codec, (rows, width), module_kind, source, provenance,
                      group_size=packed["group_size"], group_axis="flattened"), tensors)
        validate_candidate(candidate)
        choices.append(candidate)
    return min(choices, key=candidate_nbytes)


def with_rotation(candidate, signs, block_size):
    """Attach transforms to ALREADY rotated latent weights; never refit them.

    The caller must construct the candidate from W R (or E R), not W/E.
    Stored signs are authoritative; there is no seed-dependent regeneration.
    ``candidate_weight`` continues to return the latent matrix. Runtime linear
    inputs use R and runtime embedding lookup outputs use R transpose.
    """
    validate_candidate(candidate)
    if "rotation" in candidate.metadata:
        raise ValueError("candidate already has a rotation")
    validate_rotation(signs, block_size, candidate.shape[1])
    metadata = copy.deepcopy(candidate.metadata)
    metadata["rotation"] = {"block_size": block_size, "convention": "row-DH",
                            "normalization": "orthonormal"}
    tensors = {key: value.detach().cpu().contiguous().clone() for key, value in candidate.tensors.items()}
    tensors["rotation_signs"] = signs.detach().cpu().to(torch.int8).contiguous().clone()
    result = MixedWeightCandidate(metadata, tensors)
    validate_candidate(result)
    return result


def validate_candidate(candidate):
    if not isinstance(candidate, MixedWeightCandidate):
        raise ValueError("expected MixedWeightCandidate")
    m, t = candidate.metadata, candidate.tensors
    if not isinstance(m, dict) or not isinstance(t, dict) or m.get("format") != FORMAT:
        raise ValueError("invalid candidate metadata")
    common = {"format", "encoding", "shape", "module_kind", "source", "provenance"}
    rows, width = _shape(m.get("shape"))
    _metadata(m.get("encoding"), (rows, width), m.get("module_kind"), m.get("source"), m.get("provenance"))
    if not isinstance(m.get("provenance"), dict):
        raise ValueError("provenance must be a dictionary")
    if len(_json(m)) > _MAX_HEADER // 2:
        raise ValueError("candidate metadata too large")
    if any(not isinstance(key, str) or not isinstance(value, Tensor) for key, value in t.items()):
        raise ValueError("invalid tensor dictionary")
    if len({value.device for value in t.values()}) > 1:
        raise ValueError("candidate tensors must share a device")
    _bias(t.get("bias"), rows, m["module_kind"], require_bf16=m["encoding"] == "bf16")
    extra = {"bias"} if "bias" in t else set()
    if "rotation" in m:
        rotation = m["rotation"]
        if (not isinstance(rotation, dict) or set(rotation) != {"block_size", "convention", "normalization"}
                or rotation["convention"] != "row-DH" or rotation["normalization"] != "orthonormal"):
            raise ValueError("unsupported rotation contract")
        signs = t.get("rotation_signs")
        validate_rotation(signs, rotation["block_size"], width)
        if signs.dtype != torch.int8:
            raise ValueError("stored rotation signs must be int8")
        common.add("rotation")
        extra.add("rotation_signs")
    if m["encoding"] == "bf16":
        if set(m) != common or set(t) != {"weight"} | extra:
            raise ValueError("unexpected BF16 metadata or tensors")
        if t["weight"].dtype != torch.bfloat16 or t["weight"].shape != (rows, width) or not bool(torch.isfinite(t["weight"]).all()):
            raise ValueError("invalid BF16 matrix")
        return
    group_size = _positive_int(m.get("group_size"), "group_size")
    if m["encoding"] == "affine":
        if set(m) != common | {"bits", "group_size", "group_axis"} or m["group_axis"] != "row-input":
            raise ValueError("invalid affine metadata")
        if type(m["bits"]) is not int or m["bits"] not in (4, 6):
            raise ValueError("invalid affine bit width")
        if set(t) != {"codes", "scales", "zeros"} | extra:
            raise ValueError("unexpected affine tensors")
        _padding(t["codes"], rows * width, m["bits"], "codes")
        shape = (rows, (width + group_size - 1) // group_size)
        if (t["scales"].dtype not in _FLOATS or t["scales"].shape != shape
                or not bool(torch.isfinite(t["scales"]).all()) or not bool((t["scales"] > 0).all())):
            raise ValueError("invalid affine scales")
        if t["zeros"].dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64) or t["zeros"].shape != shape:
            raise ValueError("invalid affine zero points")
        return
    if m["encoding"] not in ("ternary-2bit", "ternary-bitcos"):
        raise ValueError("unsupported candidate encoding")
    if set(m) != common | {"group_size", "group_axis"} or m["group_axis"] != "flattened":
        raise ValueError("invalid ternary metadata")
    count = rows * width
    groups = (count + group_size - 1) // group_size
    padded = groups * group_size
    scales = t.get("scales")
    if (not isinstance(scales, Tensor) or scales.dtype != torch.float16 or scales.shape != (groups,)
            or not bool(torch.isfinite(scales).all()) or bool((scales < 0).any())):
        raise ValueError("invalid ternary scales")
    if m["encoding"] == "ternary-2bit":
        if set(t) != {"codes", "scales"} | extra:
            raise ValueError("unexpected ternary tensors")
        _padding(t["codes"], padded, 2, "codes")
        for start in range(0, padded, 1048576):
            positions = torch.arange(start, min(start + 1048576, padded), device=scales.device)
            codes = _unsigned_at(t["codes"], positions, 2)
            if bool((codes == 3).any()) or bool((codes[positions >= count] != 0).any()):
                raise ValueError("invalid ternary codes or group padding")
    else:
        if set(t) != {"presence", "signs", "scales", "row_sign_offsets"} | extra:
            raise ValueError("unexpected BITCOS tensors")
        _padding(t["presence"], padded, 1, "presence")
        offsets = t["row_sign_offsets"]
        if offsets.dtype != torch.int64 or offsets.shape != (rows + 1,) or int(offsets[0]) != 0:
            raise ValueError("invalid BITCOS row index")
        expected_offset = 0
        for start in range(0, rows, _CHUNK_ROWS):
            end = min(start + _CHUNK_ROWS, rows)
            pos = torch.arange(start * width, end * width, device=scales.device)
            present = _unsigned_at(t["presence"], pos, 1).reshape(end - start, width)
            expected = expected_offset + present.sum(1).cumsum(0)
            if not torch.equal(offsets[start + 1:end + 1], expected):
                raise ValueError("BITCOS row index disagrees with presence stream")
            expected_offset = int(expected[-1])
        if padded > count:
            pos = torch.arange(count, padded, device=scales.device)
            if bool((_unsigned_at(t["presence"], pos, 1) != 0).any()):
                raise ValueError("nonzero ternary group padding")
        _padding(t["signs"], expected_offset, 1, "signs")


def _rows(candidate, rows, dtype, device):
    m, t = candidate.metadata, candidate.tensors
    source_device = next(iter(t.values())).device
    ids = rows.to(device=source_device, dtype=torch.int64)
    width = m["shape"][1]
    if m["encoding"] == "bf16":
        return t["weight"][ids].to(device=device, dtype=dtype)
    pos = ids[:, None] * width + torch.arange(width, device=source_device)[None, :]
    if m["encoding"] == "affine":
        # Stored FP32 scales/zero points define the reconstructed weight. Do
        # not round metadata to BF16 before reconstruction; round its result.
        reconstruction_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        codes = _unsigned_at(t["codes"], pos, m["bits"]).to(reconstruction_dtype)
        group = torch.arange(width, device=source_device) // m["group_size"]
        scales = t["scales"][ids[:, None], group[None, :]].to(reconstruction_dtype)
        zeros = t["zeros"][ids[:, None], group[None, :]].to(reconstruction_dtype)
        value = (codes - zeros) * scales
    else:
        if m["encoding"] == "ternary-2bit":
            codes = _unsigned_at(t["codes"], pos, 2)
            codes = torch.where(codes == 2, -1, codes)
        else:
            present = _unsigned_at(t["presence"], pos, 1)
            sign_pos = t["row_sign_offsets"][ids, None] + present.cumsum(1) - 1
            codes = torch.zeros_like(present)
            mask = present.bool()
            if bool(mask.any()):
                signs = _unsigned_at(t["signs"], sign_pos[mask], 1)
                codes[mask] = 1 - 2 * signs
        value = codes.to(dtype) * t["scales"][pos // m["group_size"]].to(dtype)
    return value.to(device=device, dtype=dtype)


def candidate_weight(candidate, *, device=None, dtype=torch.float32):
    """Reconstruct the complete dense reference matrix, using bounded scratch."""
    validate_candidate(candidate)
    if dtype not in _FLOATS and dtype != torch.float64:
        raise ValueError("floating reconstruction dtype required")
    source_device = next(iter(candidate.tensors.values())).device
    device = source_device if device is None else device
    output = torch.empty(candidate.shape, dtype=dtype, device=device)
    for start in range(0, candidate.shape[0], _CHUNK_ROWS):
        end = min(start + _CHUNK_ROWS, candidate.shape[0])
        output[start:end] = _rows(candidate, torch.arange(start, end), dtype, device)
    return output


def _header(candidate):
    descriptions, offset = {}, 0
    for name, tensor in sorted(candidate.tensors.items()):
        dtype = str(tensor.dtype).removeprefix("torch.")
        size = tensor.numel() * tensor.element_size()
        descriptions[name] = {"dtype": dtype, "shape": list(tensor.shape), "offset": offset, "bytes": size}
        offset += size
    header = _json({"metadata": candidate.metadata, "tensors": descriptions, "payload_bytes": offset})
    if len(header) > _MAX_HEADER:
        raise ValueError("serialized candidate header too large")
    return header, offset


def candidate_size_breakdown(candidate):
    validate_candidate(candidate)
    header, payload = _header(candidate)
    return {"prefix_bytes": _PREFIX, "header_bytes": len(header), "payload_bytes": payload,
            "checksum_bytes": _DIGEST, "total_bytes": _PREFIX + len(header) + payload + _DIGEST,
            "tensor_bytes": {name: tensor.numel() * tensor.element_size()
                             for name, tensor in sorted(candidate.tensors.items())}}


def candidate_nbytes(candidate):
    return candidate_size_breakdown(candidate)["total_bytes"]


def _segments(candidate):
    if sys.byteorder != "little":
        raise RuntimeError("candidate encoding currently requires a little-endian host")
    validate_candidate(candidate)
    header, _ = _header(candidate)
    yield MAGIC + struct.pack("<Q", len(header)) + header
    for _, tensor in sorted(candidate.tensors.items()):
        raw = tensor.detach().cpu().contiguous().view(torch.uint8).reshape(-1)
        for start in range(0, raw.numel(), 4 * 1024 * 1024):
            yield raw[start:start + 4 * 1024 * 1024].numpy().tobytes()


def encode_candidate(candidate):
    """Return deterministic bytes; use save_candidate for bounded serialization."""
    digest, parts = hashlib.sha256(), []
    for segment in _segments(candidate):
        digest.update(segment)
        parts.append(segment)
    return b"".join(parts) + digest.digest()


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def decode_candidate(blob):
    """Verify checksum and all contracts before returning owned CPU tensors."""
    if not isinstance(blob, (bytes, bytearray)) or len(blob) < _PREFIX + _DIGEST:
        raise ValueError("truncated candidate")
    if blob[:len(MAGIC)] != MAGIC:
        raise ValueError("invalid candidate magic")
    if hashlib.sha256(memoryview(blob)[:-_DIGEST]).digest() != blob[-_DIGEST:]:
        raise ValueError("candidate checksum mismatch")
    size, = struct.unpack("<Q", blob[len(MAGIC):_PREFIX])
    if size > _MAX_HEADER or _PREFIX + size + _DIGEST > len(blob):
        raise ValueError("invalid candidate header length")
    try:
        header = json.loads(blob[_PREFIX:_PREFIX + size], object_pairs_hook=_no_duplicate_keys,
                            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("invalid candidate JSON") from error
    if not isinstance(header, dict) or set(header) != {"metadata", "tensors", "payload_bytes"}:
        raise ValueError("invalid candidate header schema")
    if type(header["payload_bytes"]) is not int or header["payload_bytes"] < 0:
        raise ValueError("invalid payload byte count")
    if _PREFIX + size + header["payload_bytes"] + _DIGEST != len(blob):
        raise ValueError("candidate file length mismatch")
    if not isinstance(header["tensors"], dict) or not header["tensors"]:
        raise ValueError("missing tensor descriptions")
    tensors, offset = {}, 0
    for name, description in sorted(header["tensors"].items()):
        if not isinstance(description, dict) or set(description) != {"dtype", "shape", "offset", "bytes"}:
            raise ValueError("invalid tensor description")
        dtype = _DTYPES.get(description["dtype"]) if isinstance(description["dtype"], str) else None
        shape = description["shape"]
        if dtype is None or not isinstance(shape, list) or not shape or any(type(x) is not int or x < 0 for x in shape):
            raise ValueError("invalid tensor dtype or shape")
        expected = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        if (type(description["offset"]) is not int or description["offset"] != offset
                or type(description["bytes"]) is not int or description["bytes"] != expected
                or offset + expected > header["payload_bytes"]):
            raise ValueError("tensor payload layout mismatch")
        begin = _PREFIX + size + offset
        if expected:
            raw = bytearray(memoryview(blob)[begin:begin + expected])
            tensors[name] = torch.frombuffer(raw, dtype=dtype).reshape(shape).clone()
        else:
            tensors[name] = torch.empty(shape, dtype=dtype)
        offset += expected
    if offset != header["payload_bytes"]:
        raise ValueError("unreferenced payload bytes")
    result = MixedWeightCandidate(header["metadata"], tensors)
    validate_candidate(result)
    if encode_candidate(result) != blob:
        raise ValueError("noncanonical candidate encoding")
    return result


def save_candidate(candidate, path):
    """Atomically publish one candidate; never overwrite an existing file."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name, delete=False) as handle:
            temporary = Path(handle.name)
            digest = hashlib.sha256()
            for segment in _segments(candidate):
                handle.write(segment)
                digest.update(segment)
            handle.write(digest.digest())
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-link publication is atomic and fails if another writer won.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return candidate_nbytes(candidate)


def load_candidate(path):
    """Read a candidate into CPU memory; serialized size is part of peak RAM."""
    return decode_candidate(Path(path).read_bytes())


class _MixedBase(nn.Module):
    def __init__(self, candidate, *, dtype=torch.bfloat16, device=None, row_chunk_size=64):
        super().__init__()
        validate_candidate(candidate)
        if dtype not in _FLOATS:
            raise ValueError("unsupported compute dtype")
        self.row_chunk_size = _positive_int(row_chunk_size, "row_chunk_size")
        self.metadata = copy.deepcopy(candidate.metadata)
        self._tensor_names = tuple(sorted(candidate.tensors))
        for name, value in candidate.tensors.items():
            self.register_buffer("payload_" + name, value.detach().clone().to(device=device))
        self.register_buffer("_compute_reference", torch.empty(0, dtype=dtype, device=device), persistent=False)

    def _candidate(self):
        return MixedWeightCandidate(self.metadata, {name: getattr(self, "payload_" + name) for name in self._tensor_names})

    def _apply(self, fn, recurse=True):
        original = {name: getattr(self, "payload_" + name) for name in self._tensor_names}
        result = super()._apply(fn, recurse=recurse)
        for name, value in original.items():
            target = getattr(self, "payload_" + name)
            self._buffers["payload_" + name] = value.to(device=target.device, dtype=value.dtype)
        return result

    @property
    def weight(self):
        return candidate_weight(self._candidate(), device=self._compute_reference.device,
                                dtype=self._compute_reference.dtype)

    @property
    def bias(self):
        return getattr(self, "payload_bias", None)


class MixedLinear(_MixedBase):
    """Frozen row-chunk dense matmul reference; no low-bit kernel or speed claim."""
    def __init__(self, candidate, **kwargs):
        if candidate.module_kind != "linear":
            raise ValueError("MixedLinear requires a linear candidate")
        super().__init__(candidate, **kwargs)
        self.out_features, self.in_features = candidate.shape

    def forward(self, x):
        if x.shape[-1] != self.in_features or not x.is_floating_point():
            raise ValueError("linear input has incompatible width or dtype")
        candidate = self._candidate()
        if "rotation" in self.metadata:
            x = rotate(x, self.payload_rotation_signs, self.metadata["rotation"]["block_size"])
        chunks = []
        for start in range(0, self.out_features, self.row_chunk_size):
            end = min(start + self.row_chunk_size, self.out_features)
            weight = _rows(candidate, torch.arange(start, end), x.dtype, x.device)
            bias = None if self.bias is None else self.bias[start:end].to(device=x.device, dtype=x.dtype)
            chunks.append(F.linear(x, weight, bias))
        return torch.cat(chunks, dim=-1)


class MixedEmbedding(_MixedBase):
    """Frozen embedding reference: decode unique requested rows in bounded chunks."""
    def __init__(self, candidate, *, padding_idx=None, **kwargs):
        if candidate.module_kind != "embedding":
            raise ValueError("MixedEmbedding requires an embedding candidate")
        super().__init__(candidate, **kwargs)
        self.num_embeddings, self.embedding_dim = candidate.shape
        if padding_idx is not None:
            if type(padding_idx) is not int or not -self.num_embeddings <= padding_idx < self.num_embeddings:
                raise ValueError("invalid padding_idx")
            padding_idx %= self.num_embeddings
        self.padding_idx = padding_idx

    def forward(self, input):
        if input.dtype not in (torch.int32, torch.int64):
            raise ValueError("embedding IDs must be int32 or int64")
        if bool(((input < 0) | (input >= self.num_embeddings)).any()):
            raise IndexError("embedding ID outside vocabulary")
        unique, reverse = torch.unique(input.reshape(-1), sorted=True, return_inverse=True)
        gathered = torch.empty((unique.numel(), self.embedding_dim),
                               device=input.device, dtype=self._compute_reference.dtype)
        candidate = self._candidate()
        for start in range(0, unique.numel(), self.row_chunk_size):
            end = min(start + self.row_chunk_size, unique.numel())
            rows = _rows(candidate, unique[start:end], gathered.dtype, gathered.device)
            if "rotation" in self.metadata:
                rows = inverse_rotate(rows, self.payload_rotation_signs, self.metadata["rotation"]["block_size"])
            gathered[start:end] = rows
        return gathered[reverse].reshape(*input.shape, self.embedding_dim)
