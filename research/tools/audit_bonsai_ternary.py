#!/usr/bin/env python3
"""Read-only PTQ1_0 code/scale audit; no model construction or inference.

Requires the pinned Prism GGUF reader and a tiny native decoder wrapper linked
against the already-built pinned libggml-base. All weight reads are mmap views;
analysis temporaries cover at most --chunk-blocks groups. Exact distributions
use 256-byte code lookup tables and a 65,536-bin FP16 histogram, not full weights.
CPU only, little-endian macOS/Linux. Downloads nothing. See
docs/bonsai-ternary-statistics.md for the native-wrapper build and interpretation.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '2'
import numpy as np

PIN = 'c9c34bc9c4042c13a31598ebf43461cd0ad8f00d'
MODEL_SHA = '3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b'
MODEL_BYTES = 1997486976
with np.errstate(invalid='ignore'):  # Enumerates unused signaling-NaN bit patterns too.
    F16_VALUES = np.arange(65536, dtype='<u2').view('<f2').astype(np.float64)
POW3 = np.array([1, 3, 9, 27, 81], dtype=np.uint16)
# C assigns the multiplication to uint8_t BEFORE multiplying by three.
LUT = ((((np.arange(256, dtype=np.uint16)[:, None] * POW3) & 255) * 3) >> 8).astype(np.int8) - 1
COUNT5 = np.stack([(LUT == k).sum(1) for k in (-1, 0, 1)], axis=1)
COUNT4 = np.stack([(LUT[:, :4] == k).sum(1) for k in (-1, 0, 1)], axis=1)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def native_setup(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    lib.audit_init.argtypes = []
    lib.audit_init.restype = None
    for name in ('audit_decode', 'audit_encode'):
        fn = getattr(lib, name)
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        fn.restype = None
    lib.audit_init()
    return lib


def native_decode(lib, blocks):
    blocks = np.ascontiguousarray(blocks, dtype=np.uint8)
    out = np.empty((len(blocks), 128), dtype=np.float32)
    lib.audit_decode(blocks.ctypes.data, out.ctypes.data, out.size)
    return out


def decode_codes(blocks):
    """Source row order: 16 lanes x5, 8 lanes x5, 2 lanes x4."""
    return np.concatenate((LUT[blocks[:, :16]].transpose(0, 2, 1).reshape(-1, 80),
                           LUT[blocks[:, 16:24]].transpose(0, 2, 1).reshape(-1, 40),
                           LUT[blocks[:, 24:26], :4].transpose(0, 2, 1).reshape(-1, 8)), axis=1)


def scale_bits(blocks):
    return blocks[:, 26].astype(np.uint16) | (blocks[:, 27].astype(np.uint16) << 8)


def verify_decoder(lib):
    patterns = [np.full(128, k, dtype=np.float32) for k in (-1, 0, 1)]
    patterns += [np.resize(np.array([-1, 0, 1], dtype=np.float32), 128)]
    for index in range(128):
        for sign in (-1, 1):
            row = np.zeros(128, dtype=np.float32)
            row[index] = sign
            patterns.append(row)
    known = np.stack(patterns) * np.float32(.5)
    packed = np.empty((len(known), 28), dtype=np.uint8)
    lib.audit_encode(known.ctypes.data, packed.ctypes.data, known.size)
    np.testing.assert_array_equal(native_decode(lib, packed), known)
    np.testing.assert_array_equal(decode_codes(packed).astype(np.float32)
                                  * F16_VALUES[scale_bits(packed), None].astype(np.float32), known)
    # Fixed scale preserves the zero-code definition even though native encoding
    # of an all-zero group chooses d=0. For codes 0, qs=128 but qh=127.
    zero = np.zeros((1, 28), dtype=np.uint8)
    zero[:, :24], zero[:, 24:26], zero[:, 26:] = 128, 127, np.array([.5], dtype='<f2').view(np.uint8)
    np.testing.assert_array_equal(decode_codes(zero), np.zeros((1, 128), dtype=np.int8))
    np.testing.assert_array_equal(native_decode(lib, zero), np.zeros((1, 128), dtype=np.float32))
    # Exhaust every byte value in both packed regions; these include noncanonical
    # but valid decoder inputs and catch integer-overflow/zero-encoding mistakes.
    exhaustive = np.repeat(np.arange(256, dtype=np.uint8)[:, None], 28, axis=1)
    exhaustive[:, 26:] = np.array([.5], dtype='<f2').view(np.uint8)
    np.testing.assert_array_equal(native_decode(lib, exhaustive), decode_codes(exhaustive).astype(np.float32) * .5)
    return {'native_encoded_known_blocks': len(known), 'all_byte_values_blocks': 256,
            'fixed_positive_scale_all_zero_code_block': True, 'exact': True,
            'known_patterns': 'all negative/zero/positive; alternating; one +/-1 at each of 128 positions',
            'zero_code_bytes': {'qs': 128, 'qh': 127}}


def distribution(hist):
    present = np.flatnonzero(hist)
    values, counts = F16_VALUES[present], hist[present]
    finite = np.isfinite(values)
    if not bool(finite.all()):
        raise ValueError('Nonfinite scale encountered')
    order = np.argsort(values)
    values, counts = values[order], counts[order]
    total = int(counts.sum())
    cumulative = counts.cumsum()
    mean = float(np.sum(values * counts) / total)
    return {'count': total, 'unique_fp16_bit_patterns': len(present),
            'negative_groups': int(counts[values < 0].sum()), 'zero_groups': int(counts[values == 0].sum()),
            'subnormal_positive_groups': int(counts[(values > 0) & (values < np.finfo(np.float16).tiny)].sum()),
            'min': float(values[0]), 'max': float(values[-1]), 'mean': mean,
            'std': float(np.sqrt(max(0., float(np.sum(values * values * counts) / total) - mean * mean))),
            'quantile_method': 'exact inverse empirical CDF; rank floor(q*(n-1)), no interpolation',
            'quantiles': {str(q): float(values[np.searchsorted(cumulative, int(q * (total - 1)), side='right')])
                          for q in (0., .01, .05, .25, .5, .75, .95, .99, 1.)}}


def family(name):
    return re.sub(r'^blk\.\d+\.', '', name)


def costs(elements, nonzero, rows):
    return {'ptq1_code_bytes': elements // 128 * 26,
            'fp16_scale_bytes': elements // 128 * 2,
            'two_bit_code_bytes': (elements + 3) // 4,
            'bitcos_presence_bytes': (elements + 7) // 8,
            'bitcos_sign_bytes': (nonzero + 7) // 8,
            'bitcos_code_bytes': (elements + 7) // 8 + (nonzero + 7) // 8,
            'flat5trit_code_bytes': (elements + 4) // 5,
            'openbonsai_row_index_bytes': (rows + 1) * 8}


def analyze_tensor(tensor, chunk_blocks, lib):
    elements, nbytes = int(tensor.n_elements), int(tensor.n_bytes)
    if (len(tensor.shape) != 2 or int(tensor.shape[0]) % 128
            or elements % 128 or nbytes != elements // 128 * 28):
        raise ValueError('Expected complete, unpadded PTQ1_0 matrix groups')
    raw = tensor.data.reshape(-1, 28)
    assert np.shares_memory(raw, tensor.data)
    counts = np.zeros(3, dtype=np.int64)
    hist = np.zeros(65536, dtype=np.int64)
    zero_hist = np.zeros(129, dtype=np.int64)
    zero_scale_nonzero_codes = 0
    min_zero, max_zero = (129, None), (-1, None)
    first_subnormal = None
    for first in range(0, len(raw), chunk_blocks):
        block = raw[first:first + chunk_blocks]
        bits = scale_bits(block)
        hist += np.bincount(bits, minlength=65536)
        counts += np.bincount(block[:, :24].reshape(-1), minlength=256) @ COUNT5
        counts += np.bincount(block[:, 24:26].reshape(-1), minlength=256) @ COUNT4
        zero = COUNT5[block[:, :24], 1].sum(1) + COUNT4[block[:, 24:26], 1].sum(1)
        zero_hist += np.bincount(zero, minlength=129)
        zero_scale_nonzero_codes += int((128 - zero)[(bits & 0x7fff) == 0].sum())
        lo, hi = int(np.argmin(zero)), int(np.argmax(zero))
        if int(zero[lo]) < min_zero[0]:
            min_zero = (int(zero[lo]), first + lo)
        if int(zero[hi]) > max_zero[0]:
            max_zero = (int(zero[hi]), first + hi)
        if first_subnormal is None:
            subnormal = np.flatnonzero((bits > 0) & (bits < 1024))
            if len(subnormal):
                first_subnormal = first + int(subnormal[0])
    assert int(counts.sum()) == elements and int(hist.sum()) == len(raw)
    # First/middle/last, zero-count extrema and optional subnormal-scale blocks
    # compared against the pinned C decoder. At most six real blocks resident.
    locations = sorted({0, len(raw) // 2, len(raw) - 1, min_zero[1], max_zero[1]}
                       | ({first_subnormal} if first_subnormal is not None else set()))
    check = np.array(raw[locations], copy=True)
    expected = decode_codes(check).astype(np.float32) * F16_VALUES[scale_bits(check), None].astype(np.float32)
    native = native_decode(lib, check)
    np.testing.assert_array_equal(native, expected)
    latent_zero_counts = (decode_codes(check) == 0).sum(1)
    native_zero_counts = (native == 0).sum(1)
    if bool((F16_VALUES[scale_bits(check)] > 0).all()):
        np.testing.assert_array_equal(native_zero_counts, latent_zero_counts)
    rows = int(np.prod(tensor.shape[1:]))
    result = {'name': tensor.name, 'family': family(tensor.name),
              'shape_ggml_order': list(map(int, tensor.shape)), 'elements': elements,
              'groups': len(raw), 'rows': rows, 'packed_bytes': nbytes,
              'negative_codes': int(counts[0]), 'zero_codes': int(counts[1]), 'positive_codes': int(counts[2]),
              'zero_fraction': float(counts[1] / elements),
              'effective_zero_fraction': float((counts[1] + zero_scale_nonzero_codes) / elements),
              'zero_scale_nonzero_codes': zero_scale_nonzero_codes,
              'group_zero_count_histogram': zero_hist.tolist(), 'scale_distribution': distribution(hist),
              'costs': costs(elements, elements - int(counts[1]), rows),
              'native_decode_verified_block_indices': locations,
              'native_verified_zero_counts': native_zero_counts.tolist(),
              'lookup_verified_zero_counts': latent_zero_counts.tolist(),
              'native_verified_scales': F16_VALUES[scale_bits(check)].tolist(),
              'padding_elements': 0}
    return result, hist, counts, zero_hist


def aggregate(items, hist):
    keys = ('elements', 'groups', 'rows', 'packed_bytes', 'negative_codes', 'zero_codes', 'positive_codes', 'zero_scale_nonzero_codes')
    out = {k: sum(x[k] for x in items) for k in keys}
    out['tensors'] = len(items)
    out['zero_fraction'] = out['zero_codes'] / out['elements']
    out['effective_zero_fraction'] = (out['zero_codes'] + out['zero_scale_nonzero_codes']) / out['elements']
    out['scale_distribution'] = distribution(hist)
    zeros = np.sum([x['group_zero_count_histogram'] for x in items], axis=0)
    present = np.flatnonzero(zeros)
    out['group_zero_count_distribution'] = {'histogram': zeros.tolist(),
        'minimum': int(present[0]), 'maximum': int(present[-1]),
        'groups_with_42_zeros': int(zeros[42]), 'fraction_with_42_zeros': float(zeros[42] / zeros.sum()),
        'interpretation': 'Observed checkpoint structure; neither a packing requirement nor proof of an unpublished training constraint'}
    out['costs'] = {k: sum(x['costs'][k] for x in items) for k in items[0]['costs']}
    out['costs']['bitcos_plus_scales_bytes'] = out['costs']['bitcos_code_bytes'] + out['costs']['fp16_scale_bytes']
    out['costs']['bitcos_plus_scales_and_row_indexes_bytes'] = out['costs']['bitcos_plus_scales_bytes'] + out['costs']['openbonsai_row_index_bytes']
    out['costs']['bitcos_payload_delta_vs_ptq1_bytes'] = out['costs']['bitcos_plus_scales_bytes'] - out['packed_bytes']
    out['costs']['indexed_bitcos_payload_delta_vs_ptq1_bytes'] = out['costs']['bitcos_plus_scales_and_row_indexes_bytes'] - out['packed_bytes']
    out['costs']['bitcos_code_bits_per_weight'] = out['costs']['bitcos_code_bytes'] * 8 / out['elements']
    out['costs']['flat5trit_plus_scales_bytes'] = out['costs']['flat5trit_code_bytes'] + out['costs']['fp16_scale_bytes']
    out['costs']['flat5trit_saving_vs_ptq1_bytes'] = out['packed_bytes'] - out['costs']['flat5trit_plus_scales_bytes']
    out['costs']['flat5trit_saving_vs_bitcos_bytes'] = out['costs']['bitcos_code_bytes'] - out['costs']['flat5trit_code_bytes']
    out['costs']['flat5trit_saving_vs_indexed_bitcos_bytes'] = out['costs']['flat5trit_saving_vs_bitcos_bytes'] + out['costs']['openbonsai_row_index_bytes']
    return out


def prior_openbonsai(path):
    evidence = json.loads(Path(path).read_text())
    for item in evidence['items']:
        raw = item['output']['text']
        try:
            parsed, _ = json.JSONDecoder().raw_decode(raw.lstrip())
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and 'rotated' in parsed and 'candidate_payload' in parsed['rotated']:
            out = {'source_path': str(path), 'sha256': sha256(path), 'source_item_id': item['id'],
                   'scope': 'Completed historical 16x8 decoder-only artifacts; endpoints BF16 and vision retained; not current active pair',
                   'conditions': {}}
            for name in ('unrotated', 'rotated'):
                row = parsed[name]
                p = row['candidate_payload']
                n = p['original_weight_elements']
                signs = p['tensor_bytes_by_key']['signs']
                count = p['codec_distribution']['ternary-bitcos']
                # Sign-stream padding can add 0..7 bits per tensor; payload
                # bytes give a strict occupancy interval, not an exact count.
                out['conditions'][name] = {'elements': n, 'ternary_tensors': count,
                    'zero_fraction_lower_bound': 1 - signs * 8 / n,
                    'zero_fraction_upper_bound': 1 - (signs * 8 - count * 7) / n,
                    'candidate_payload': p, 'file_bytes': row['file_bytes']}
            return out
    raise ValueError('Recorded artifact accounting not found')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--prism-source', type=Path, required=True)
    p.add_argument('--native-library', type=Path, required=True)
    p.add_argument('--prior-evidence', type=Path,
                   help='Optional historical OpenBonsai completed-command evidence JSON; not needed for the official audit')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--chunk-blocks', type=int, default=16384)
    args = p.parse_args()
    if platform.system() not in ('Darwin', 'Linux') or sys.byteorder != 'little':
        p.error('This CPU audit supports little-endian macOS/Linux only')
    import resource
    if not 1 <= args.chunk_blocks <= 32768:
        p.error('chunk-blocks must be between 1 and 32768')
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.output.parent.is_dir():
        raise ValueError('Output parent directory must already exist')
    for name, path in (('model', args.model), ('native-library', args.native_library)):
        if not path.is_file():
            raise ValueError(f'--{name} must name an existing local file')
    start = time.perf_counter()
    historical = prior_openbonsai(args.prior_evidence) if args.prior_evidence is not None else None
    source = args.prism_source.resolve()
    git = lambda *a: subprocess.check_output(['git', '-C', str(source), *a], text=True).strip()
    if git('rev-parse', 'HEAD') != PIN or git('status', '--porcelain', '--untracked-files=all'):
        raise ValueError('Prism source must be clean and pinned')
    if args.model.stat().st_size != MODEL_BYTES or sha256(args.model) != MODEL_SHA:
        raise ValueError('Official model identity mismatch')
    sys.path.insert(0, str(source / 'gguf-py'))
    from gguf import GGUFReader, GGMLQuantizationType
    lib = native_setup(args.native_library)
    verification = verify_decoder(lib)
    reader = GGUFReader(str(args.model), 'r')
    if reader.byte_order != 'I' or sys.byteorder != 'little':
        raise ValueError('This bounded audit requires native little-endian GGUF')
    tensors, histograms = [], {}
    total_hist = np.zeros(65536, dtype=np.int64)
    for tensor in reader.tensors:
        if tensor.tensor_type != GGMLQuantizationType.PTQ1_0:
            continue
        result, hist, _, _ = analyze_tensor(tensor, args.chunk_blocks, lib)
        tensors.append(result)
        histograms.setdefault(result['family'], np.zeros(65536, dtype=np.int64))[:] += hist
        total_hist += hist
        if len(tensors) % 25 == 0:
            print(f'inspected {len(tensors)} ternary tensors', flush=True)
    families = {name: aggregate([x for x in tensors if x['family'] == name], hist)
                for name, hist in histograms.items()}
    total = aggregate(tensors, total_hist)
    decoder = aggregate([x for x in tensors if x['name'].startswith('blk.')],
                        sum((hist for name, hist in histograms.items()
                             if name not in ('output.weight', 'token_embd.weight')), np.zeros(65536, dtype=np.int64)))
    retained_bytes = sum(int(t.n_bytes) for t in reader.tensors if t.tensor_type != GGMLQuantizationType.PTQ1_0)
    container = MODEL_BYTES - total['packed_bytes'] - retained_bytes
    indexed_choice = sum(min(x['packed_bytes'], x['costs']['bitcos_code_bytes'] + x['costs']['fp16_scale_bytes']
                             + x['costs']['openbonsai_row_index_bytes']) for x in tensors)
    report = {'format': 'openbonsai.official-ternary-statistics.v1', 'quality_claim': False,
              'inference_executed': False, 'gpu_used': False, 'cpu_thread_limit': 2,
              'source': {'model_repository': 'prism-ml/Ternary-Bonsai-2-9B-gguf',
                  'model_revision': '5b2abb02362efbed46bbe204014b02837778b2c2',
                  'model_sha256': MODEL_SHA, 'model_bytes': MODEL_BYTES, 'runtime_sha': PIN,
                  'reader_sha256': sha256(source / 'gguf-py/gguf/gguf_reader.py'),
                  'quants_source_sha256': sha256(source / 'ggml/src/ggml-quants.c'),
                  'struct_source_sha256': sha256(source / 'ggml/src/ggml-common.h'),
                  'native_library_sha256': sha256(args.native_library),
                  'native_wrapper_source_sha256': sha256(Path(__file__).with_name('bonsai_ternary_decoder.cpp')),
                  'script_sha256': sha256(Path(__file__)), 'numpy_version': np.__version__},
              'scope': 'All 202 stored language-model ternary matrices including embedding/head; no vision/MTP; latent Hadamard basis',
              'decoder_verification': verification,
              'measurement': {'chunk_blocks': args.chunk_blocks, 'max_packed_chunk_bytes': args.chunk_blocks * 28,
                  'method': 'memory map plus byte lookup tables and exact FP16 histograms; no full reconstructed weight',
                  'scale_distribution_weighting': 'one observation per 128-weight group, including zero groups',
                  'native_real_blocks_verified': sum(len(x['native_decode_verified_block_indices']) for x in tensors)},
              'totals': total, 'decoder_only': decoder, 'families': families, 'tensors': tensors,
              'hypothetical_file_payload_substitution': {
                  'retained_other_tensor_bytes': retained_bytes, 'existing_container_metadata_alignment_bytes': container,
                  'all_bitcos_without_indexes_bytes': container + retained_bytes + total['costs']['bitcos_plus_scales_bytes'],
                  'all_bitcos_with_row_indexes_bytes': container + retained_bytes + total['costs']['bitcos_plus_scales_and_row_indexes_bytes'],
                  'all_flat5trit_bytes': container + retained_bytes + total['costs']['flat5trit_plus_scales_bytes'],
                  'per_tensor_min_ptq1_or_indexed_bitcos_bytes': container + retained_bytes + indexed_choice,
                  'limitations': 'Arithmetic payload substitutions only, not constructed GGUF/OpenBonsai artifacts. Existing metadata is held fixed. New codec headers/runtime support and transform representation changes are excluded. Original FP16 scales and trit codes are preserved.'},
              'historical_openbonsai_comparison': historical}
    report['measurement']['elapsed_seconds'] = time.perf_counter() - start
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['measurement']['peak_rss_bytes'] = usage if platform.system() == 'Darwin' else usage * 1024
    report['measurement']['rss_note'] = 'RSS includes file-backed mmap pages and tokenizer metadata; it is not decoded weight storage.'
    # Exclusive creation also refuses a path created after the initial check.
    with args.output.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(args.output), 'zero_fraction': total['zero_fraction'],
                      'families': {k: v['zero_fraction'] for k,v in families.items()},
                      'costs': total['costs'], 'elapsed_seconds': report['measurement']['elapsed_seconds']}, indent=2))


if __name__ == '__main__':
    main()
