"""Stored-head cache KL: real native caches, independent dense oracle, no fit."""
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file

from openbonsai.adapter import tiny_model
from openbonsai.endpoint_export import export_head_candidate
from openbonsai.endpoint_quantizer import BoundedTernaryHead
from openbonsai.hadamard import rotate
from openbonsai.head_calibration import cache_original_head_inputs
from openbonsai.head_evaluation import evaluate_packed_head_from_cache
from openbonsai import head_evaluation as evaluation
from openbonsai import mixed_formats
from openbonsai.mixed_formats import MixedLinear, save_candidate, load_candidate
from openbonsai.rotations import fixed_signs

SOURCE = {'source_checkpoint': 'native-random-head-evaluation-fixture', 'source_revision': 'seed-413'}


def fixture(tmp_path, *, rotated=False, codec='base3', bias=False, role='calibration'):
    model = tiny_model(seed=413, multimodal=True).to(torch.bfloat16)
    if bias:
        # BF16 subnormals need not survive the separate FP16 export rounding.
        b = torch.linspace(-.2, .2, model.lm_head.out_features).to(torch.bfloat16)
        b[0] = 1e-10
        model.lm_head.bias = nn.Parameter(b, requires_grad=False)
    records = [
        {'input_ids': torch.tensor([[0, 4, 5, 6, 2, 0], [7, 8, 9, 2, 0, 0]]),
         'attention_mask': torch.tensor([[0, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0]]),
         'labels': torch.tensor([[-100, 4, -100, 6, 2, -100], [7, 8, 9, 2, -100, -100]])},
        {'input_ids': torch.tensor([[11, 12, 13, 2]]), 'record_id': 'second-record'},
    ]
    cache = cache_original_head_inputs(model, records, tmp_path / 'cache', SOURCE,
                                       role=role, shard_positions=3)
    signs = fixed_signs(128, seed=5090, name='lm_head') if rotated else None
    with torch.random.fork_rng():
        torch.manual_seed(414)
        head = BoundedTernaryHead(model.lm_head, group_size=16, row_chunk_size=37,
                                  signs=signs, block_size=32, lora_rank=2)
        with torch.no_grad():
            head.lora_B.normal_(0, .002)
            head.raw_scale.add_(.037)
    head.begin_step(1.)
    candidate = export_head_candidate(head, packing=codec, provenance={
        **SOURCE, 'source_provenance': SOURCE, 'pipeline_job_sha256': 'a' * 64})
    path = tmp_path / 'head.obc'
    save_candidate(candidate, path)
    packed = MixedLinear(load_candidate(path), dtype=torch.bfloat16, row_chunk_size=37)
    return model.lm_head, head, packed, cache, path


def files(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def snapshot(*modules):
    tensors, grads, modes = {}, {}, {}
    for i, module in enumerate(modules):
        modes[i] = [(name, child.training) for name, child in module.named_modules()]
        for name, tensor in (*module.named_parameters(), *module.named_buffers()):
            tensors[i, name] = (id(tensor), tensor.data_ptr(), tensor._version,
                                tensor.requires_grad, tensor.detach().clone())
        for name, p in module.named_parameters():
            grads[i, name] = None if p.grad is None else (
                id(p.grad), p.grad._version, p.grad.clone())
    return {'tensors': tensors, 'gradients': grads, 'modes': modes,
            'torch_rng': torch.random.get_rng_state().clone(), 'python_rng': random.getstate()}


def unchanged(before, *modules):
    after = snapshot(*modules)
    assert before['modes'] == after['modes'] and before['python_rng'] == after['python_rng']
    assert torch.equal(before['torch_rng'], after['torch_rng'])
    assert before['tensors'].keys() == after['tensors'].keys()
    for key, expected in before['tensors'].items():
        actual = after['tensors'][key]
        assert actual[:4] == expected[:4] and torch.equal(actual[4], expected[4])
    for key, expected in before['gradients'].items():
        actual = after['gradients'][key]
        if expected is None:
            assert actual is None
        else:
            assert actual[:2] == expected[:2] and torch.equal(actual[2], expected[2])


@torch.no_grad()
def dense_oracle(original, hard, packed, cache, *, stored=True):
    # Codes from the exporting head; no mixed decoder or streaming KL reducer.
    codes, raw_scales = [], []
    for start in range(0, hard.out_features, hard.row_chunk_size):
        c, s = hard.hard_codes_rows(start, min(start + hard.row_chunk_size, hard.out_features))
        codes.append(c)
        raw_scales.append(s)
    code = torch.cat(codes).reshape(-1, hard.group_size).float()
    scale = packed.payload_scales.float() if stored else torch.cat(raw_scales).reshape(-1)
    weight = (code * scale[:, None]).reshape(original.weight.shape).bfloat16()
    student_bias = (None if original.bias is None else
                    packed.bias.bfloat16() if stored else original.bias)
    total = 0.
    for index in range(len(cache)):
        for batch in cache.iter_record(index, token_microbatch=2):
            x = batch['hidden']
            xq = x if hard.rotation_signs is None else rotate(x, hard.rotation_signs, hard.block_size)
            lp = F.linear(x, original.weight, original.bias).double().log_softmax(-1)
            lq = F.linear(xq, weight, student_bias).double().log_softmax(-1)
            total += float((lp.exp() * (lp - lq)).sum())
    return total / cache.positions


@pytest.mark.parametrize('rotated', [False, True])
@pytest.mark.parametrize('codec', ['2bit', 'bitcos', 'base3'])
def test_native_cache_all_codecs_dense_oracle_and_unchanged(tmp_path, monkeypatch, rotated, codec):
    original, hard, packed, cache, path = fixture(tmp_path, rotated=rotated, codec=codec, bias=True)
    expected = dense_oracle(original, hard, packed, cache)
    original.train(); packed.train()
    original.weight.grad = torch.ones_like(original.weight)
    before, disk = snapshot(original, packed), files(tmp_path)
    calls = []
    forward_rows = packed.forward_rows
    def rows(x, start, end):
        calls.append((len(x), start, end))
        return forward_rows(x, start, end)
    monkeypatch.setattr(packed, 'forward_rows', rows)
    def forbidden(*args, **kwargs):
        raise AssertionError('Full head/weight/encoded blob must not be materialized')
    monkeypatch.setattr(MixedLinear, 'forward', forbidden)
    monkeypatch.setattr(MixedLinear, 'weight', property(forbidden))
    monkeypatch.setattr(mixed_formats, 'candidate_weight', forbidden)
    monkeypatch.setattr(mixed_formats, 'encode_candidate', forbidden)
    result = evaluate_packed_head_from_cache(packed, original, cache,
        source_provenance=SOURCE, token_microbatch=2, vocabulary_chunk=71)
    assert result['kl'] == pytest.approx(expected, abs=2e-12)
    assert result['positions'] == cache.positions == 8
    assert result['microbatches'] == 5 and result['records'] == 2
    assert result['effective_vocabulary_chunk'] == 37
    assert len(calls) == 2 * result['microbatches'] * 7
    assert all(n <= 2 and end - start <= 37 for n, start, end in calls)
    assert result['candidate_serialization']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result['candidate_serialization']['bytes'] == path.stat().st_size
    assert result['role'] == 'calibration' and result['normalization'] == 'FP64'
    assert result['candidate_provenance']['pipeline_job_sha256'] == 'a' * 64
    assert result['pipeline_receipt_verified'] is False
    assert result['scale_storage_dtype'] == result['student_bias_storage_dtype'] == 'float16'
    assert result['teacher_bias_dtype'] == result['compute_dtype'] == 'bfloat16'
    assert packed._decoded_weight_cache is None
    unchanged(before, original, packed)
    assert files(tmp_path) == disk


@pytest.mark.parametrize('role', ['calibration', 'search_validation', 'evaluation'])
def test_only_explicit_supplied_role_scored_and_chunk_invariance(tmp_path, role):
    original, _, packed, cache, _ = fixture(tmp_path, role=role)
    first = evaluate_packed_head_from_cache(packed, original, cache.directory,
        source_provenance=SOURCE, token_microbatch=1, vocabulary_chunk=17)
    second = evaluate_packed_head_from_cache(packed, original, cache,
        source_provenance=SOURCE, token_microbatch=4, vocabulary_chunk=37)
    assert first['kl'] == pytest.approx(second['kl'], abs=2e-12)
    assert first['role'] == second['role'] == role
    assert first['positions'] == second['positions'] == cache.positions
    assert first['cache_manifest_sha256'] == second['cache_manifest_sha256']


def test_stored_scale_rounding_is_scored_not_preexport_hard(tmp_path):
    original, hard, _, cache, _ = fixture(tmp_path)
    # Force known FP32->FP16->BF16 double rounding at a BF16 midpoint.
    assert torch.tensor(1.004).bfloat16() != torch.tensor(1.004).half().bfloat16()
    real_rows = hard.hard_codes_rows
    def codes_and_scales(start, end):
        codes, scales = real_rows(start, end)
        return codes, torch.full_like(scales, 1.004)
    hard.hard_codes_rows = codes_and_scales
    candidate = export_head_candidate(hard, packing='base3', provenance=SOURCE)
    packed = MixedLinear(candidate, dtype=torch.bfloat16, row_chunk_size=37)
    result = evaluate_packed_head_from_cache(packed, original, cache, source_provenance=SOURCE,
        token_microbatch=2, vocabulary_chunk=37)
    assert result['kl'] == pytest.approx(dense_oracle(original, hard, packed, cache), abs=1e-10)
    assert abs(result['kl'] - dense_oracle(original, hard, packed, cache, stored=False)) > 1e-4
    assert 'not pre-export' in result['representation']


@pytest.mark.parametrize('case', [
    'source-weight', 'source-dtype', 'source-revision', 'nested-source', 'bias-hash',
    'bias-presence', 'bias-payload', 'scale-claim', 'export-method', 'export-source',
    'source-changed', 'source-bias-changed', 'not-frozen', 'compute-dtype', 'dimensions',
    'not-ternary', 'malformed-code', 'orphan-signs', 'warm-cache', 'stale-cache',
])
def test_reject_invalid_bindings_without_mutation(tmp_path, case):
    original, _, packed, cache, _ = fixture(tmp_path, bias=True)
    details = packed.metadata['provenance']
    with torch.no_grad():
        if case == 'source-weight': details['source_weight_sha256'] = 'b' * 64
        elif case == 'source-dtype': details['source_dtype'] = 'float32'
        elif case == 'source-revision': details['source_revision'] = 'other'
        elif case == 'nested-source': details['source_provenance'] = {**SOURCE, 'source_revision': 'other'}
        elif case == 'bias-hash': details['source_bias_sha256'] = 'b' * 64
        elif case == 'bias-presence': original.bias = None
        elif case == 'bias-payload': packed.payload_bias[0] = .5
        elif case == 'scale-claim': details['scale_storage'] = 'float32'
        elif case == 'export-method': details['export_method'] = 'arbitrary'
        elif case == 'export-source': packed.metadata['source'] = 'arbitrary'
        elif case == 'source-changed': original.weight[0, 0] += 1
        elif case == 'source-bias-changed': original.bias[1] += 1
        elif case == 'not-frozen': original.weight.requires_grad_(True)
        elif case == 'compute-dtype': packed._compute_reference = torch.empty(0, dtype=torch.float32)
        elif case == 'dimensions': packed.out_features += 1
        elif case == 'not-ternary': packed.metadata['encoding'] = 'affine'
        elif case == 'malformed-code': packed.payload_codes[0] = 255
        elif case == 'orphan-signs': packed.metadata['rotation'] = None
        elif case in {'warm-cache', 'stale-cache'}:
            packed.prepare_decoded_weight_cache()
            if case == 'stale-cache': packed.payload_scales[0] += .001
    before = snapshot(original, packed)
    cached = packed._decoded_weight_cache
    signature = packed._decoded_weight_cache_signature
    with pytest.raises((ValueError, TypeError, KeyError)):
        evaluate_packed_head_from_cache(packed, original, cache, source_provenance=SOURCE)
    unchanged(before, original, packed)
    assert packed._decoded_weight_cache is cached
    assert packed._decoded_weight_cache_signature == signature


@pytest.mark.parametrize('kwargs', [
    {'token_microbatch': 0}, {'vocabulary_chunk': True},
    {'source_provenance': {**SOURCE, 'source_revision': 'other'}},
])
def test_reject_bad_call(tmp_path, kwargs):
    original, _, packed, cache, _ = fixture(tmp_path)
    with pytest.raises(ValueError):
        evaluate_packed_head_from_cache(packed, original, cache,
            **{'source_provenance': SOURCE, **kwargs})


def test_fresh_process_disk_reload_matches_parent_dense_oracle_and_files(tmp_path):
    original, hard, packed, cache, path = fixture(tmp_path, rotated=True, bias=True, role='search_validation')
    expected = dense_oracle(original, hard, packed, cache)
    teacher_path = tmp_path / 'original-head.safetensors'
    save_file({'weight': original.weight.detach(), 'bias': original.bias.detach()}, str(teacher_path))
    before = files(tmp_path)
    script = r'''
import json, sys, torch
from torch import nn
from safetensors.torch import load_file
from openbonsai.mixed_formats import MixedLinear, load_candidate
from openbonsai.head_evaluation import evaluate_packed_head_from_cache
torch.set_num_threads(2)
state = load_file(sys.argv[1])
head = nn.Linear(state['weight'].shape[1], state['weight'].shape[0], bias=True, dtype=torch.bfloat16)
head.load_state_dict(state); head.requires_grad_(False)
packed = MixedLinear(load_candidate(sys.argv[2]), dtype=torch.bfloat16, row_chunk_size=37)
report = evaluate_packed_head_from_cache(packed, head, sys.argv[3], source_provenance=json.loads(sys.argv[4]),
    token_microbatch=2, vocabulary_chunk=37)
print(json.dumps(report, sort_keys=True))
'''
    result = subprocess.run([sys.executable, '-c', script, str(teacher_path), str(path),
        str(cache.directory), json.dumps(SOURCE)], check=True, text=True, capture_output=True,
        timeout=60, env={**os.environ, 'OMP_NUM_THREADS': '2'})
    report = json.loads(result.stdout)
    assert report['kl'] == pytest.approx(expected, abs=2e-12)
    assert report['candidate_encoding'] == 'ternary-base3' and report['rotation'] is not None
    assert report['role'] == 'search_validation' and report['positions'] == 8
    assert report['candidate_serialization']['sha256'] == before['head.obc']
    assert report['cache_manifest_sha256'] == before['cache/manifest.json']
    assert files(tmp_path) == before


@pytest.mark.parametrize('mode', ['none', 'hadamard'])
def test_actual_staged_export_cache_and_source_are_compatible(tmp_path, mode):
    """Consume actual code-bound stages, not synthetic provenance lookalikes."""
    from openbonsai import pipeline_endpoints as stages
    from openbonsai import pipeline_sources as sources
    from test_pipeline_endpoints import make_context, recipes

    recipe, _ = recipes()
    recipe['quantization']['rotation_mode'] = mode
    context = make_context(tmp_path, head=recipe)
    cache_receipt = stages.cache_head_stage(context, 'calibration', tmp_path / 'calibration')
    stage_receipt = stages.fit_head_stage(context, tmp_path / 'calibration', tmp_path / 'head')
    candidates, loaded_receipt = sources.load_candidate_stage(context, tmp_path / 'head', expected_stage='head')
    candidate = candidates['lm_head']
    cache, loaded_cache_receipt = stages.load_head_cache_stage(context, tmp_path / 'calibration', role='calibration')
    original_model = sources.load_original(context, device='cpu')
    original = original_model.lm_head
    packed = MixedLinear(candidate, dtype=torch.bfloat16,
                         row_chunk_size=context.job['head']['quantization']['row_chunk_size'])
    # The staged exporter really uses flat original attribution plus job fields.
    provenance = candidate.metadata['provenance']
    assert 'source_provenance' not in provenance
    assert {key: provenance[key] for key in context.source_provenance} == context.source_provenance
    assert provenance['pipeline_job_sha256'] == context.job_sha256
    assert ('rotation' in candidate.metadata) == (mode == 'hadamard')
    assert ('rotation_signs' in candidate.tensors) == (mode == 'hadamard')
    if mode == 'hadamard':
        assert torch.equal(candidate.tensors['rotation_signs'], fixed_signs(128, seed=41, name='lm_head'))
    assert loaded_receipt == stage_receipt and loaded_cache_receipt == cache_receipt
    disk, state = files(tmp_path), snapshot(original, packed)
    result = evaluate_packed_head_from_cache(packed, original, cache,
        source_provenance=context.source_provenance, token_microbatch=2, vocabulary_chunk=31)
    entry = loaded_receipt['candidates']['lm_head']
    candidate_file = tmp_path / 'head' / entry['file']
    assert result['candidate_serialization']['sha256'] == entry['sha256'] == hashlib.sha256(candidate_file.read_bytes()).hexdigest()
    assert result['candidate_serialization']['bytes'] == entry['bytes'] == candidate_file.stat().st_size
    assert result['source_weight_sha256'] == provenance['source_weight_sha256'] == cache.manifest['source_weight_sha256']
    assert result['source_provenance'] == context.source_provenance
    assert result['candidate_provenance'] == provenance
    assert result['rotation'] == candidate.metadata.get('rotation')
    assert result['role'] == 'calibration' and result['positions'] == 7
    # Independently normalize complete logits from the loaded module in this
    # tiny fixture; production evaluation remains bounded to vocabulary rows.
    total = 0.
    with torch.no_grad():
        for index in range(len(cache)):
            for batch in cache.iter_record(index, token_microbatch=2):
                lp = F.linear(batch['hidden'], original.weight, original.bias).double().log_softmax(-1)
                lq = packed(batch['hidden']).double().log_softmax(-1)
                total += float((lp.exp() * (lp - lq)).sum())
    assert result['kl'] == pytest.approx(total / cache.positions, abs=2e-12)
    unchanged(state, original, packed)
    assert files(tmp_path) == disk
    # Both completed, authenticated handoffs remain valid after scoring.
    assert sources.load_candidate_stage(context, tmp_path / 'head', expected_stage='head')[1] == stage_receipt
    assert stages.load_head_cache_stage(context, tmp_path / 'calibration', role='calibration')[1] == cache_receipt
