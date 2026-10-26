"""Opt-in unrotated head stages; CPU fixtures are not pretrained quality evidence."""
import copy
import hashlib
import json

import pytest
import torch

from openbonsai import pipeline_endpoints as stages
from openbonsai import pipeline_sources as sources
from openbonsai.candidates import source_weight_sha256
from openbonsai.endpoint_quantizer import BoundedTernaryHead
from openbonsai.mixed_export import load_mixed
from openbonsai.mixed_formats import (MixedLinear, encode_candidate, save_candidate,
                                     ternary_candidate)
from openbonsai.packing import pack_grouped_weight
from openbonsai.pipeline_decoder import assemble_stage
from openbonsai.rotations import fixed_signs
from openbonsai.text_export import extract_text_model
from test_pipeline_endpoints import make_context, recipes


def head_recipe(mode):
    recipe = recipes()[0]
    if mode is not None:
        recipe['quantization']['rotation_mode'] = mode
    return recipe


def sample_candidate(context, *, rotated):
    # Storage-only fixture; real learned stages are covered separately below.
    weight = torch.zeros(2, 128, dtype=torch.bfloat16)
    packed = pack_grouped_weight(torch.zeros_like(weight, dtype=torch.int8), torch.ones(2),
                                 shape=weight.shape, group_size=128)
    return ternary_candidate(packed, packing='base3',
        provenance=sources.candidate_provenance(context, 'head', 'lm_head',
            source_weight_sha256=source_weight_sha256(weight)),
        rotation_signs=fixed_signs(128, seed=41, name='lm_head') if rotated else None,
        rotation_block_size=64 if rotated else None)


def test_omitted_rotation_mode_keeps_recipe_and_default_math_exact(tmp_path):
    recipe = head_recipe(None)
    before = copy.deepcopy(recipe)
    digest = sources.json_sha256(recipe)
    context = make_context(tmp_path, head=recipe)
    parsed, _ = stages._recipe(context, 'head')
    assert context.job['head'] == parsed == recipe == before
    assert 'rotation_mode' not in context.job['head']['quantization']
    assert sources.json_sha256(context.job['head']) == digest
    job_bytes = context.job_path.read_bytes()
    assert sources.load_context(context.job_path).job == context.job
    assert context.job_path.read_bytes() == job_bytes

    source = sources.load_original(context).lm_head
    rng = torch.random.get_rng_state().clone()
    default = stages._construct(source, 'head', 'lm_head', parsed)
    explicit = stages._construct(source, 'head', 'lm_head', head_recipe('hadamard'))
    kwargs = copy.deepcopy(parsed['quantization'])
    seed, initial_seed = kwargs.pop('rotation_seed'), kwargs.pop('initialization_seed')
    with torch.random.fork_rng():
        torch.manual_seed(initial_seed)
        oracle = BoundedTernaryHead(source, signs=fixed_signs(128, seed=seed, name='lm_head'), **kwargs)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert default.state_dict().keys() == explicit.state_dict().keys() == oracle.state_dict().keys()
    for name, value in default.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[name])
        assert torch.equal(value, oracle.state_dict()[name])
    hidden = torch.arange(128, dtype=torch.float32).reshape(1, 128).bfloat16() / 128
    for progress in (.5, 1.):
        for head in (default, explicit, oracle):
            head.begin_step(progress)
        expected = oracle.forward_rows(hidden, 0, 31)
        assert torch.equal(default.forward_rows(hidden, 0, 31), expected)
        assert torch.equal(explicit.forward_rows(hidden, 0, 31), expected)


@pytest.mark.parametrize('multimodal', [False, True])
def test_real_unrotated_head_fit_h_only_assembly_and_native_reload(tmp_path, multimodal):
    context = make_context(tmp_path, multimodal=multimodal, head=head_recipe('none'))
    source_identity = copy.deepcopy(context.job['source']['model_directory'])
    source_job = context.job_path.read_bytes()
    stages.cache_head_stage(context, 'calibration', tmp_path / 'cache')
    stages.cache_head_stage(context, 'search_validation', tmp_path / 'search')
    receipt = stages.fit_head_stage(context, tmp_path / 'cache', tmp_path / 'head',
                                    search_cache=tmp_path / 'search')
    assert len(receipt['report']['steps']) == 2
    assert receipt['report']['head_configuration']['rotation'] is None
    assert receipt['report']['search_validation']['role'] == 'search_validation'
    assert receipt['candidates']['lm_head']['rotation'] is None
    assert receipt['candidates']['lm_head']['rotation_signs_sha256'] is None
    candidates, _ = sources.load_candidate_stage(context, tmp_path / 'head', expected_stage='head')
    candidate = candidates['lm_head']
    assert 'rotation' not in candidate.metadata and 'rotation_signs' not in candidate.tensors
    assert candidate.metadata['provenance']['pipeline_stage_config_sha256'] == sources.json_sha256(context.job['head'])
    payload = encode_candidate(candidate)
    control = dict(quantized_components=['head'], retained_original_components=['decoder', 'embedding'],
                   output_scope='language-only', evaluation_label='H-only', budget_bytes=10**8)
    report = assemble_stage(context, [tmp_path / 'head'], tmp_path / 'assembled', assembly_control=control)
    assert report['module_count'] == 1 and report['delivery_target_complete'] is False
    assert report['candidate_payloads_unchanged'] is True

    # Independent composition from the unchanged source plus the exact candidate.
    original = sources.load_original(context)
    if multimodal:
        original = extract_text_model(original,
            source_id=context.source_provenance['source_checkpoint'],
            source_revision=context.source_provenance['source_revision']).model
    retained = {name: value.clone() for name, value in original.state_dict().items()
                if not name.startswith('lm_head.')}
    original.lm_head = MixedLinear(candidate, row_chunk_size=31)
    loaded = load_mixed(tmp_path / 'assembled' / 'artifact')
    assert isinstance(loaded.lm_head, MixedLinear)
    assert encode_candidate(loaded.lm_head._candidate()) == payload
    for name, expected in retained.items():
        actual = loaded.state_dict()[name]
        assert actual.dtype == expected.dtype and torch.equal(actual, expected)
    with torch.no_grad():
        for record in context.records_by_role['search_validation']:
            assert torch.equal(loaded(**record).logits, original(**record).logits)
    assert sources.local_directory_identity(source_identity['path']) == source_identity
    assert context.job_path.read_bytes() == source_job


@pytest.mark.parametrize('mode', ['invalid', '', None, True, 0, {}, []])
def test_invalid_head_mode_rejected_without_rewriting_recipe(tmp_path, mode):
    recipe = head_recipe(None)
    recipe['quantization']['rotation_mode'] = mode
    before = copy.deepcopy(recipe)
    with pytest.raises(ValueError, match='rotation_mode'):
        make_context(tmp_path, head=recipe)
    assert recipe == before
    assert not (tmp_path / 'job').exists()


@pytest.mark.parametrize('stage', ['decoder', 'embedding'])
@pytest.mark.parametrize('mode', ['none', 'hadamard'])
def test_rotation_mode_field_is_head_only(tmp_path, stage, mode):
    context = make_context(tmp_path)
    settings = {name: copy.deepcopy(context.job[name]) for name in ('decoder', 'head', 'embedding', 'assembly')}
    settings[stage]['quantization']['rotation_mode'] = mode
    with pytest.raises(ValueError):
        sources._validate_recipes(**settings)
    if stage == 'embedding':
        context.job[stage] = settings[stage]
        with pytest.raises(ValueError):
            stages._recipe(context, stage)


@pytest.mark.parametrize('mode,rotated', [('none', True), ('hadamard', False), (None, False)])
def test_candidate_publish_rejects_rotation_different_from_sealed_mode(tmp_path, mode, rotated):
    context = make_context(tmp_path, head=head_recipe(mode))
    with pytest.raises(ValueError, match='rotation_mode|authoritative signs'):
        sources.publish_candidates(context, 'head', {'lm_head': sample_candidate(context, rotated=rotated)},
                                   tmp_path / 'stage', report={})
    assert not (tmp_path / 'stage' / 'stage.json').exists()


@pytest.mark.parametrize('mode', ['none', 'hadamard'])
def test_lazy_load_rejects_resealed_payload_with_wrong_rotation(tmp_path, mode):
    context = make_context(tmp_path, head=head_recipe(mode))
    correct = sample_candidate(context, rotated=mode == 'hadamard')
    receipt = sources.publish_candidates(context, 'head', {'lm_head': correct}, tmp_path / 'stage', report={})
    wrong = sample_candidate(context, rotated=mode != 'hadamard')
    entry = receipt['candidates']['lm_head']
    path = tmp_path / 'stage' / entry['file']
    path.unlink()
    save_candidate(wrong, path)
    entry.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size,
        rotation=wrong.metadata.get('rotation'), rotation_signs_sha256=(
            hashlib.sha256(wrong.tensors['rotation_signs'].numpy().tobytes()).hexdigest()
            if 'rotation_signs' in wrong.tensors else None))
    receipt['sha256'] = sources.json_sha256({key: value for key, value in receipt.items() if key != 'sha256'})
    (tmp_path / 'stage' / 'stage.json').write_text(json.dumps(receipt))
    candidates, _ = sources.load_candidate_stage(context, tmp_path / 'stage')
    with pytest.raises(ValueError, match='rotation_mode|authoritative signs'):
        candidates['lm_head']


def test_unrotated_payload_cannot_publish_orphan_signs(tmp_path):
    context = make_context(tmp_path, head=head_recipe('none'))
    candidate = sample_candidate(context, rotated=False)
    candidate.tensors['rotation_signs'] = torch.ones(128, dtype=torch.int8)
    with pytest.raises(ValueError, match='unexpected .*tensors'):
        sources.publish_candidates(context, 'head', {'lm_head': candidate}, tmp_path / 'stage', report={})
    assert not (tmp_path / 'stage' / 'stage.json').exists()
