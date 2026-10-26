"""Frozen native-runner contract tests; random CPU fixtures prove no 9B quality."""
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from openbonsai import evaluation
from openbonsai.adapter import tiny_model
from openbonsai.candidates import source_weight_sha256
from openbonsai.eval_backends import TransformersBackend, _byte_decoder, local_directory_identity
from openbonsai.mixed_export import load_mixed, prepare_mixed_export, save_mixed
from openbonsai.mixed_formats import MixedEmbedding, MixedLinear, bf16_candidate, ternary_candidate, with_rotation
from openbonsai.packing import pack_grouped_weight

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'native_manifest_runner', ROOT / 'tools/evaluate_transformers_manifest.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def frozen_text(name, ids):
    pieces = [bytes([64 + token]) for token in ids]
    raw = b''.join(pieces)
    return evaluation.freeze_text(name, raw_text=raw, rendered_text=raw,
                                  input_ids=ids, token_pieces=pieces)


def synthetic_fixture():
    texts, nll, choices, generations = [], [], [], []
    for index, ids in enumerate((list(range(1, 10)), list(range(2, 9)))):
        positions = list(range(1, len(ids)))
        name = f'nll-{index}'
        texts.append(frozen_text(name, ids))
        nll.append(dict(id=name, text_id=name, input_ids=ids, score_positions=positions,
                        token_start=0, token_end=len(ids)))
    for index, (prefix, labels) in enumerate((([11, 12], [21, 22]),
                                             ([13, 14, 15, 16], [23, 24, 25]))):
        options = []
        for option_index, label in enumerate(labels):
            name = f'choice-{index}-{option_index}'
            ids = prefix + [label]
            texts.append(frozen_text(name, ids))
            options.append(dict(id=str(option_index), text_id=name, input_ids=ids,
                                prefix_length=len(prefix), score_positions=[len(prefix)]))
        choices.append(dict(id=f'choice-{index}', options=options, answer_id='0'))
    for index, (ids, cap) in enumerate((([31, 32], 3), ([33, 34, 35], 4))):
        name = f'generation-{index}'
        texts.append(frozen_text(name, ids))
        generations.append(dict(id=name, text_id=name, input_ids=ids,
                                max_new_tokens=cap, eos_ids=[99]))
    manifest = evaluation.freeze_manifest(
        source_provenance={'kind': 'synthetic-unit-test', 'sha256': 'a' * 64},
        tokenizer_identity={'kind': 'fake-byte-tokenizer', 'sha256': 'b' * 64},
        texts=texts, nll=nll, choices=choices, generations=generations,
        role='search_validation')
    profile = dict(format='openbonsai.transformers-profile.v1',
                   manifest_sha256=manifest['frozen_sha256'], context_length=16,
                   vocab_size=128, threads=2, logits_chunk_size=2,
                   minimum_free_cuda_bytes=0,
                   resource_probes={'nll': {'record_index': 0},
                                    'choice': {'question_index': 1, 'option_index': 0},
                                    'generation': {'text_index': 0, 'token_start': 0,
                                                   'token_end': 8, 'target_position': 7}})
    return manifest, profile


def reseal(manifest, profile):
    manifest['frozen_sha256'] = evaluation.canonical_sha256(
        {key: value for key, value in manifest.items() if key != 'frozen_sha256'})
    profile['manifest_sha256'] = manifest['frozen_sha256']


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class FakeBackend:
    def __init__(self, manifest, clock, *, bad_ids=False, bad_pieces=False,
                 score_seconds=.01, bad_score=None):
        self.clock, self.bad_ids, self.bad_pieces = clock, bad_ids, bad_pieces
        self.score_seconds, self.bad_score = score_seconds, bad_score
        self.scores, self.generations, self.tokenizations = [], [], []
        self.lookup = {(bytes.fromhex(row['rendered_utf8_hex']).decode(), row['parse_special']): row['input_ids']
                       for row in manifest['texts']}
        self.pieces = {token: bytes.fromhex(piece) for row in manifest['texts']
                       for token, piece in zip(row['input_ids'], row['token_pieces_hex'])}
        self.pieces[99] = b'<EOS>'

    def metadata(self):
        self.clock.now += .01
        return dict(protocol=evaluation.EVALUATION_PROTOCOL, reduction=evaluation.REDUCTION,
                    cache_mode='full-prefix-reference', reset_policy='per-record',
                    context_length=16, vocab_size=128, source_identity={'sha256': 'c' * 64},
                    tokenizer_identity={'sha256': 'd' * 64})

    def tokenize(self, text, parse_special=False):
        self.clock.now += .01
        self.tokenizations.append((text, parse_special))
        ids = list(self.lookup[text, parse_special])
        if self.bad_ids:
            ids[0] += 1
        return ids

    def token_pieces(self, ids):
        self.clock.now += .01
        return [b'wrong' if self.bad_pieces else self.pieces[token] for token in ids]

    def score_tokens(self, ids, positions):
        self.clock.now += self.score_seconds
        self.scores.append((list(ids), list(positions)))
        result = dict(score_positions=list(positions), target_logprobs=[-1.] * len(positions),
                      count=len(positions), nll_sum=float(len(positions)))
        if self.bad_score == 'positions':
            result['score_positions'] = [1]
        elif self.bad_score == 'nonfinite':
            result['target_logprobs'][0] = float('nan')
        return result

    def generate(self, ids, maximum, eos):
        self.clock.now += .01
        self.generations.append((list(ids), maximum, list(eos)))
        return dict(generated_ids=[99], stop_reason='eos')


def test_native_shape_counts_every_input_even_after_last_selected_target():
    assert runner.score_shape({'input_ids': [1] * 17, 'score_positions': [1, 3]}, 1) == {
        'input_tokens': 17, 'output_rows': 2, 'maximum_head_rows': 1}
    manifest, profile = synthetic_fixture()
    plan = runner.build_plan(manifest, profile)
    work = plan['workload']
    assert work['nll_records'] == 2 and work['nll_targets'] == 14
    assert work['choice_questions'] == 2 and work['choice_option_calls'] == 5
    assert work['generation_records'] == 2 and work['generation_tokens_maximum'] == 7
    assert work['tokenization_texts'] == 9


def test_complete_fake_suite_preserves_every_frozen_request_and_cap(tmp_path):
    manifest, profile = synthetic_fixture()
    clock = Clock()
    backend = FakeBackend(manifest, clock)
    result = runner.evaluate_plan(manifest, profile, backend, tmp_path, 10000, clock=clock)
    assert result['status'] == 'complete'
    expected = [(row['input_ids'], row['score_positions']) for row in manifest['nll']]
    expected += [(row['input_ids'], row['score_positions'])
                 for question in manifest['choices'] for row in question['options']]
    assert backend.scores[3:] == expected and len(backend.scores) == 10
    assert backend.generations == [(row['input_ids'], row['max_new_tokens'], row['eos_ids'])
                                    for row in manifest['generations']]
    assert len(backend.tokenizations) == 18
    admission = json.loads((tmp_path / 'admission.json').read_text())
    assert admission['workload']['generation_tokens_maximum'] == 7
    assert admission['early_eos_assumed'] is False
    report = json.loads((tmp_path / 'evaluation.json').read_text())
    assert report['aggregate']['scored_tokens'] == 14
    assert report['legacy_fp32_cross_entropy_compatible'] is False
    assert [row['id'] for row in report['nll']] == [row['id'] for row in manifest['nll']]


@pytest.mark.parametrize('kwargs', [{'bad_ids': True}, {'bad_pieces': True}])
def test_all_native_tokenization_checks_precede_scoring(tmp_path, kwargs):
    manifest, profile = synthetic_fixture()
    clock = Clock()
    backend = FakeBackend(manifest, clock, **kwargs)
    with pytest.raises(ValueError, match='mismatch'):
        runner.evaluate_plan(manifest, profile, backend, tmp_path, 10000, clock=clock)
    assert not backend.scores and not backend.generations


@pytest.mark.parametrize('bad', ['positions', 'nonfinite'])
def test_invalid_resource_score_cannot_admit(tmp_path, bad):
    manifest, profile = synthetic_fixture()
    clock = Clock()
    backend = FakeBackend(manifest, clock, bad_score=bad)
    with pytest.raises(ValueError):
        runner.evaluate_plan(manifest, profile, backend, tmp_path, 10000, clock=clock)
    assert len(backend.scores) == 1 and not backend.generations
    assert not (tmp_path / 'admission.json').exists()


def test_budget_failure_keeps_full_caps_without_starting_suite(tmp_path):
    manifest, profile = synthetic_fixture()
    clock = Clock()
    backend = FakeBackend(manifest, clock, score_seconds=1.)
    with pytest.raises(RuntimeError, match='does not fit'):
        runner.evaluate_plan(manifest, profile, backend, tmp_path, 310, clock=clock)
    assert len(backend.scores) == 3 and not backend.generations
    assert not (tmp_path / 'evaluation.json').exists()
    admission = json.loads((tmp_path / 'admission.json').read_text())
    assert admission['workload']['generation_tokens_maximum'] == 7


def test_probe_must_cover_input_length_not_just_last_scored_position():
    manifest, profile = synthetic_fixture()
    # The long first record only scores early targets. A complementary shorter
    # window covers its tail, preserving the complete-document NLL contract.
    manifest['nll'][0]['score_positions'] = [1, 2]
    source = manifest['nll'][0]
    manifest['nll'].append({**source, 'id': 'nll-complement', 'token_start': 2,
                            'input_ids': source['input_ids'][2:],
                            'score_positions': list(range(1, 7))})
    profile['resource_probes']['nll']['record_index'] = 1
    reseal(manifest, profile)
    evaluation.validate_manifest(manifest)
    with pytest.raises(ValueError, match='NLL probe'):
        runner.build_plan(manifest, profile)


def test_probe_must_cover_total_and_chunked_head_rows():
    manifest, profile = synthetic_fixture()
    source = manifest['nll'][0]
    source['score_positions'] = [1, 3, 5]
    manifest['nll'].append({**source, 'id': 'nll-complement',
                            'score_positions': [2, 4, 6, 7, 8]})
    reseal(manifest, profile)
    evaluation.validate_manifest(manifest)
    with pytest.raises(ValueError, match='NLL probe'):
        runner.build_plan(manifest, profile)


@pytest.mark.parametrize('change', [
    lambda p: p.update(manifest_sha256='0' * 64),
    lambda p: p.update(logits_chunk_size=0),
    lambda p: p.update(logits_chunk_size=True),
    lambda p: p.update(unexpected=True),
    lambda p: p['resource_probes']['generation'].update(token_end=5, target_position=4),
])
def test_invalid_profile_rejected_before_model_work(change):
    manifest, profile = synthetic_fixture()
    change(profile)
    with pytest.raises(ValueError):
        runner.build_plan(manifest, profile)


def test_native_forecast_uses_derived_counts_and_full_caps():
    manifest, profile = synthetic_fixture()
    plan = runner.build_plan(manifest, profile)
    value = runner.forecast(plan, {'nll': 2., 'choice': 3.,
                                  'generation_last_logit': 4., 'tokenization': 5.}, 10000)
    assert value['projected_remaining_seconds'] == 1.25 * (2 * 2 + 5 * 3 + 7 * 4 + 5) + 300


def tokenizer_fixture():
    vocabulary = {char: i for i, char in enumerate(_byte_decoder())}
    tokens = Tokenizer(models.BPE(vocab=vocabulary, merges=[]))
    tokens.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokens.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=tokens,
                                   additional_special_tokens=['<|test|>'])


@pytest.fixture(scope='module')
def native_artifacts(tmp_path_factory):
    torch.set_num_threads(2)
    root = tmp_path_factory.mktemp('native-manifest-artifacts')
    tokenizer = tokenizer_fixture()
    token_path = root / 'tokenizer'
    tokenizer.save_pretrained(token_path)
    model = tiny_model(seed=73, multimodal=True)
    model.resize_token_embeddings(260, mean_resizing=False)
    model = model.bfloat16().eval().requires_grad_(False)
    with torch.no_grad():
        # Keep padded rows in the denominator while avoiding unrenderable argmax.
        model.lm_head.weight[len(tokenizer):].zero_()
    original_path = root / 'original'
    model.save_pretrained(original_path)
    names = ['model.language_model.embed_tokens', 'lm_head',
             'model.language_model.layers.0.mlp.down_proj']
    assignment = {}
    for name in names:
        module = model.get_submodule(name)
        if name.endswith('down_proj'):
            codes = ((torch.arange(module.weight.numel()).reshape(module.weight.shape) % 3) - 1).to(torch.int8)
            scales = torch.full((codes.numel() // 128,), .001, dtype=torch.float32)
            packed = pack_grouped_weight(codes, scales, shape=codes.shape, group_size=128)
            candidate = ternary_candidate(packed, packing='bitcos')
            # Deterministic latent-code fixture with authoritative stored signs.
            signs = torch.where(torch.arange(codes.shape[1]) % 3 == 0, -1, 1).to(torch.int8)
            candidate = with_rotation(candidate, signs, 128)
        else:
            candidate = bf16_candidate(module.weight,
                module_kind='embedding' if name.endswith('embed_tokens') else 'linear')
        candidate.metadata['provenance']['source_weight_sha256'] = source_weight_sha256(module.weight)
        assignment[name] = candidate
    mixed_path = root / 'mixed'
    with prepare_mixed_export(model, names, metadata={'source_checkpoint': 'random-native-test-fixture'},
                              tokenizer_dir=token_path, row_chunk_size=37,
                              cache_decoded_weights=False) as plan:
        save_mixed(plan, assignment, mixed_path, budget_bytes=10**8)
    expected = copy.deepcopy(model)
    for name, candidate in assignment.items():
        cls = MixedEmbedding if candidate.module_kind == 'embedding' else MixedLinear
        expected.set_submodule(name, cls(candidate, row_chunk_size=37))
    return dict(original=original_path, mixed=mixed_path, tokenizer=token_path,
                expected_mixed=expected, original_model=model)


def load_native_backend(paths, kind):
    return TransformersBackend.from_local(
        paths[kind], tokenizer_path=paths['tokenizer'], context_length=32,
        model_kind=kind, device='cpu', dtype='bfloat16', logits_chunk_size=2,
        expected_model_sha256=local_directory_identity(paths[kind])['sha256'],
        expected_tokenizer_sha256=local_directory_identity(paths['tokenizer'])['sha256'])


def direct_logprobs(model, ids, positions):
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                       use_cache=False).logits[0].float().double()
    values = []
    for j in positions:
        centered = logits[j - 1] - logits[j - 1].max()
        # Independent full-model oracle, without backend hidden/head helpers.
        values.append(float(centered[ids[j]] - torch.log(torch.exp(centered).sum())))
    return values


@pytest.mark.parametrize('kind', ['original', 'mixed'])
def test_real_local_multimodal_reload_sparse_full_vocab_and_state_reset(native_artifacts, kind):
    backend = load_native_backend(native_artifacts, kind)
    ids, positions = [10, 20, 30, 40, 259, 50, 60, 70, 80], [1, 4, 6]
    expected = direct_logprobs(backend.model, ids, positions)
    observed_inputs, head_rows = [], []
    hook = backend.model.model.register_forward_pre_hook(
        lambda _, args, kwargs: observed_inputs.append(kwargs['input_ids'].shape[-1]), with_kwargs=True)
    head_hook = backend.model.lm_head.register_forward_pre_hook(lambda _, args: head_rows.append(args[0].shape[1]))
    try:
        result = backend.score_tokens(ids, positions)
    finally:
        hook.remove()
        head_hook.remove()
    assert observed_inputs == [len(ids)] and head_rows == [2, 1]
    assert result['target_logprobs'] == pytest.approx(expected, abs=1e-7)
    assert result['nll_sum'] == pytest.approx(-math.fsum(expected), abs=1e-7)
    assert backend.vocab_size == 260 > len(backend.tokenizer)
    backend.score_tokens([2, 3, 4], [1, 2])
    assert backend.score_tokens(ids, positions) == result
    assert backend.metadata()['source_identity']['model_kind'] == kind
    if kind == 'mixed':
        assert isinstance(backend.model.get_input_embeddings(), MixedEmbedding)
        assert isinstance(backend.model.lm_head, MixedLinear)
        assert direct_logprobs(native_artifacts['expected_mixed'], ids, positions) == expected
        for name, tensor in native_artifacts['original_model'].model.visual.state_dict().items():
            assert torch.equal(tensor, backend.model.model.visual.state_dict()[name])


def test_mixed_disk_cold_and_warm_decoded_cache_have_exact_probabilities(native_artifacts):
    identity_before = local_directory_identity(native_artifacts['mixed'])
    cold = load_native_backend(native_artifacts, 'mixed')
    cached_model = load_mixed(native_artifacts['mixed'], cache_decoded_weights=True)
    warm = TransformersBackend(cached_model, cold.tokenizer, context_length=32,
        source_identity=identity_before, tokenizer_identity=cold.metadata()['tokenizer_identity'], logits_chunk_size=2)
    modules = [module for module in cached_model.modules() if isinstance(module, MixedLinear)]
    assert modules and sum(module.decoded_weight_cache_bytes for module in modules) > 0
    ids, positions = [10, 20, 30, 40, 50, 60, 70], [1, 3, 6]
    assert cold.score_tokens(ids, positions) == warm.score_tokens(ids, positions)
    assert cold.generate([10, 20], 3, [2]) == warm.generate([10, 20], 3, [2])
    assert local_directory_identity(native_artifacts['mixed']) == identity_before


def test_actual_native_generation_exact_eos_and_cap_after_disk_load(native_artifacts):
    backend = load_native_backend(native_artifacts, 'original')
    before = local_directory_identity(native_artifacts['original'])
    with torch.no_grad():
        backend.model.lm_head.weight.zero_()
    assert backend.generate([10, 20], 4, [0, 2]) == {'generated_ids': [0], 'stop_reason': 'eos'}
    assert backend.generate([10, 20], 4, [1, 2]) == {'generated_ids': [0] * 4, 'stop_reason': 'length'}
    assert local_directory_identity(native_artifacts['original']) == before


def native_manifest(backend):
    texts, nll, options, generations = [], [], [], []

    def add(name, text):
        ids = backend.tokenize(text)
        texts.append(evaluation.freeze_text(name, raw_text=text, rendered_text=text,
            input_ids=ids, token_pieces=backend.token_pieces(ids)))
        return ids

    ids = add('native-nll', 'abcdefghijklm')
    positions = [1, 3, 7, 12]
    nll.append(dict(id='native-nll', text_id='native-nll', input_ids=ids,
                    score_positions=positions, token_start=0, token_end=len(ids)))
    remaining_positions = [j for j in range(1, len(ids)) if j not in positions]
    nll.append(dict(id='native-nll-complement', text_id='native-nll', input_ids=ids,
                    score_positions=remaining_positions, token_start=0, token_end=len(ids)))
    prefix = backend.tokenize('Q:')
    for label in [' A', ' B']:
        name = 'native-choice-' + label[-1]
        full = add(name, 'Q:' + label)
        options.append(dict(id=label[-1], text_id=name, input_ids=full,
                            prefix_length=len(prefix), score_positions=list(range(len(prefix), len(full)))))
    for index, (text, cap) in enumerate([('ab', 3), ('xyz', 2)]):
        name = f'native-generation-{index}'
        generations.append(dict(id=name, text_id=name, input_ids=add(name, text), max_new_tokens=cap, eos_ids=[2]))
    manifest = evaluation.freeze_manifest(source_provenance={'kind': 'random-fixture', 'sha256': 'e' * 64},
        tokenizer_identity=backend.metadata()['tokenizer_identity'], texts=texts, nll=nll,
        choices=[dict(id='native-choice', options=options, answer_id='A')], generations=generations,
        role='search_validation')
    _, profile = synthetic_fixture()
    profile.update(manifest_sha256=manifest['frozen_sha256'], context_length=32, vocab_size=260)
    profile['resource_probes']['nll'] = {'record_index': 1}
    profile['resource_probes']['choice'] = {'question_index': 0, 'option_index': 0}
    return manifest, profile


def test_real_mixed_reload_through_runner_matches_independent_oracle(native_artifacts, tmp_path):
    backend = load_native_backend(native_artifacts, 'mixed')
    manifest, profile = native_manifest(backend)
    ids, positions = manifest['nll'][0]['input_ids'], manifest['nll'][0]['score_positions']
    options = manifest['choices'][0]['options']
    expected = direct_logprobs(backend.model, ids, positions)
    expected_options = [direct_logprobs(backend.model, o['input_ids'], o['score_positions']) for o in options]
    result = runner.evaluate_plan(manifest, profile, backend, tmp_path, time.monotonic() + 600)
    assert result['status'] == 'complete'
    report = json.loads((tmp_path / 'evaluation.json').read_text())
    assert report['nll'][0]['target_logprobs'] == pytest.approx(expected, abs=1e-7)
    for row, wanted in zip(report['choices'][0]['options'], expected_options):
        assert row['target_logprobs'] == pytest.approx(wanted, abs=1e-7)
    assert report['aggregate']['scored_tokens'] == 12
    assert report['aggregate']['generations'] == 2
    assert report['legacy_fp32_cross_entropy_compatible'] is False


@pytest.mark.parametrize('kind,policy', [('original', 'cold'), ('mixed', 'cold'), ('mixed', 'warm')])
def test_real_runner_factory_and_explicit_cache_policy(native_artifacts, tmp_path, kind, policy):
    manifest, profile = synthetic_fixture()
    profile.update(context_length=32, vocab_size=260)
    plan = runner.build_plan(manifest, profile)
    identities = {key: local_directory_identity(native_artifacts[key]) for key in [kind, 'tokenizer']}
    args = SimpleNamespace(kind=kind, device='cpu', cache_policy=policy,
        model=native_artifacts[kind], tokenizer=native_artifacts['tokenizer'],
        expected_model_sha256=identities[kind]['sha256'],
        expected_tokenizer_sha256=identities['tokenizer']['sha256'])
    native, execution = runner.load_backend(args, plan, tmp_path, time.monotonic() + 600)
    try:
        loaded = json.loads((tmp_path / 'loaded-source.json').read_text())
        assert loaded['source'] == identities[kind]
        assert loaded['tokenizer'] == identities['tokenizer']
        assert loaded['execution'] == execution
        assert execution['native_model_class'] == 'Qwen3_5ForConditionalGeneration'
        assert execution['device'] == 'cpu' and execution['attention_backend'] == 'eager'
        assert execution['use_cache'] is False and execution['cache_bytes'] == 0
        assert execution['native_generation_eos_ids'] == 2
        assert execution['cache_policy'] == policy
        assert not any(p.requires_grad for p in native.model.parameters())
        assert not any(module.training for module in native.model.modules())
        ids, positions = [10, 20, 30, 40, 259, 50, 60], [1, 4, 6]
        cold_score = native.score_tokens(ids, positions)
        cold_generation = native.generate([10, 20], 3, [2])
        assert cold_score['target_logprobs'] == pytest.approx(
            direct_logprobs(native.model, ids, positions), abs=1e-7)
        if kind == 'mixed':
            assert len(execution['mixed_modules']) == 3
            assert {value['row_chunk_size'] for value in execution['mixed_modules'].values()} == {37}
            assert {value['compute_dtype'] for value in execution['mixed_modules'].values()} == {'torch.bfloat16'}
            assert execution['artifact_manifest_sha256'] == hashlib.sha256(
                (native_artifacts[kind] / 'manifest.json').read_bytes()).hexdigest()
            name = 'model.language_model.layers.0.mlp.down_proj'
            expected_candidate = native_artifacts['expected_mixed'].get_submodule(name)._candidate()
            loaded_candidate = native.model.get_submodule(name)._candidate()
            assert execution['mixed_modules'][name]['rotation'] == expected_candidate.metadata['rotation']
            assert loaded_candidate.metadata == expected_candidate.metadata
            assert all(torch.equal(loaded_candidate.tensors[key], value)
                       for key, value in expected_candidate.tensors.items())
        else:
            assert execution['mixed_modules'] == {}
            assert execution['original_load_dtype'] == 'bfloat16'
        if policy == 'warm':
            receipt = runner.warm_cache(native, execution, plan, tmp_path, time.monotonic() + 600)
            assert receipt['status'] == 'passed' and receipt['all_probe_logits_bitwise_equal'] is True
            assert receipt['cache_bytes'] > 0
            assert receipt['cache_bytes'] == execution['cache_bytes'] == sum(row['bytes'] for row in receipt['modules'])
            assert receipt['serialized_payload_modified'] is False
            assert native.score_tokens(ids, positions) == cold_score
            assert native.generate([10, 20], 3, [2]) == cold_generation
            warmed_candidate = native.model.get_submodule(name)._candidate()
            assert warmed_candidate.metadata == expected_candidate.metadata
            assert all(torch.equal(warmed_candidate.tensors[key], value)
                       for key, value in expected_candidate.tensors.items())
            wrapped = runner.ExecutionBackend(native, execution)
            metadata = wrapped.metadata()
            assert metadata['native_execution']['cache_bytes'] == receipt['cache_bytes']
            execution['cache_bytes'] = -1
            assert wrapped.metadata() == metadata  # Immutable run receipt.
        assert {key: local_directory_identity(native_artifacts[key]) for key in identities} == identities
    finally:
        native.close()


def test_complete_native_cpu_subprocess_cli(native_artifacts, tmp_path):
    backend = load_native_backend(native_artifacts, 'original')
    manifest, profile = native_manifest(backend)
    expected = direct_logprobs(backend.model, manifest['nll'][0]['input_ids'],
                              manifest['nll'][0]['score_positions'])
    backend.close()
    source_before = {key: local_directory_identity(native_artifacts[key]) for key in ['original', 'tokenizer']}
    frozen_paths = {}
    for name, value in [('manifest', manifest), ('profile', profile)]:
        path = tmp_path / f'{name}.json'
        path.write_text(json.dumps(value, sort_keys=True, allow_nan=False), encoding='utf-8')
        frozen_paths[name] = (path, hashlib.sha256(path.read_bytes()).hexdigest())
    output = tmp_path / 'fresh-subprocess-result'
    command = [sys.executable, str(ROOT / 'tools/evaluate_transformers_manifest.py'), '--run',
               '--kind', 'original', '--device', 'cpu', '--cache-policy', 'cold',
               '--max-seconds', '600', '--output', str(output)]
    for name, (path, sha) in frozen_paths.items():
        command.extend([f'--{name}', str(path), f'--expected-{name}-sha256', sha])
    for argument, name in [('model', 'original'), ('tokenizer', 'tokenizer')]:
        command.extend([f'--{argument}', str(native_artifacts[name]),
                        f'--expected-{argument}-sha256', source_before[name]['sha256']])
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120,
        env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='-1',
                 PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false',
                 HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1'))
    details = {path.name: path.read_text() for path in output.glob('*.json')} if output.exists() else {}
    assert completed.returncode == 0, (completed.stdout, completed.stderr, details)
    report = json.loads((output / 'evaluation.json').read_text())
    result = json.loads((output / 'result.json').read_text())
    child = json.loads((output / 'child-status.json').read_text())
    execution = json.loads((output / 'execution.json').read_text())
    loaded = json.loads((output / 'loaded-source.json').read_text())
    assert result['status'] == child['status'] == 'complete'
    assert result['model_kind'] == 'original' and result['manifest_sha256'] == manifest['frozen_sha256']
    assert result['maximum_whole_process_seconds'] == 600
    assert result['cleanup']['no_live_owned_processes'] is True
    assert result['cleanup']['live_pids_after_cleanup'] == []
    assert result['cleanup']['ownership_handle_closed'] is True
    supervisor = json.loads((output / 'supervisor-process.json').read_text())
    assert supervisor['status'] == 'complete'
    assert supervisor['cleanup'] == result['cleanup']
    assert 0 < result['whole_process_elapsed_seconds'] < 120
    assert child['models_per_child'] == 1
    assert execution == loaded['execution']
    assert execution['device'] == 'cpu' and execution['cache_policy'] == 'cold'
    assert execution['cache_bytes'] == 0 and execution['native_text_prefix'] == 'model.language_model'
    assert loaded['source'] == source_before['original'] and loaded['tokenizer'] == source_before['tokenizer']
    assert report['aggregate']['scored_tokens'] == 12
    assert report['aggregate']['choice_labelled'] == 1
    assert report['aggregate']['generations'] == 2
    assert report['nll'][0]['target_logprobs'] == pytest.approx(expected, abs=1e-7)
    assert report['legacy_fp32_cross_entropy_compatible'] is False
    assert report['backend']['native_execution'] == execution
    assert {key: local_directory_identity(native_artifacts[key]) for key in source_before} == source_before
    for path, sha in frozen_paths.values():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha
    assert not (output / 'failure.json').exists()
    assert not (output / 'cache-warmup.json').exists()
