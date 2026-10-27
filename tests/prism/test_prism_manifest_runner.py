"""Portable runner controls: synthetic tokens, fake backend, harmless processes."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from openbonsai import evaluation

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('prism_manifest_runner', ROOT / 'tools/evaluate_prism_manifest.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def pieces(ids):
    return [bytes([64 + value]) for value in ids]


def frozen_text(name, ids):
    raw = b''.join(pieces(ids))
    return evaluation.freeze_text(name, raw_text=raw, rendered_text=raw,
                                  input_ids=ids, token_pieces=pieces(ids))


def fixture():
    texts, nll, choices, generations = [], [], [], []
    for index, ids in enumerate((list(range(1, 10)), list(range(2, 9)))):
        name = 'synthetic-nll-' + str(index)
        texts.append(frozen_text(name, ids))
        nll.append(dict(id=name, text_id=name, token_start=0, token_end=len(ids),
                        input_ids=ids, score_positions=list(range(1, len(ids)))))
    for index, (prefix, labels) in enumerate((([11, 12, 13], [21, 22]),
                                             ([14, 15, 16, 17, 18], [23, 24, 25]))):
        options = []
        for option_index, label in enumerate(labels):
            name = f'synthetic-choice-{index}-{option_index}'
            ids = prefix + [label]
            texts.append(frozen_text(name, ids))
            options.append(dict(id=str(option_index), text_id=name, input_ids=ids,
                                prefix_length=len(prefix), score_positions=[len(prefix)]))
        choices.append(dict(id='synthetic-choice-' + str(index), options=options, answer_id='0'))
    for index, (ids, cap) in enumerate((([31, 32], 3), ([33, 34, 35], 4))):
        name = 'synthetic-generation-' + str(index)
        texts.append(frozen_text(name, ids))
        generations.append(dict(id=name, text_id=name, input_ids=ids, max_new_tokens=cap, eos_ids=[99]))
    manifest = evaluation.freeze_manifest(source_provenance={'kind': 'synthetic-unit-test', 'sha256': 'a' * 64},
        tokenizer_identity={'kind': 'fake-byte-tokenizer', 'sha256': 'b' * 64},
        texts=texts, nll=nll, choices=choices, generations=generations, role='search_validation')
    profile = {'format': runner.PROFILE_FORMAT, 'manifest_sha256': manifest['frozen_sha256'],
               'context_length': 9, 'vocab_size': 128, 'threads': 1,
               'resource_probes': {'nll': {'record_index': 0},
                                   'choice': {'question_index': 1, 'option_index': 0},
                                   'generation': {'text_index': 0, 'token_start': 0,
                                                  'token_end': 8, 'target_position': 7}}}
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

    def advance(self, seconds=.01):
        self.now += seconds


class FakeBackend:
    def __init__(self, manifest, clock, *, mismatch=False, fail_score=None, score_seconds=.01):
        self.clock, self.mismatch = clock, mismatch
        self.fail_score, self.score_seconds = fail_score, score_seconds
        self.score_calls, self.generations, self.tokenizations = [], [], []
        self.lookup = {(bytes.fromhex(row['rendered_utf8_hex']).decode(), row['parse_special']): row['input_ids']
                       for row in manifest['texts']}
        self.raw_pieces = {token: bytes.fromhex(piece) for row in manifest['texts']
                           for token, piece in zip(row['input_ids'], row['token_pieces_hex'])}
        self.raw_pieces[99] = b'<EOS>'

    def metadata(self):
        self.clock.advance()
        return dict(protocol=evaluation.EVALUATION_PROTOCOL, reduction=evaluation.REDUCTION,
                    cache_mode='full-prefix-reference', reset_policy='per-record', context_length=9,
                    vocab_size=128, source_identity={'sha256': 'c' * 64}, tokenizer_identity={'sha256': 'd' * 64})

    def tokenize(self, text, parse_special=False):
        self.clock.advance()
        self.tokenizations.append((text, parse_special))
        ids = list(self.lookup[text, parse_special])
        if self.mismatch:
            ids[0] += 1
        return ids

    def token_pieces(self, ids):
        self.clock.advance()
        return [self.raw_pieces[token] for token in ids]

    def score_tokens(self, ids, positions):
        self.clock.advance(self.score_seconds)
        self.score_calls.append((list(ids), list(positions)))
        if len(self.score_calls) == self.fail_score:
            raise RuntimeError('intentional fake score failure')
        return dict(score_positions=list(positions), target_logprobs=[-1.] * len(positions),
                    nll_sum=float(len(positions)), count=len(positions))

    def generate(self, ids, maximum, eos):
        self.clock.advance()
        self.generations.append((list(ids), maximum, list(eos)))
        return dict(generated_ids=[99], stop_reason='eos')


def test_workload_is_computed_from_manifest_without_private_data():
    manifest, profile = fixture()
    plan = runner.build_plan(manifest, profile)
    assert plan['workload'] == dict(nll_records=2, nll_targets=14, choice_questions=2,
        choice_labelled=2, choice_option_calls=5, choice_targets=5, generation_records=2,
        generation_tokens_maximum=7, generation_forward_positions_maximum=6,
        generation_context_guard_maximum=7, tokenization_texts=9)
    assert plan['probes'][2]['shape'] == dict(input_tokens=8, decoder_positions=7,
                                            full_vocab_rows=1, max_rows_per_batch=1)


def test_full_evaluator_preserves_all_tokens_positions_caps_and_eos(tmp_path):
    manifest, profile = fixture()
    clock = Clock()
    backend = FakeBackend(manifest, clock)
    result = runner.evaluate_plan(manifest, profile, backend, tmp_path, 10000, clock=clock)
    assert result['status'] == 'complete' and result['aggregate']['scored_tokens'] == 14
    assert len(backend.score_calls) == 3 + 2 + 5 and len(backend.generations) == 2
    assert len(backend.tokenizations) == 9 * 2
    expected_scores = [(row['input_ids'], row['score_positions']) for row in manifest['nll']]
    expected_scores += [(row['input_ids'], row['score_positions'])
                        for question in manifest['choices'] for row in question['options']]
    assert backend.score_calls[3:] == expected_scores
    assert backend.generations == [(row['input_ids'], row['max_new_tokens'], row['eos_ids'])
                                   for row in manifest['generations']]
    admission = json.loads((tmp_path / 'admission.json').read_text())
    assert admission['workload']['generation_tokens_maximum'] == 7 and not admission['early_eos_assumed']
    assert admission['components_seconds']['generation_full_cap'] == pytest.approx(7 * .01)
    report = json.loads((tmp_path / 'evaluation.json').read_text())
    assert report['legacy_fp32_cross_entropy_compatible'] is False
    calls = [json.loads(path.read_text()) for path in (tmp_path / 'calls').glob('*.json')]
    assert calls and all(row['status'] == 'complete' for row in calls)
    assert all(row['manifest_sha256'] == manifest['frozen_sha256'] for row in calls)


def test_native_tokenization_failure_precedes_any_score(tmp_path):
    manifest, profile = fixture()
    clock = Clock()
    backend = FakeBackend(manifest, clock, mismatch=True)
    with pytest.raises(ValueError, match='Tokenization mismatch'):
        runner.evaluate_plan(manifest, profile, backend, tmp_path, 10000, clock=clock)
    assert not backend.score_calls and not backend.generations
    assert not (tmp_path / 'admission.json').exists()


def test_admission_rejects_before_full_suite(tmp_path):
    manifest, profile = fixture()
    clock = Clock()
    backend = FakeBackend(manifest, clock, score_seconds=1.)
    with pytest.raises(RuntimeError, match='does not fit'):
        runner.evaluate_plan(manifest, profile, backend, tmp_path, 310, clock=clock)
    assert len(backend.score_calls) == 3 and not backend.generations
    assert json.loads((tmp_path / 'admission.json').read_text())['admitted'] is False
    assert not (tmp_path / 'evaluation.json').exists()


def test_failed_call_preserves_partial_receipts(tmp_path):
    manifest, profile = fixture()
    clock = Clock()
    backend = FakeBackend(manifest, clock, fail_score=4)
    with pytest.raises(RuntimeError, match='intentional fake'):
        runner.evaluate_plan(manifest, profile, backend, tmp_path, 10000, clock=clock)
    calls = [json.loads(path.read_text()) for path in (tmp_path / 'calls').glob('*.json')]
    failed = [row for row in calls if row['status'] == 'failed']
    assert len(failed) == 1 and failed[0]['phase'] == 'full-fixed-evaluation-suite'
    assert failed[0]['request']['score_positions'] == list(range(1, 9))
    assert (tmp_path / 'resource-preflight.json').exists()
    assert not (tmp_path / 'evaluation.json').exists()


def test_longest_nll_input_does_not_override_larger_logit_row_work():
    manifest, profile = fixture()
    first = manifest['nll'][0]
    first['score_positions'] = [8]
    manifest['nll'].insert(1, {**copy.deepcopy(first), 'id': 'synthetic-dense-prefix',
        'input_ids': first['input_ids'][:8], 'token_end': 8, 'score_positions': list(range(1, 8))})
    reseal(manifest, profile)
    evaluation.validate_manifest(manifest)
    with pytest.raises(ValueError, match='NLL probe does not dominate'):
        runner.build_plan(manifest, profile)


def test_longest_choice_input_does_not_override_multitoken_label_rows():
    manifest, profile = fixture()
    option = manifest['choices'][0]['options'][0]
    option['input_ids'].append(26)
    option['score_positions'].append(4)
    index = next(i for i, row in enumerate(manifest['texts']) if row['id'] == option['text_id'])
    manifest['texts'][index] = frozen_text(option['text_id'], option['input_ids'])
    reseal(manifest, profile)
    evaluation.validate_manifest(manifest)
    with pytest.raises(ValueError, match='Choice probe does not dominate'):
        runner.build_plan(manifest, profile)


def test_output_density_is_part_of_dominance():
    sparse = runner.score_shape({'input_ids': [1] * 130, 'score_positions': [1, 2, 65, 66]})
    dense = runner.score_shape({'input_ids': [1] * 10, 'score_positions': [1, 2, 3, 4]})
    assert sparse['full_vocab_rows'] == dense['full_vocab_rows'] == 4
    assert not runner.dominates(sparse, dense)


@pytest.mark.parametrize('mutation,message', [
    (lambda p: p.update(manifest_sha256='0' * 64), 'different frozen manifest'),
    (lambda p: p.update(unrecognized=True), 'Invalid profile fields'),
    (lambda p: p.update(threads=True), 'Invalid CPU threads'),
    (lambda p: p['resource_probes']['nll'].update(record_index=99), 'Invalid NLL index'),
    (lambda p: p['resource_probes']['choice'].update(question_index=0), 'Choice probe does not dominate'),
    (lambda p: p['resource_probes']['generation'].update(token_end=6, target_position=5), 'shorter than a capped prefix'),
    (lambda p: p['resource_probes']['generation'].update(token_end=500), 'Invalid generation slice end'),
    (lambda p: p['resource_probes']['generation'].update(target_position=6), 'last source logit'),
])
def test_profile_mismatch_fails_closed(mutation, message):
    manifest, profile = fixture()
    mutation(profile)
    with pytest.raises(ValueError, match=message):
        runner.build_plan(manifest, profile)


def test_unreferenced_text_cannot_be_a_resource_probe_reservoir():
    manifest, profile = fixture()
    manifest['texts'].append(frozen_text('unused-text', list(range(1, 10))))
    profile['resource_probes']['generation']['text_index'] = len(manifest['texts']) - 1
    reseal(manifest, profile)
    with pytest.raises(ValueError, match='not an existing evaluation input'):
        runner.build_plan(manifest, profile)


def test_forecast_counts_margin_and_full_cap_are_derived():
    manifest, profile = fixture()
    plan = runner.build_plan(manifest, profile)
    result = runner.forecast(plan, dict(nll=2, choice=3, generation_last_logit=4, tokenization=5), 10000)
    assert result['projected_remaining_seconds'] == 1.25 * (2 * 2 + 5 * 3 + 7 * 4 + 5) + 300


@pytest.mark.parametrize('value', [0, -1, float('inf'), float('nan')])
def test_invalid_probe_duration_cannot_admit(value):
    manifest, profile = fixture()
    with pytest.raises(ValueError):
        runner.forecast(runner.build_plan(manifest, profile),
                        dict(nll=value, choice=1, generation_last_logit=1, tokenization=1), 10000)


def test_deadline_prevents_backend_call(tmp_path):
    manifest, profile = fixture()
    clock = Clock()
    backend = FakeBackend(manifest, clock)
    wrapped = runner.RecordedBackend(backend, tmp_path / 'calls', 0, manifest['frozen_sha256'], clock)
    with pytest.raises(TimeoutError):
        wrapped.score_tokens([1, 2], [1])
    assert not backend.score_calls and not list((tmp_path / 'calls').iterdir())


def test_pinned_file_tampering_rejected(tmp_path):
    path = tmp_path / 'manifest.json'
    path.write_bytes(b'{"example":1}')
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert runner.read_pinned_json(path, expected) == {'example': 1}
    path.write_bytes(b'{"example":2}')
    with pytest.raises(ValueError, match='File SHA mismatch'):
        runner.read_pinned_json(path, expected)


@pytest.mark.skipif(os.name != 'posix', reason='POSIX supervisor')
def test_external_deadline_kills_harmless_runner_and_descendant(tmp_path):
    pytest.importorskip('psutil')
    code = 'import subprocess,sys,time;subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]);time.sleep(30)'
    result = runner.supervise_command([sys.executable, '-c', code], tmp_path,
                                      time.monotonic() + .7, sample_interval=.05)
    assert result['status'] == 'failed' and result['reason'] == 'whole-process-deadline'
    assert result['cleanup']['no_live_group_members'] and not runner.group_members(result['pid'])
    assert result['elapsed_seconds'] < 4 and result['sampled_peak_summed_rss_bytes'] > 0
    assert len(result['observed_processes']) >= 2


@pytest.mark.skipif(os.name != 'posix', reason='POSIX supervisor')
def test_normal_exit_cleans_orphaned_harmless_descendant(tmp_path):
    pytest.importorskip('psutil')
    code = 'import subprocess,sys,time;subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"]);time.sleep(.15)'
    result = runner.supervise_command([sys.executable, '-c', code], tmp_path,
                                      time.monotonic() + 5, sample_interval=.03)
    assert result['reason'] == 'exited' and result['exit_code'] == 0
    assert result['cleanup']['pids_before_cleanup'] and result['cleanup']['no_live_group_members']
    assert not runner.group_members(result['pid'])


@pytest.mark.skipif(os.name != 'posix', reason='POSIX supervisor')
def test_supervisor_sigterm_also_cleans_harmless_child(tmp_path):
    pytest.importorskip('psutil')
    code = ("import importlib.util,pathlib,sys,time;"
            f"s=importlib.util.spec_from_file_location('r',{str(ROOT/'tools/evaluate_prism_manifest.py')!r});"
            "r=importlib.util.module_from_spec(s);s.loader.exec_module(r);"
            f"r.supervise_command([sys.executable,'-c','import time;time.sleep(30)'],pathlib.Path({str(tmp_path)!r}),time.monotonic()+20,sample_interval=.05)")
    process = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        until, report = time.monotonic() + 5, None
        while time.monotonic() < until:
            path = tmp_path / 'supervisor-process.json'
            if path.exists():
                report = json.loads(path.read_text())
                if report.get('status') == 'running':
                    break
            time.sleep(.02)
        assert report and report.get('status') == 'running'
        process.terminate()
        process.communicate(timeout=4)
        final = json.loads((tmp_path / 'supervisor-process.json').read_text())
        assert final['status'] == 'failed' and 'InterruptedError' in final['reason']
        assert final['cleanup']['no_live_group_members'] and not runner.group_members(final['pid'])
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2)
        path = tmp_path / 'supervisor-process.json'
        if path.exists():
            group = json.loads(path.read_text()).get('pid')
            if group:
                runner.cleanup_group(group, time.monotonic())


def test_no_implicit_cli_launch():
    result = subprocess.run([sys.executable, str(ROOT/'tools/evaluate_prism_manifest.py')],
                            capture_output=True, text=True)
    assert result.returncode != 0 and '--run and all explicit' in result.stderr


def test_published_profile_has_no_tokens_prompts_or_answers():
    profile = json.loads((ROOT/'configs/evaluation/bonsai-development-cpu.json').read_text())
    assert profile['format'] == runner.PROFILE_FORMAT
    encoded = json.dumps(profile)
    assert all(key not in encoded for key in ('input_ids', 'raw_text', 'answer_id', 'question_id', 'prompt'))
    assert set(profile['resource_probes']) == {'nll', 'choice', 'generation'}
