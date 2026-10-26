"""Tokenizer and fake-worker tests; never load official model weights or C API."""
import hashlib
import json
import math
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, normalizers
from transformers import PreTrainedTokenizerFast

from openbonsai.eval_backends import _byte_decoder, local_directory_identity
from openbonsai.evaluation import EVALUATION_PROTOCOL, REDUCTION, validate_manifest
from tools import preflight_bonsai_reference as preflight


class TokenizerFixture:
    """Real ByteLevel/NFC processing; remap only test specials to Qwen IDs."""
    is_fast = True

    def __init__(self):
        vocabulary = {char: i for i, char in enumerate(_byte_decoder())}
        native = Tokenizer(models.BPE(vocab=vocabulary, merges=[]))
        native.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        native.decoder = decoders.ByteLevel()
        native.normalizer = normalizers.NFC()
        self.native = PreTrainedTokenizerFast(tokenizer_object=native,
            additional_special_tokens=['<|endoftext|>', '<|im_start|>', '<|im_end|>'])
        self.backend_tokenizer = self.native.backend_tokenizer
        self.forward = {256: 248044, 257: 248045, 258: 248046}
        self.reverse = {value: key for key, value in self.forward.items()}

    def get_added_vocab(self):
        return {key: self.forward.get(value, value) for key, value in self.native.get_added_vocab().items()}

    def convert_tokens_to_ids(self, text):
        value = self.native.convert_tokens_to_ids(text)
        return self.forward.get(value, value)

    def convert_ids_to_tokens(self, values):
        return self.native.convert_ids_to_tokens([self.reverse.get(value, value) for value in values])

    def encode(self, text, **kwargs):
        return [self.forward.get(value, value) for value in self.native.encode(text, **kwargs)]

    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{'role': 'user', 'content': preflight.INSTRUCTION}]
        assert kwargs == {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False}
        return preflight.NONTHINKING_PROMPT


def test_fixed_official_identity_and_tokenizer_only_manifest_bounds():
    assert preflight.GGUF_SHA256 == '3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b'
    assert preflight.GGUF_BYTES == 1997486976
    reference = preflight.TokenizerReference(TokenizerFixture(), {'sha256': 'a' * 64})
    assert reference.byte_decoder == _byte_decoder()
    manifest = preflight.build_manifest(reference)
    validate_manifest(manifest, context_length=256, vocab_size=248320)
    assert manifest['role'] == 'search_validation'
    assert manifest['metadata']['scored_targets_including_reset'] < 100
    assert manifest['generations'][0]['max_new_tokens'] == 16
    assert manifest['generations'][0]['eos_ids'] == [248044, 248046]
    reset = manifest['metadata']['state_reset_probe']
    assert len(reset['a']['input_ids']) == 97
    assert reset['a']['score_positions'] == [1, 63, 64, 65, 96]
    text = next(value for value in manifest['texts'] if value['id'] == 'combining')
    assert bytes.fromhex(text['raw_utf8_hex']).startswith('e\u0301'.encode())
    assert bytes.fromhex(text['rendered_utf8_hex']).startswith('é'.encode())
    for item in manifest['texts']:
        assert b''.join(reference.token_pieces(item['input_ids'])) == bytes.fromhex(item['rendered_utf8_hex'])


@pytest.mark.parametrize('change', ['decoder', 'normalizer', 'template', 'eos'])
def test_tokenizer_contract_refuses_incompatible_assets(monkeypatch, change):
    tokenizer = TokenizerFixture()
    if change == 'decoder': tokenizer.backend_tokenizer.decoder = decoders.WordPiece()
    elif change == 'normalizer': tokenizer.backend_tokenizer.normalizer = None
    elif change == 'template': monkeypatch.setattr(tokenizer, 'apply_chat_template', lambda *a, **k: 'thinking prompt')
    elif change == 'eos': monkeypatch.setattr(tokenizer, 'convert_tokens_to_ids', lambda _: 0)
    with pytest.raises(ValueError):
        preflight.TokenizerReference(tokenizer, {'sha256': 'a' * 64})


@pytest.fixture
def setup_run(tmp_path, monkeypatch):
    from openbonsai import eval_backends
    tokenizer = TokenizerFixture()
    directory = tmp_path / 'tokenizer'
    directory.mkdir()
    (directory / 'fixture').write_bytes(b'not model weights; tokenizer loader replaced by fixture')
    identity = local_directory_identity(directory)
    reference = preflight.TokenizerReference(tokenizer, identity)
    gguf, worker = tmp_path / 'official.fixture', tmp_path / 'worker.fixture'
    gguf.write_bytes(b'fake GGUF, no native runtime execution')
    worker.write_bytes(b'fake executable, never executed')
    monkeypatch.setattr(preflight, 'GGUF_BYTES', gguf.stat().st_size)
    monkeypatch.setattr(preflight, 'GGUF_SHA256', hashlib.sha256(gguf.read_bytes()).hexdigest())
    worker_sha = hashlib.sha256(worker.read_bytes()).hexdigest()
    load_calls = []
    def load(path, **kwargs):
        load_calls.append((path, kwargs))
        return tokenizer
    monkeypatch.setattr('transformers.AutoTokenizer.from_pretrained', load)
    created, events = [], []
    behavior = {'fail_score_call': None, 'native_eog': [248044, 248046], 'fresh_drift': False}

    class FakeBackend:
        def __init__(self, executable, model, **kwargs):
            assert Path(executable) == worker and Path(model) == gguf
            assert kwargs['context_length'] == 256 and kwargs['threads'] == 4 and kwargs['gpu_layers'] == 0
            assert kwargs['expected_executable_sha256'] == worker_sha
            assert kwargs['expected_model_sha256'] == preflight.GGUF_SHA256
            assert (tmp_path / 'out' / 'frozen-inputs.json').is_file()
            if created:
                assert created[-1].closed  # Never keep two model workers loaded.
            self.closed = False
            self.index = len(created)
            self.calls = 0
            self.kwargs = kwargs
            created.append(self)
            events.append(('start', self.index))

        def metadata(self):
            return {'protocol': EVALUATION_PROTOCOL, 'reduction': REDUCTION, 'context_length': 256,
                    'vocab_size': 248320, 'cache_mode': 'full-prefix-reference', 'reset_policy': 'per-record',
                    'source_identity': {'sha256': preflight.GGUF_SHA256}, 'tokenizer_identity': identity,
                    'native_eog_ids': behavior['native_eog'], 'gpu_layers': 0, 'threads': 4}

        def tokenize(self, text, parse_special=False): return reference.tokenize(text, parse_special)
        def token_pieces(self, ids): return reference.token_pieces(ids)

        def score_tokens(self, ids, positions):
            self.calls += 1
            events.append(('score', self.index, list(positions)))
            if self.calls == behavior['fail_score_call']:
                raise RuntimeError('intentional scored failure')
            value = -1.1 if self.index and behavior['fresh_drift'] else -1.
            return {'score_positions': positions, 'target_logprobs': [value] * len(positions),
                    'nll_sum': -value * len(positions), 'count': len(positions)}

        def generate(self, ids, maximum, eos):
            token = reference.tokenize('t')[0]
            return {'generated_ids': [token] if token in eos else [token] * maximum,
                    'stop_reason': 'eos' if token in eos else 'length'}

        def close(self):
            self.closed = True
            events.append(('close', self.index))

    monkeypatch.setattr(eval_backends, 'PrismBackend', FakeBackend)
    kwargs = {'output': tmp_path / 'out', 'gguf': gguf, 'worker': worker, 'prism_source': tmp_path / 'source',
              'tokenizer_path': directory, 'expected_tokenizer_sha256': identity['sha256'],
              'expected_worker_sha256': worker_sha}
    return kwargs, created, events, behavior, load_calls


def test_complete_fake_preflight_freezes_before_scoring_closes_and_records_every_call(setup_run):
    kwargs, created, events, behavior, load_calls = setup_run
    # Different native EOG is visible; it never rewrites the declared EOS policy.
    behavior['native_eog'] = [248044]
    report = preflight.run(**kwargs)
    assert report['passed'] and not report['quality_claim'] and not report['original_bf16_comparison']
    assert report['eog']['same_set'] is False
    assert report['eog']['expected_evaluation_eos_ids'] == [248044, 248046]
    assert report['state_reset']['exact_repeat'] and report['fresh_process_state_reset']['exact_repeat']
    assert len(created) == 2 and all(backend.closed for backend in created)
    assert events.index(('close', 0)) < events.index(('start', 1))
    assert load_calls == [(str(kwargs['tokenizer_path']), {'local_files_only': True, 'trust_remote_code': False})]
    outputs = list(kwargs['output'].glob('scored-call-*.json'))
    assert len(outputs) == report['work_budget']['model_score_calls_maximum'] + 2
    calls = [json.loads(path.read_text()) for path in outputs]
    assert all(call['status'] == 'complete' for call in calls)
    targets = sum(len(call['request']['score_positions']) for call in calls if call['op'] == 'score')
    assert targets == report['work_budget']['scored_targets_including_reset'] < 100
    assert (kwargs['output'] / 'scored-output.json').is_file()
    assert json.loads((kwargs['output'] / 'report.json').read_text())['status'] == 'complete'


def test_failed_score_retains_frozen_manifest_prior_results_and_closes(setup_run):
    kwargs, created, _, behavior, _ = setup_run
    behavior['fail_score_call'] = 2
    with pytest.raises(RuntimeError, match='intentional scored failure'):
        preflight.run(**kwargs)
    assert created[0].closed
    report = json.loads((kwargs['output'] / 'report.json').read_text())
    assert report['status'] == 'failed' and not report['passed']
    assert (kwargs['output'] / 'frozen-inputs.json').is_file()
    assert json.loads((kwargs['output'] / 'scored-call-01.json').read_text())['status'] == 'complete'
    assert json.loads((kwargs['output'] / 'scored-call-02.json').read_text())['status'] == 'failed'


def test_fresh_process_drift_fails_closed(setup_run):
    kwargs, created, _, behavior, _ = setup_run
    behavior['fresh_drift'] = True
    with pytest.raises(AssertionError, match='fresh worker'):
        preflight.run(**kwargs)
    assert all(backend.closed for backend in created)
    assert not json.loads((kwargs['output'] / 'report.json').read_text())['passed']


def test_report_write_failure_still_closes_worker(setup_run, monkeypatch):
    kwargs, created, _, _, _ = setup_run
    save = preflight.save
    def failing(path, value):
        if Path(path).name == 'report.json' and created:
            raise OSError('intentional disk failure')
        save(path, value)
    monkeypatch.setattr(preflight, 'save', failing)
    with pytest.raises(OSError, match='disk failure'):
        preflight.run(**kwargs)
    assert created and all(backend.closed for backend in created)


@pytest.mark.parametrize('timeout', [math.nan, math.inf, -1, 601])
def test_invalid_timeout_still_saves_json_failure_before_loading(setup_run, timeout):
    kwargs, created, _, _, load_calls = setup_run
    with pytest.raises(ValueError, match='timeout'):
        preflight.run(**kwargs, request_timeout=timeout)
    assert not created and not load_calls
    assert json.loads((kwargs['output'] / 'report.json').read_text())['status'] == 'failed'


@pytest.mark.parametrize('change', ['gguf_size', 'gguf_hash', 'worker_hash', 'tokenizer_hash'])
def test_input_identity_failure_precedes_any_runtime_or_model_load(setup_run, change):
    kwargs, created, _, _, load_calls = setup_run
    if change == 'gguf_size': kwargs['gguf'].write_bytes(b'wrong size')
    elif change == 'gguf_hash': kwargs['gguf'].write_bytes(b'x' * preflight.GGUF_BYTES)
    elif change == 'worker_hash': kwargs['expected_worker_sha256'] = 'b' * 64
    elif change == 'tokenizer_hash': kwargs['expected_tokenizer_sha256'] = 'c' * 64
    with pytest.raises(ValueError):
        preflight.run(**kwargs)
    assert not created and not load_calls
    assert json.loads((kwargs['output'] / 'report.json').read_text())['status'] == 'failed'
