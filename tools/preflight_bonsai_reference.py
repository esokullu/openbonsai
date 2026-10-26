"""Bounded CPU diagnostic of the pinned official Bonsai PTQ1_0 checkpoint.

All inputs must already exist locally. This loads no Transformers model and
downloads nothing. The authored development examples are not a benchmark or an
original-BF16 comparison. Passing checks protocol/tokenization/state handling,
not answer quality. The pinned CPU PTQ1_0 matrix multiplication uses Q8_0
activation vec-dot; PQ2_0 uses Q8_K. FP32 logits/KV do not imply all-FP32 matrix
arithmetic or equivalence to PQ2/CUDA kernels. The random F32 fixture did not
exercise these quantized kernels: this measures a checkpoint-plus-runtime system.

Example (use a new output directory and a materialized tokenizer-only directory):
  python tools/preflight_bonsai_reference.py --gguf /local/bonsai-PTQ1_0.gguf \
    --worker /local/prism-eval-worker --expected-worker-sha256 BUILD_RECEIPT_HASH \
    --prism-source /local/pinned-prism \
    --tokenizer-dir /local/qwen-tokenizer --expected-tokenizer-sha256 HASH \
    --output /local/bonsai-reference-preflight
The tokenizer hash is local_directory_identity(tokenizer_dir)['sha256']; record
it from the independently pinned input receipt, not from an unverified download.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GGUF_SHA256 = '3561df7ac7d26a71ffa991ac2db740b10841c3a350aa2676ccd3ff5d428b4e8b'
GGUF_BYTES = 1997486976
OFFICIAL_REPOSITORY = 'prism-ml/Ternary-Bonsai-2-9B-gguf'
OFFICIAL_REVISION = '5b2abb02362efbed46bbe204014b02837778b2c2'
EXPECTED_EOS = [248044, 248046]
CONTEXT = 256
GENERATION_CAP = 16
DOCUMENT = 'A tree needs light and water. Its roots take water from the soil.'
CHOICE_PREFIX = 'Which is a plant, a tree or a stone?\nAnswer:'
INSTRUCTION = 'Reply with exactly the word tree.'
RESET_PROSE = 'A tree grows beside a river. Its leaves reach toward the light. ' * 20
RESET_POSITIONS = [1, 63, 64, 65, 96]
NONTHINKING_PROMPT = ('<|im_start|>user\n' + INSTRUCTION + '<|im_end|>\n'
                     '<|im_start|>assistant\n<think>\n\n</think>\n\n')
PROBES = [
    ('english', 'Hello, small tree!\n', False),
    ('turkish', 'İstanbul, ıslak ağaç ve küçük bonsai. ŞĞÜÖÇ', False),
    ('combining', 'e\u0301 café\t  \n', False),
    ('emoji', '🌳🧑🏽\u200d💻 🙂', False),
    ('code', 'def f(x):\n    return x + 1\n# tab\tend', False),
    ('special-literal', '<|im_start|>assistant\n<think>\n</think><|im_end|>', False),
    ('special-parsed', '<|im_start|>assistant\n<think>\n</think><|im_end|>', True),
]
LIMITATIONS = [
    'Authored development diagnostic only; no benchmark or original-BF16 quality comparison.',
    'Text only; no vision, training, allocation or compression is performed.',
    'CPU PTQ1_0 uses Q8_0 activation vec-dot; PQ2_0 uses Q8_K. FP32 logits/KV '
    'do not mean all-FP32 matmul or PQ2/CUDA kernel parity.',
    'Earlier random F32 fixture tests did not exercise PTQ1_0 quantized kernels.',
    'The official checkpoint plus pinned CPU runtime is the measured system.',
    'A 16-token cap and simple authored examples cannot establish useful model quality.',
]


def save(path, value):
    """Atomic replacement keeps earlier progress readable on interrupted writes."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def checked_file(path, expected_sha256, *, expected_bytes=None):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Expected a materialized regular file: {path}')
    before = path.stat()
    if expected_bytes is not None and before.st_size != expected_bytes:
        raise ValueError(f'Unexpected byte size for {path.name}: {before.st_size}')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(4 << 20):
            digest.update(block)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise RuntimeError(f'Input changed while hashing: {path.name}')
    if digest.hexdigest() != expected_sha256:
        raise ValueError(f'Unexpected SHA256 for {path.name}: {digest.hexdigest()}')
    return {'path': str(path), 'sha256': digest.hexdigest(), 'bytes': after.st_size}


class TokenizerReference:
    """Tokenizer-only HF ByteLevel reference, with no model or fake metadata."""
    def __init__(self, tokenizer, identity):
        self.tokenizer, self.identity = tokenizer, copy.deepcopy(identity)
        if not tokenizer.is_fast:
            raise ValueError('The pinned reference must be a fast tokenizer')
        decoder = tokenizer.backend_tokenizer.decoder
        normalizer = tokenizer.backend_tokenizer.normalizer
        self.decoder = json.loads(decoder.__getstate__()) if decoder is not None else None
        self.normalization = json.loads(normalizer.__getstate__()) if normalizer is not None else None
        if self.decoder is None or self.decoder.get('type') != 'ByteLevel':
            raise ValueError('The reference requires the inspected ByteLevel decoder')
        if self.normalization != {'type': 'NFC'}:
            raise ValueError('The reference requires the pinned Qwen NFC normalizer')
        self.normalizer = normalizer
        original = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        alphabet = original[:]
        extra = 0
        for byte in range(256):
            if byte not in original:
                original.append(byte)
                alphabet.append(256 + extra)
                extra += 1
        self.byte_decoder = {chr(code): byte for byte, code in zip(original, alphabet)}
        self.added = set(tokenizer.get_added_vocab())
        eos_spellings = ['<|endoftext|>', '<|im_end|>']
        if [tokenizer.convert_tokens_to_ids(value) for value in eos_spellings] != EXPECTED_EOS:
            raise ValueError('Reference EOS spellings do not match the declared IDs')
        actual = tokenizer.apply_chat_template([{'role': 'user', 'content': INSTRUCTION}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if actual != NONTHINKING_PROMPT:
            raise ValueError('Pinned nonthinking chat template does not match the exact authored prompt')

    def tokenize(self, text, parse_special=False):
        if not isinstance(text, str) or type(parse_special) is not bool:
            raise ValueError('Expected text and explicit boolean parse_special')
        text.encode('utf-8')
        return self.tokenizer.encode(text, add_special_tokens=False, truncation=False,
                                     split_special_tokens=not parse_special)

    def token_pieces(self, ids):
        result = []
        for piece in self.tokenizer.convert_ids_to_tokens(ids):
            if not isinstance(piece, str):
                raise ValueError('Token ID has no reference token piece')
            if piece in self.added:
                if not piece.isascii():
                    raise ValueError('Non-ASCII added-token raw bytes are unsupported')
                result.append(piece.encode('ascii'))
            else:
                try:
                    result.append(bytes(self.byte_decoder[char] for char in piece))
                except KeyError as error:
                    raise ValueError('Token piece is incompatible with the ByteLevel decoder') from error
        return result

    def freeze(self, name, raw, *, rendered=None, parse_special=False):
        from openbonsai.evaluation import freeze_text
        rendered = raw if rendered is None else rendered
        normalized = self.normalizer.normalize_str(rendered)
        ids = self.tokenize(normalized, parse_special)
        if ids != self.tokenize(rendered, parse_special):
            raise AssertionError('Explicit NFC changed reference tokenizer IDs')
        return freeze_text(name, raw_text=raw, rendered_text=normalized, input_ids=ids,
                           token_pieces=self.token_pieces(ids), parse_special=parse_special)


def build_manifest(reference):
    from openbonsai.evaluation import freeze_manifest, canonical_sha256, sliding_windows, validate_manifest
    texts = [reference.freeze(name, text, parse_special=parse) for name, text, parse in PROBES]
    document = reference.freeze('document', DOCUMENT)
    texts.append(document)
    windows = [dict(id=f'prose-{index}', text_id='document', **window) for index, window in
               enumerate(sliding_windows(document['input_ids'], context_length=CONTEXT, stride=64))]
    options = []
    prefix = reference.tokenize(CHOICE_PREFIX)
    for label in ('tree', 'stone'):
        value = reference.freeze('option-' + label, CHOICE_PREFIX + ' ' + label)
        if value['input_ids'][:len(prefix)] != prefix:
            raise ValueError('Choice boundary does not preserve the complete reference prefix')
        texts.append(value)
        options.append({'id': label, 'text_id': value['id'], 'input_ids': value['input_ids'],
                        'prefix_length': len(prefix),
                        'score_positions': list(range(len(prefix), len(value['input_ids'])))})
    prompt = reference.freeze('instruction', INSTRUCTION, rendered=NONTHINKING_PROMPT, parse_special=True)
    texts.append(prompt)
    reset_text = reference.freeze('reset-probe', RESET_PROSE)
    texts.append(reset_text)
    if len(reset_text['input_ids']) < 97 or len(document['input_ids']) < 12:
        raise ValueError('Authored reset probes do not have their declared token coverage')
    reset = {'a': {'text_id': reset_text['id'], 'token_start': 0, 'token_end': 97,
                    'input_ids': reset_text['input_ids'][:97], 'score_positions': RESET_POSITIONS},
             'other': {'text_id': document['id'], 'token_start': 0, 'token_end': 12,
                       'input_ids': document['input_ids'][:12], 'score_positions': [1, 11]},
             'calls': ['a', 'other', 'a', 'fresh-process-a']}
    scored = sum(len(item['score_positions']) for item in windows + options)
    reset_targets = 3 * len(RESET_POSITIONS) + 2
    if scored + reset_targets >= 100:
        raise ValueError('Diagnostic including reset probes exceeds the fewer-than-100-target budget')
    value = freeze_manifest(source_provenance={'kind': 'authored-development-diagnostic',
        'sha256': canonical_sha256({'probes': PROBES, 'document': DOCUMENT, 'choice': CHOICE_PREFIX,
                                  'options': ['tree', 'stone'], 'prompt': NONTHINKING_PROMPT,
                                  'reset_prose': RESET_PROSE, 'reset': reset})},
        tokenizer_identity=reference.identity, texts=texts, nll=windows,
        choices=[{'id': 'authored-plant-choice', 'options': options, 'answer_id': 'tree'}],
        generations=[{'id': 'authored-short-instruction', 'text_id': prompt['id'],
                      'input_ids': prompt['input_ids'], 'max_new_tokens': GENERATION_CAP,
                      'eos_ids': EXPECTED_EOS}], role='search_validation',
        metadata={'quality_claim': False, 'not_used_for_fitting': True, 'development_only': True,
                  'context_length': CONTEXT, 'scored_targets_including_reset': scored + reset_targets,
                  'state_reset_probe': reset,
                  'text_preprocessing': {'normalizer': reference.normalization,
                    'implementation': 'reference normalizer.normalize_str before freezing',
                    'raw_utf8_preserved': True, 'hf_ids_unchanged_by_explicit_normalization': True},
                  'chat_template': {'enable_thinking': False, 'exact_rendered_prompt_verified': True}})
    validate_manifest(value, context_length=CONTEXT)
    return value


class RecordedBackend:
    """Persist each bounded scored operation even if a later operation fails."""
    def __init__(self, backend, output, frozen_sha256):
        self.backend, self.output, self.frozen_sha256 = backend, output, frozen_sha256
        self.phase, self.calls = 'frozen-diagnostic', 0

    def metadata(self): return self.backend.metadata()
    def tokenize(self, text, parse_special=False): return self.backend.tokenize(text, parse_special)
    def token_pieces(self, ids): return self.backend.token_pieces(ids)

    def call(self, op, request, function):
        self.calls += 1
        path = self.output / f'scored-call-{self.calls:02d}.json'
        receipt = {'phase': self.phase, 'op': op, 'request': request, 'status': 'started',
                   'frozen_sha256': self.frozen_sha256}
        save(path, receipt)
        start = time.perf_counter()
        try:
            result = function()
            receipt.update(status='complete', result=result)
            return result
        except BaseException as error:
            receipt.update(status='failed', error=f'{type(error).__name__}: {error}')
            raise
        finally:
            receipt['seconds'] = time.perf_counter() - start
            save(path, receipt)

    def score_tokens(self, input_ids, score_positions):
        return self.call('score', {'input_ids': input_ids, 'score_positions': score_positions},
                         lambda: self.backend.score_tokens(input_ids, score_positions))

    def generate(self, input_ids, max_new_tokens, eos_ids):
        return self.call('generate', {'input_ids': input_ids, 'max_new_tokens': max_new_tokens, 'eos_ids': eos_ids},
                         lambda: self.backend.generate(input_ids, max_new_tokens, eos_ids))


def run(output, gguf, worker, prism_source, tokenizer_path, *, expected_tokenizer_sha256, expected_worker_sha256,
        request_timeout=300., forced_eos=True):
    from transformers import AutoTokenizer
    from openbonsai.eval_backends import PrismBackend, PRISM_RUNTIME_SHA, local_directory_identity
    from openbonsai.evaluation import evaluate_manifest, validate_score_result, validate_generation_result
    output = Path(output).absolute()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Use a new/empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    backend = None
    start = time.perf_counter()
    report = {'format': 'openbonsai.official-bonsai-cpu-preflight.v1', 'status': 'started',
              'passed': False, 'quality_claim': False, 'original_bf16_comparison': False,
              'scope': 'official PTQ1_0 checkpoint plus pinned CPU runtime; authored development inputs',
              'limitations': LIMITATIONS, 'phase': 'input-verification',
              'official_source': {'repository': OFFICIAL_REPOSITORY, 'revision': OFFICIAL_REVISION,
                                  'format': 'PTQ1_0'},
              'expected': {'gguf_sha256': GGUF_SHA256, 'gguf_bytes': GGUF_BYTES,
                           'worker_sha256': str(expected_worker_sha256), 'runtime_sha': PRISM_RUNTIME_SHA,
                           'tokenizer_sha256': str(expected_tokenizer_sha256), 'eos_ids': EXPECTED_EOS},
              'settings': {'context_length': CONTEXT, 'threads': 4, 'gpu_layers': 0,
                           'request_timeout_seconds': (request_timeout if type(request_timeout) in (int, float)
                                                      and math.isfinite(request_timeout) else repr(request_timeout)),
                           'generation_cap': GENERATION_CAP,
                           'forced_eos_enabled': forced_eos if type(forced_eos) is bool else repr(forced_eos)}}
    def progress(phase):
        report['phase'] = phase
        save(output / 'report.json', report)
    try:
        if (type(forced_eos) is not bool or isinstance(request_timeout, bool)
                or not isinstance(request_timeout, (int, float)) or not math.isfinite(request_timeout)
                or not 1 <= request_timeout <= 600):
            raise ValueError('Use a boolean forced_eos and a finite request timeout from 1 to 600 seconds')
        for digest in (expected_tokenizer_sha256, expected_worker_sha256):
            if not isinstance(digest, str) or len(digest) != 64 or any(char not in '0123456789abcdef' for char in digest):
                raise ValueError('Expected worker/tokenizer SHA256 must be a lowercase 64-character digest from a pinned receipt')
        progress('input-verification')
        report['gguf'] = checked_file(gguf, GGUF_SHA256, expected_bytes=GGUF_BYTES)
        report['worker'] = checked_file(worker, expected_worker_sha256)
        identity = local_directory_identity(tokenizer_path)
        if identity['sha256'] != expected_tokenizer_sha256:
            raise ValueError('Tokenizer directory SHA256 disagrees with the pinned input receipt')
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
        if local_directory_identity(tokenizer_path) != identity:
            raise RuntimeError('Tokenizer files changed during loading')
        reference = TokenizerReference(tokenizer, identity)
        report['tokenizer'] = identity
        manifest = build_manifest(reference)
        save(output / 'frozen-inputs.json', manifest)
        report['frozen_sha256'] = manifest['frozen_sha256']
        report['work_budget'] = {'scored_targets_including_reset': manifest['metadata']['scored_targets_including_reset'],
                                 'generation_tokens_maximum': GENERATION_CAP + int(forced_eos),
                                 'model_score_calls_maximum': len(manifest['nll']) + 6,
                                 'model_generation_calls_maximum': 1 + int(forced_eos)}
        progress('starting-pinned-cpu-worker')
        backend = PrismBackend(worker, gguf, context_length=CONTEXT, runtime_source_dir=prism_source,
            stderr_path=output / 'prism.stderr', threads=4, gpu_layers=0, request_timeout=request_timeout,
            expected_executable_sha256=expected_worker_sha256, expected_model_sha256=GGUF_SHA256)
        report['backend'] = backend.metadata()
        observed = report['backend'].get('native_eog_ids')
        report['eog'] = {'expected_evaluation_eos_ids': EXPECTED_EOS, 'observed_native_eog_ids': observed,
                         'same_set': isinstance(observed, list) and set(observed) == set(EXPECTED_EOS),
                         'policy': 'Use the explicitly frozen expected EOS IDs; native EOG is reported, never substituted.'}
        if observed is None:
            raise ValueError('Pinned worker did not report its native EOG IDs')
        # Preserve evidence of raw-normalization behavior instead of requiring
        # the official checkpoint to repeat the earlier converted fixture bug.
        raw = 'e\u0301'
        hf_ids, prism_ids = reference.tokenize(raw), backend.tokenize(raw)
        report['raw_nfc_probe'] = {'raw_utf8_hex': raw.encode().hex(), 'reference_ids': hf_ids,
                                   'native_gguf_ids': prism_ids, 'equal_before_explicit_nfc': hf_ids == prism_ids}
        if reference.tokenize('') != [] or backend.tokenize('') != []:
            raise AssertionError('Empty tokenization inserted a token')
        recorded = RecordedBackend(backend, output, manifest['frozen_sha256'])
        progress('evaluating-frozen-development-diagnostic')
        result = evaluate_manifest(recorded, manifest)
        save(output / 'scored-output.json', result)
        report['diagnostic_aggregate'] = result['aggregate']
        report['generation'] = result['generations'][0]
        report['choice'] = {'selected_id': result['choices'][0]['selected_id'],
                            'correct': result['choices'][0]['correct'], 'not_a_benchmark': True}
        progress('checking-independent-record-reset')
        recorded.phase = 'state-reset-A-other-A'
        reset = manifest['metadata']['state_reset_probe']
        a, positions = reset['a']['input_ids'], reset['a']['score_positions']
        other, other_positions = reset['other']['input_ids'], reset['other']['score_positions']
        before = validate_score_result(recorded.score_tokens(a, positions), a, positions)
        validate_score_result(recorded.score_tokens(other, other_positions), other, other_positions)
        after = validate_score_result(recorded.score_tokens(a, positions), a, positions)
        report['state_reset'] = {'input_ids': a, 'other_input_ids': other, 'score_positions': positions,
                                 'exact_repeat': before == after, 'before': before, 'after': after}
        if before != after:
            raise AssertionError('Record A changed after an unrelated record')
        if forced_eos:
            progress('checking-forced-first-token-eos')
            recorded.phase = 'forced-eos-stop-rule-only'
            first = result['generations'][0]['generated_ids'][0]
            request = {'input_ids': manifest['generations'][0]['input_ids'], 'max_new_tokens': 1, 'eos_ids': [first]}
            save(output / 'forced-eos-input.json', {'purpose': 'Stopping-rule check only; chosen from prior first greedy token', **request})
            stop = validate_generation_result(recorded.generate(**request), max_new_tokens=1, eos_ids=[first])
            report['forced_eos'] = {'request': request, 'result': stop, 'quality_claim': False}
            if stop != {'generated_ids': [first], 'stop_reason': 'eos'}:
                raise AssertionError('Forced first-token EOS was not included and stopped exactly')
        progress('checking-fresh-process-reset')
        backend.close()
        report['first_worker_final'] = backend.metadata()
        backend = None
        backend = PrismBackend(worker, gguf, context_length=CONTEXT, runtime_source_dir=prism_source,
            stderr_path=output / 'prism-fresh.stderr', threads=4, gpu_layers=0, request_timeout=request_timeout,
            expected_executable_sha256=expected_worker_sha256, expected_model_sha256=GGUF_SHA256)
        recorded.backend, recorded.phase = backend, 'fresh-process-A'
        fresh = validate_score_result(recorded.score_tokens(a, positions), a, positions)
        report['fresh_process_state_reset'] = {'exact_repeat': fresh == before, 'result': fresh,
                                               'backend': backend.metadata()}
        if fresh != before:
            raise AssertionError('Record A changed in a fresh worker process')
        report['status'], report['passed'], report['phase'] = 'complete', True, 'complete'
        return report
    except BaseException as error:
        report.update(status='failed', passed=False, error=f'{type(error).__name__}: {error}')
        raise
    finally:
        report['seconds'] = time.perf_counter() - start
        try:
            if backend is not None:
                report['backend_final'] = backend.metadata()
            save(output / 'report.json', report)
        finally:
            if backend is not None:
                backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--gguf', required=True)
    parser.add_argument('--worker', required=True)
    parser.add_argument('--expected-worker-sha256', required=True)
    parser.add_argument('--prism-source', required=True)
    parser.add_argument('--tokenizer-dir', required=True)
    parser.add_argument('--expected-tokenizer-sha256', required=True)
    parser.add_argument('--request-timeout', type=float, default=300.)
    parser.add_argument('--no-forced-eos', action='store_true')
    args = parser.parse_args()
    report = run(args.output, args.gguf, args.worker, args.prism_source, args.tokenizer_dir,
                 expected_tokenizer_sha256=args.expected_tokenizer_sha256,
                 expected_worker_sha256=args.expected_worker_sha256,
                 request_timeout=args.request_timeout, forced_eos=not args.no_forced_eos)
    print(json.dumps({key: report[key] for key in ('passed', 'seconds', 'work_budget', 'diagnostic_aggregate', 'eog')}, indent=2))


if __name__ == '__main__':
    main()
