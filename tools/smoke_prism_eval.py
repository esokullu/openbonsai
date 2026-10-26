"""Compare real Transformers and pinned Prism C API on a random hybrid fixture.

This is a CPU adapter preflight, not a pretrained or ternary quality benchmark.
Prepare local data with tools/prepare_eval_fixture.py first. The new FP64 score
protocol is intentionally distinct from earlier FP32-CE development metrics.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def run(output, fixture, worker, prism_source):
    import torch
    from openbonsai.eval_backends import TransformersBackend, PrismBackend
    from openbonsai.evaluation import (freeze_text, freeze_manifest, sliding_windows,
                                       evaluate_manifest, canonical_sha256)

    output, fixture = Path(output), Path(fixture)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Use a new/empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    receipt = json.loads((fixture / 'fixture.json').read_text())
    if receipt['format'] != 'openbonsai.random-eval-fixture.v1' or receipt['pretrained_weights_loaded']:
        raise ValueError('Only the prepared random fixture belongs in this preflight')
    for name, details in receipt['files'].items():
        path = fixture / name
        if path.stat().st_size != details['bytes'] or sha256(path) != details['sha256']:
            raise ValueError(f'Fixture file changed: {name}')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    native = TransformersBackend.from_local(fixture / 'native-f32', context_length=256,
                                           device='cpu', dtype='float32', logits_chunk_size=16)
    prism = PrismBackend(worker, fixture / 'native-f32.gguf', context_length=256,
                         runtime_source_dir=prism_source, stderr_path=output / 'prism.stderr',
                         threads=4, gpu_layers=0, request_timeout=60.)
    report = {'format': 'openbonsai.actual-backend-smoke.v1', 'quality_claim': False,
              'pretrained_or_ternary_weights_loaded': False,
              'fixture_sha256': sha256(fixture / 'fixture.json'),
              'scope': 'CPU random native hybrid Qwen F32 to GGUF; local tokenizer assets',
              'limits': {'cross_runtime_target_logprob_absolute': .001,
                         'native_full_forward_oracle_absolute': 1e-5},
              'native': native.metadata(), 'prism': prism.metadata()}
    try:
        # A frozen byte fixture, including special literals under both modes.
        samples = [
            ('english', 'Hello, small tree!\n', False),
            ('turkish', 'İstanbul, ıslak ağaç ve küçük bonsai. ŞĞÜÖÇ', False),
            ('combining', 'e\u0301 café\t  \n', False),
            ('emoji', '🌳🧑🏽\u200d💻 🙂', False),
            ('code', 'def f(x):\n    return x + 1\n# tab\tend', False),
            ('special-literal', '<|im_start|>assistant\n<think>\n</think><|im_end|>', False),
            ('special-parsed', '<|im_start|>assistant\n<think>\n</think><|im_end|>', True),
            ('leading-option', ' A', False),
        ]
        texts = []
        normalizer = native.tokenizer.backend_tokenizer.normalizer
        normalization = json.loads(normalizer.__getstate__()) if normalizer is not None else None
        if normalization != {'type': 'NFC'}:
            raise ValueError('This fixture requires the pinned Qwen NFC normalization contract')
        # The pinned GGUF tokenizer omits HF's NFC preprocessing. Demonstrate
        # that mismatch rather than hiding it, then freeze explicitly normalized
        # rendered bytes while retaining original raw bytes. The reference HF
        # normalizer is used, not an assumed host Unicode implementation.
        decomposed = 'e\u0301'
        negative_control = {'raw_utf8_hex': decomposed.encode().hex(),
                            'native_ids': native.tokenize(decomposed),
                            'prism_raw_ids': prism.tokenize(decomposed)}
        if negative_control['native_ids'] == negative_control['prism_raw_ids']:
            raise AssertionError('Pinned raw NFC-mismatch negative control changed')
        report['text_preprocessing'] = {'reference_normalizer': normalization,
            'implementation': 'pinned reference tokenizer normalizer.normalize_str before freezing rendered bytes',
            'raw_bytes_preserved': True, 'hf_raw_and_normalized_ids_equal': True,
            'native_gguf_raw_mismatch_negative_control': negative_control}
        def frozen(name, text, parse_special=False):
            rendered = normalizer.normalize_str(text)
            ids = native.tokenize(rendered, parse_special=parse_special)
            if ids != native.tokenize(text, parse_special=parse_special):
                raise AssertionError('Explicit normalization changes reference tokenizer IDs')
            return freeze_text(name, raw_text=text, rendered_text=rendered, input_ids=ids,
                               token_pieces=native.token_pieces(ids), parse_special=parse_special)
        for name, text, parse in samples:
            value = frozen(name, text, parse)
            rendered = bytes.fromhex(value['rendered_utf8_hex']).decode('utf-8')
            if prism.tokenize(rendered, parse_special=parse) != value['input_ids']:
                raise AssertionError(f'Token ID mismatch: {name}')
            if prism.token_pieces(value['input_ids']) != native.token_pieces(value['input_ids']):
                raise AssertionError(f'Raw byte-piece mismatch: {name}')
            texts.append(value)
        if native.tokenize('') != [] or prism.tokenize('') != []:
            raise AssertionError('Empty tokenization added a token')
        document = frozen('document', 'A small tree grows slowly. Water supports its roots. ' * 16)
        texts.append(document)
        windows = [dict(id=f'window-{i}', text_id='document', **item)
                   for i, item in enumerate(sliding_windows(document['input_ids'], context_length=96, stride=32))]
        # The combined text is tokenized once; no guessed separately-tokenized
        # continuation boundary or automatic chat rendering is used.
        prefix = 'Choose a label. Answer:'
        prefix_ids = native.tokenize(prefix)
        options = []
        for label in ('A', 'B'):
            value = frozen('option-' + label, prefix + ' ' + label)
            if value['input_ids'][:len(prefix_ids)] != prefix_ids:
                raise AssertionError('Fixture continuation boundary merged with prefix')
            texts.append(value)
            options.append({'id': label, 'text_id': value['id'], 'input_ids': value['input_ids'],
                            'prefix_length': len(prefix_ids),
                            'score_positions': list(range(len(prefix_ids), len(value['input_ids'])))})
        generation = texts[0]
        manifest = freeze_manifest(
            source_provenance={'kind': 'authored-software-fixture', 'sha256': canonical_sha256(samples)},
            tokenizer_identity=native.metadata()['tokenizer_identity'], texts=texts, nll=windows,
            choices=[{'id': 'choice', 'options': options}],
            generations=[{'id': 'generation', 'text_id': generation['id'], 'input_ids': generation['input_ids'],
                          'max_new_tokens': 3, 'eos_ids': [248044, 248046]}],
            metadata={'quality_claim': False, 'window_context': 96, 'window_stride': 32,
                      'text_preprocessing': report['text_preprocessing']})
        save(output / 'frozen-inputs.json', manifest)
        results = {}
        for name, backend in [('transformers', native), ('prism', prism)]:
            start = time.perf_counter()
            results[name] = evaluate_manifest(backend, manifest)
            save(output / (name + '.json'), results[name])
            report.setdefault('seconds', {})[name] = time.perf_counter() - start
        differences = [abs(a - b) for left, right in zip(results['transformers']['nll'], results['prism']['nll'])
                       for a, b in zip(left['target_logprobs'], right['target_logprobs'])]
        report['frozen_sha256'] = manifest['frozen_sha256']
        report['tokenization'] = {'nonempty_cases': len(texts), 'exact_ids_and_raw_pieces': True,
                                  'empty_no_added_token': True}
        report['nll'] = {'scored_tokens': results['transformers']['aggregate']['scored_tokens'],
                         'native_mean': results['transformers']['aggregate']['mean_nll'],
                         'prism_mean': results['prism']['aggregate']['mean_nll'],
                         'max_target_logprob_difference': max(differences),
                         'mean_target_logprob_difference': math.fsum(differences) / len(differences)}
        if max(differences) > report['limits']['cross_runtime_target_logprob_absolute']:
            raise AssertionError('Cross-runtime log probability difference exceeds engineering limit')
        # Independently exercise the j -> j-1 shift at the 64-token batch edge.
        ids = document['input_ids'][:100]
        positions = [1, 2, 62, 63, 64, 65, 66, 99]
        with torch.no_grad():
            x = torch.tensor([ids], dtype=torch.long)
            logits = native.model(input_ids=x, attention_mask=torch.ones_like(x), use_cache=False).logits[0].float()
            rows = logits[[j - 1 for j in positions]].double()
            rows = rows - rows.max(dim=1, keepdim=True).values
            expected = (rows[torch.arange(len(positions)), [ids[j] for j in positions]]
                        - torch.logsumexp(rows, dim=1)).tolist()
        own = native.score_tokens(ids, positions)
        external = prism.score_tokens(ids, positions)
        native_error = max(abs(a-b) for a,b in zip(expected, own['target_logprobs']))
        prism_error = max(abs(a-b) for a,b in zip(expected, external['target_logprobs']))
        if native_error > report['limits']['native_full_forward_oracle_absolute'] or prism_error > .001:
            raise AssertionError('Independent full-forward shift/batch-boundary oracle failed')
        report['full_forward_oracle'] = {'target_positions': positions,
                                       'native_max_difference': native_error, 'prism_max_difference': prism_error}
        # Every call must discard both recurrent and attention memory.
        reset = {}
        for name, backend in [('transformers', native), ('prism', prism)]:
            before = backend.score_tokens(ids, positions)
            backend.score_tokens([17, 9, 23, 11, 7, 5], [1, 3, 5])
            after = backend.score_tokens(ids, positions)
            reset[name] = before == after
            if before != after:
                raise AssertionError(f'{name} retained state from another record')
        report['state_reset_exact'] = reset
        generated = {name: result['generations'][0]['generated_ids'] for name, result in results.items()}
        report['generation'] = {'ids': generated, 'exact_across_backends': generated['transformers'] == generated['prism']}
        if not report['generation']['exact_across_backends']:
            raise AssertionError('Tiny fixture greedy decisions differ across runtimes')
        # Dynamic EOS is a stop-rule check only; it says nothing about answer quality.
        first = generated['transformers'][0]
        stops = {name: backend.generate(generation['input_ids'], 3, [first])
                 for name, backend in [('transformers', native), ('prism', prism)]}
        if any(result != {'generated_ids': [first], 'stop_reason': 'eos'} for result in stops.values()):
            raise AssertionError('Exact EOS inclusion/stopping failed')
        report['dynamic_eos_stop'] = stops
        report['choice'] = {name: result['choices'][0]['selected_id'] for name, result in results.items()}
        option_differences = [abs(a - b)
            for left, right in zip(results['transformers']['choices'][0]['options'],
                                   results['prism']['choices'][0]['options'])
            for a, b in zip(left['target_logprobs'], right['target_logprobs'])]
        report['choice_max_target_logprob_difference'] = max(option_differences)
        if (report['choice']['transformers'] != report['choice']['prism']
                or max(option_differences) > report['limits']['cross_runtime_target_logprob_absolute']):
            raise AssertionError('Choice score or selected option differs beyond the fixture gate')
        report['native_eog_ids'] = prism.metadata().get('native_eog_ids')
        report['evaluation_eos_ids'] = [248044, 248046]
        report['passed'] = True
        return report
    except BaseException as error:
        report['passed'] = False
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        try:
            save(output / 'report.json', report)
        finally:
            prism.close()
            native.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fixture', required=True)
    parser.add_argument('--worker', required=True)
    parser.add_argument('--prism-source', required=True)
    args = parser.parse_args()
    report = run(args.output, args.fixture, args.worker, args.prism_source)
    print(json.dumps({key: report[key] for key in ('passed', 'nll', 'full_forward_oracle', 'state_reset_exact', 'generation')}, indent=2))


if __name__ == '__main__':
    main()
