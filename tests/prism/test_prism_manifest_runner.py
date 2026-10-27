"""The optional Prism manifest runner, with a fake worker backend (no Prism build needed)."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from openbonsai.eval import manifest as evaluation

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('prism_manifest_runner', ROOT / 'tools/prism/evaluate_prism_manifest.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def pieces(ids):
    return [bytes([64 + value]) for value in ids]


def frozen_text(name, ids):
    raw = b''.join(pieces(ids))
    return evaluation.freeze_text(name, raw_text=raw, rendered_text=raw, input_ids=ids, token_pieces=pieces(ids))


def fixture():
    texts, nll, choices, generations = [], [], [], []
    for index, ids in enumerate((list(range(1, 10)), list(range(2, 9)))):
        name = f'nll-{index}'
        texts.append(frozen_text(name, ids))
        nll.append(dict(id=name, text_id=name, token_start=0, token_end=len(ids), input_ids=ids,
                        score_positions=list(range(1, len(ids)))))
    options = []
    for k, label in enumerate((21, 22)):
        name = f'choice-{k}'
        ids = [11, 12, 13, label]
        texts.append(frozen_text(name, ids))
        options.append(dict(id=str(k), text_id=name, input_ids=ids, prefix_length=3, score_positions=[3]))
    choices.append(dict(id='choice', options=options, answer_id='1'))
    texts.append(frozen_text('gen', [31, 32]))
    generations.append(dict(id='gen', text_id='gen', input_ids=[31, 32], max_new_tokens=3, eos_ids=[99]))
    return evaluation.freeze_manifest(source_provenance={'kind': 'synthetic'}, tokenizer_identity={'kind': 'fake'},
                                      texts=texts, nll=nll, choices=choices, generations=generations)


class FakePrism:
    """Speaks the PrismBackend interface; logits favor token (last + 1)."""
    instances = []

    def __init__(self, executable, model, **kwargs):
        self.kwargs, self.closed = kwargs, False
        FakePrism.instances.append(self)

    def metadata(self):
        return {'protocol': evaluation.EVALUATION_PROTOCOL, 'reduction': evaluation.REDUCTION,
                'context_length': self.kwargs['context_length'], 'vocab_size': 128}

    def tokenize(self, text, parse_special=False):
        return [b - 64 for b in text.encode()]

    def token_pieces(self, ids):
        return pieces(ids)

    def score_tokens(self, ids, positions):
        import torch
        rows = torch.zeros(len(positions), 128)
        for k, j in enumerate(positions):
            rows[k, (ids[j - 1] + 1) % 128] = 4.0
        lp = evaluation.reduce_target_logprobs(rows, [ids[j] for j in positions])
        return {'score_positions': positions, 'target_logprobs': lp, 'nll_sum': -sum(lp), 'count': len(lp)}

    def generate(self, ids, maximum, eos):
        return {'generated_ids': [ids[-1] + 1 + k for k in range(maximum)], 'stop_reason': 'length'}

    def close(self):
        self.closed = True


def test_runner_scores_a_manifest_and_writes_the_report(tmp_path):
    manifest = tmp_path / 'm.json'
    manifest.write_text(json.dumps(fixture()))
    args = ['--worker', 'w', '--model', 'm.gguf', '--prism-source', 'src', '--output', str(tmp_path / 'out'),
            '--manifest', str(manifest), '--threads', '2']
    assert runner.main(args, backend_factory=FakePrism) == 0
    fake = FakePrism.instances[-1]
    assert fake.closed and fake.kwargs['context_length'] == 9 and fake.kwargs['threads'] == 2
    assert fake.kwargs['runtime_sha'] == runner.RUNTIME_SHA
    report = json.loads((tmp_path / 'out' / 'report.json').read_text())
    assert report['aggregate']['scored_tokens'] == 14 and report['aggregate']['choice_correct'] == 0
    assert report['generations'][0]['generated_ids'] == [33, 34, 35]
    with pytest.raises(FileExistsError):
        runner.main(args, backend_factory=FakePrism)
    assert runner.main(args + ['--overwrite'], backend_factory=FakePrism) == 0


def test_default_manifest_is_the_pinned_dev_suite():
    args = runner.build_parser().parse_args(['--worker', 'w', '--model', 'm', '--prism-source', 's', '--output', 'o'])
    manifest = runner.load(args)
    assert manifest['frozen_sha256'] == '8021d9cc8f54943a3b816af74cbc41d588173d385f51b73a681e979393ef5fdf'
    assert runner.context_length(manifest) == 513


def test_no_implicit_launch_without_paths():
    result = subprocess.run([sys.executable, str(ROOT / 'tools/prism/evaluate_prism_manifest.py')],
                            capture_output=True, text=True)
    assert result.returncode != 0 and '--worker' in result.stderr
