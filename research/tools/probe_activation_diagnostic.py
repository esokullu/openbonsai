"""Check read-only activation diagnostics on a random native BF16 Qwen fixture.

No pretrained quality inference. Fits one small rotated decoder, then measures
two predeclared windows on separate random records. Repeated measurements must
be identical and preserve source weights, learned state and incoming flags.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def digest(tensor):
    import torch
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run(output):
    import torch
    from openbonsai.activation_capture import iter_activation_records
    from openbonsai.activation_metrics import ActivationMetrics
    from openbonsai.adapter import tiny_model, iter_ternary_modules
    from openbonsai.rotations import replace_rotated_projections
    from openbonsai.trainer import TrainingConfig, train

    root = Path(output).absolute()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError('Use a new or empty output directory')
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    seed = 5090
    original = tiny_model(seed=seed).to(dtype=torch.bfloat16).eval()
    source_hashes = {n: digest(t) for n, t in original.state_dict().items()}
    model = copy.deepcopy(original)
    torch.manual_seed(seed)
    replace_rotated_projections(model, block_size=64, lora_rank=4, seed=seed)
    rng = torch.Generator().manual_seed(seed + 10)
    rows = []
    for active in (16, 16, 13, 11, 16):
        ids = torch.randint(3, 240, (1, 16), generator=rng)
        mask = torch.arange(16).reshape(1, -1) < active
        ids[~mask] = 0
        rows.append({'input_ids': ids, 'attention_mask': mask.long()})
    fit, search = rows[:2], rows[2:]
    fit_ids = {tuple(x['input_ids'].flatten().tolist()) for x in fit}
    assert not fit_ids.intersection(tuple(x['input_ids'].flatten().tolist()) for x in search)
    training = train(model, fit, TrainingConfig(epochs=2, batch_size=1,
        window_size=1, stride=1, learning_rate=.005, lora_learning_rate=.001,
        seed=seed, reconstruction_loss='huber', gradient_checkpointing=True))
    # A caller can leave flags nonuniform. The diagnostic defines its hard
    # student explicitly and must restore the caller's state before yielding.
    for index, (_, module) in enumerate(iter_ternary_modules(model)):
        module.quant_enabled = bool(index % 2)

    def fingerprint():
        return {'state': {n: digest(t) for n, t in model.state_dict().items()},
                'flags': {n: {'quant_enabled': m.quant_enabled, 'progress': m.progress,
                              'rotation_only': getattr(m, 'rotation_only', False)}
                          for n, m in iter_ternary_modules(model)},
                'training': {n: m.training for n, m in model.named_modules()}}

    before = fingerprint()
    report = {'format': 'openbonsai.activation-diagnostic-probe.v1', 'passed': False,
              'pretrained_weights_loaded': False, 'quality_claim': False,
              'device': 'cpu', 'dtype': 'bfloat16', 'threads': 2, 'seed': seed,
              'scope': 'random four-block hybrid Qwen; rotated decoder; retained original endpoints',
              'training': training, 'fit_token_ids': [r['input_ids'].tolist() for r in fit],
              'search_token_ids': [r['input_ids'].tolist() for r in search],
              'search_masks': [r['attention_mask'].tolist() for r in search],
              'source_sha256': source_hashes, 'windows': []}
    try:
        for start, end in ((0, 1), (3, 4)):
            metrics = ActivationMetrics()
            hashes = []
            for index, values in enumerate(iter_activation_records(model, search, start, end)):
                assert before == fingerprint()
                metrics.update(**values, record_ids=[f'search-{index}'])
                hashes.append({name: digest(value) for name, value in values.items()})
                if start == 0:
                    assert torch.equal(values['xq'], values['x0'])
                    assert torch.equal(values['tq'], values['t0'])
            # Reverse the same records to detect accidental carried KV/recurrent
            # state. This also changes which input is first in a fresh iterator.
            repeat = list(iter_activation_records(model, list(reversed(search)), start, end))
            assert list(reversed(hashes)) == [
                {name: digest(value) for name, value in values.items()} for values in repeat]
            del repeat
            assert before == fingerprint()
            report['windows'].append({'start': start, 'end': end,
                'activation_sha256': hashes, 'reverse_order_replay_exact': True,
                'state_unchanged_after_every_yield': True, 'metrics': metrics.summary()})
        assert source_hashes == {n: digest(t) for n, t in original.state_dict().items()}
        report.update(passed=True, source_unchanged=True, model_state_and_flags_unchanged=True,
                      window_zero_targets_exact=True)
        return report
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        write(root/'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = run(args.output)
    print(json.dumps({'passed': report['passed'], 'quality_claim': False,
                      'windows': [{k: v for k, v in window.items() if k != 'activation_sha256'}
                                  for window in report['windows']]}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
