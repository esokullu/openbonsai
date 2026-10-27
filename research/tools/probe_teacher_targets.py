"""Paired teacher-prefix calibration on a random native hybrid Qwen fixture.

Software/target-semantics control only. No pretrained quality, CUDA or paper
reproduction claim. Both arms start from the same BF16 weights, random samples,
rotation signs and factors. Only teacher target provenance differs. Export and
fresh-process reload are checked against the same stored FP16-scale runtime.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def tensor_digest(tensor):
    import torch
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def target_diagnostics(model, rows):
    """Compare both targets on identical hard-prefix inputs, without fitting."""
    import torch
    from openbonsai.adapter import get_text_layers
    from openbonsai.trainer import cache_window, window_mse
    layers = get_text_layers(model)
    result = []
    for index in range(len(layers)):
        conditional = cache_window(model, rows, index, index + 1, teacher_prefix='quantized')
        original = cache_window(model, rows, index, index + 1, teacher_prefix='original')
        numerator, denominator = 0., 0
        for left, right in zip(conditional, original):
            torch.testing.assert_close(left['input'], right['input'], atol=0, rtol=0)
            mask = left['mask'].unsqueeze(-1)
            numerator += float(((left['target'].float() - right['target'].float()).square() * mask).sum())
            denominator += int(mask.sum()) * left['target'].shape[-1]
        result.append({'window': index, 'target_disagreement_mse': numerator / denominator,
                       'hard_mse_to_conditional_teacher': window_mse(layers, conditional, index, index + 1, 'cpu'),
                       'hard_mse_to_original_teacher': window_mse(layers, original, index, index + 1, 'cpu')})
    return result


def reload_check(artifact, inputs, expected, output):
    import torch
    from openbonsai.mixed_export import load_mixed
    torch.set_num_threads(min(4, torch.get_num_threads()))
    model = load_mixed(artifact, device='cpu')
    rows = torch.load(inputs, map_location='cpu', weights_only=True)
    references = torch.load(expected, map_location='cpu', weights_only=True)
    if len(rows) != len(references):
        raise ValueError('Reload reference count differs from input count')
    with torch.no_grad():
        results = [model(**row, use_cache=False).logits for row in rows]
    for actual, reference in zip(results, references):
        torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    save(output, {'exact': True, 'fresh_process': True,
                  'logits_sha256': [tensor_digest(value) for value in results]})


def run(output, seed=5090):
    import torch
    from openbonsai.adapter import tiny_model, iter_ternary_modules, set_quant_enabled, set_progress
    from openbonsai.allocation import causal_lm_loss
    from openbonsai.candidates import source_weight_sha256
    from openbonsai.mixed_export import prepare_mixed_export, save_mixed
    from openbonsai.mixed_formats import MixedLinear, candidate_nbytes
    from openbonsai.rotations import replace_rotated_projections
    from openbonsai.trainer import TrainingConfig, train

    output = Path(output).absolute()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Use a new/empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    original = tiny_model(seed=seed).to(dtype=torch.bfloat16).eval()
    rng = torch.Generator().manual_seed(seed + 1)
    rows = [{'input_ids': torch.randint(3, 240, (1, 16), generator=rng),
             'attention_mask': torch.ones(1, 16, dtype=torch.long)} for _ in range(6)]
    fit, validation = rows[:4], rows[4:]
    if {tuple(r['input_ids'][0].tolist()) for r in fit} & {tuple(r['input_ids'][0].tolist()) for r in validation}:
        raise AssertionError('Fit and validation records overlap')
    torch.save(validation, output / 'validation-inputs.pt')
    source = {name: tensor_digest(value) for name, value in original.state_dict().items()}
    with torch.no_grad():
        baseline = [original(**row, use_cache=False).logits.detach() for row in validation]
    report = {'format': 'openbonsai.teacher-target-probe.v1', 'quality_claim': False,
              'pretrained_weights_loaded': False, 'device': 'cpu', 'dtype': 'bfloat16',
              'scope': 'Random four-block native hybrid Qwen; decoder projections only; endpoints retained',
              'seed': seed, 'rotation_block_size': 64, 'group_size': 128, 'lora_rank': 4,
              'calibration_ids': [r['input_ids'].tolist() for r in fit],
              'validation_ids': [r['input_ids'].tolist() for r in validation],
              'source_tensor_sha256': source, 'conditions': {}, 'passed': False}
    try:
        for prefix in ('quantized', 'original'):
            model = copy.deepcopy(original)
            torch.manual_seed(seed)
            names = replace_rotated_projections(model, group_size=128, block_size=64, seed=seed,
                                                 lora_rank=4, factor_mapping='sigmoid')
            initial = {name: [tensor_digest(p) for p in module.trainable_parameters()]
                       for name, module in iter_ternary_modules(model)}
            if report['conditions']:
                if initial != report['conditions']['quantized']['initial_parameter_sha256']:
                    raise AssertionError('Paired initialization differs')
            set_quant_enabled(model, True)
            set_progress(model, 1.)
            initial_targets = target_diagnostics(model, validation)
            if report['conditions'] and initial_targets != report['conditions']['quantized']['initial_validation_targets']:
                raise AssertionError('Paired initial validation targets differ')
            set_quant_enabled(model, False)
            with torch.no_grad():
                for row, reference in zip(validation, baseline):
                    torch.testing.assert_close(model(**row, use_cache=False).logits, reference, atol=0, rtol=0)
            config = TrainingConfig(epochs=8, batch_size=1, window_size=1, stride=1,
                teacher_prefix=prefix, seed=seed, learning_rate=.005, lora_learning_rate=.001,
                reconstruction_loss='huber', huber_delta=1., gradient_checkpointing=True)
            start = time.perf_counter()
            training = train(model, fit, config)
            training_seconds = time.perf_counter() - start
            set_quant_enabled(model, False)
            with torch.no_grad():
                for row, reference in zip(validation, baseline):
                    torch.testing.assert_close(model(**row, use_cache=False).logits, reference, atol=0, rtol=0)
            set_quant_enabled(model, True)
            set_progress(model, 1.)
            final_targets = target_diagnostics(model, validation)
            with torch.no_grad():
                hard = [model(**row, use_cache=False).logits.detach() for row in validation]
            candidates = {name: module.mixed_candidate(provenance={
                'source_checkpoint': 'random-native-fixture',
                'source_weight_sha256': source_weight_sha256(original.get_submodule(name).weight),
                'teacher_prefix': prefix}) for name, module in iter_ternary_modules(model)}
            with prepare_mixed_export(original, names, metadata={
                    'source_checkpoint': 'random-native-fixture', 'teacher_prefix': prefix,
                    'quality_claim': False, 'training_config': asdict(config)}) as plan:
                budget = plan.fixed_bytes + sum(candidate_nbytes(value) for value in candidates.values())
                manifest = save_mixed(plan, candidates, output / prefix, budget_bytes=budget)
            for name, candidate in candidates.items():
                model.set_submodule(name, MixedLinear(candidate, device='cpu'))
            with torch.no_grad():
                packed = [model(**row, use_cache=False).logits.detach() for row in validation]
            expected = output / f'{prefix}-expected.pt'
            torch.save(packed, expected)
            result_path = output / f'{prefix}-fresh-reload.json'
            command = [sys.executable, str(Path(__file__).absolute()), '--reload', str(output / prefix),
                       '--inputs', str(output / 'validation-inputs.pt'), '--expected', str(expected),
                       '--output', str(result_path)]
            with (output / f'{prefix}-reload.log').open('wb') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120, check=True)
            def drift(values):
                return sum(float((actual.float() - reference.float()).square().mean())
                           for actual, reference in zip(values, baseline)) / len(baseline)
            def nll(values):
                return sum(float(causal_lm_loss(value, row['input_ids'], row['attention_mask']))
                           for value, row in zip(values, validation)) / len(values)
            report['conditions'][prefix] = {
                'initial_parameter_sha256': initial, 'training': training,
                'initial_validation_targets': initial_targets, 'final_validation_targets': final_targets,
                'training_seconds': training_seconds, 'teacher_bypass_exact_before_and_after': True,
                'hard_logit_mse_to_original': drift(hard), 'packed_logit_mse_to_original': drift(packed),
                'original_random_nll': nll(baseline), 'hard_random_nll': nll(hard),
                'packed_random_nll': nll(packed),
                'export_rounding_and_runtime_max_logit_difference': max(float((a-b).abs().max()) for a,b in zip(hard,packed)),
                'fresh_reload': json.loads(result_path.read_text()),
                'actual_artifact_bytes': int(manifest['storage']['artifact_bytes']),
                'runtime': 'floating weight reconstruction; no ternary kernel'}
            save(output / 'report.json', report)
            del model, candidates, hard, packed
        if {name: tensor_digest(value) for name, value in original.state_dict().items()} != source:
            raise AssertionError('Original BF16 fixture changed')
        report.update(passed=True, original_source_unchanged=True)
        return report
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        save(output / 'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=5090)
    parser.add_argument('--reload')
    parser.add_argument('--inputs')
    parser.add_argument('--expected')
    args = parser.parse_args()
    if args.reload:
        if not args.inputs or not args.expected:
            parser.error('--reload requires --inputs and --expected')
        reload_check(args.reload, args.inputs, args.expected, args.output)
    else:
        report = run(args.output, args.seed)
        print(json.dumps({'passed': report['passed'], 'conditions': {
            name: {key: value for key, value in item.items() if key not in ('training', 'initial_parameter_sha256')}
            for name, item in report['conditions'].items()}}, indent=2))


if __name__ == '__main__':
    main()
