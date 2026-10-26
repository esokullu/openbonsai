"""Bounded paired native-Qwen rotation experiment; random fixture, no 9B claim.

Run with: python -m openbonsai.rotation_smoke --output runs/rotation-smoke
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import torch

from .adapter import tiny_model, replace_projections, iter_ternary_modules, set_quant_enabled, set_progress
from .allocation import causal_lm_loss
from .candidates import source_weight_sha256
from .mixed_export import prepare_mixed_export, save_mixed, load_mixed
from .mixed_formats import MixedLinear, candidate_nbytes, ternary_candidate
from .rotations import replace_rotated_projections
from .trainer import train, TrainingConfig


def run(output, *, device='cpu', seed=5090):
    root = Path(output)
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError('Use a new/empty output directory')
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    original = tiny_model(seed=seed).to(device=device, dtype=torch.bfloat16)
    generator = torch.Generator().manual_seed(seed + 1)
    records = [{'input_ids': torch.randint(3, 240, (1, 12), generator=generator).to(device),
                'attention_mask': torch.ones(1, 12, dtype=torch.long, device=device)} for _ in range(4)]
    calibration, validation = records[:2], records[2:]
    assert not ({tuple(x['input_ids'].flatten().tolist()) for x in calibration}
                & {tuple(x['input_ids'].flatten().tolist()) for x in validation})
    with torch.no_grad():
        baseline_logits = [original(**row, use_cache=False).logits.detach() for row in validation]
    config = TrainingConfig(epochs=2, window_size=2, stride=2, seed=seed, learning_rate=.005,
                            lora_learning_rate=.001, gradient_checkpointing=True,
                            reconstruction_loss='huber', huber_delta=1.)
    results = {'kind': 'paired random native Qwen fixture; software/placement validation only',
               'real_9b_weights_loaded': False, 'quality_claim': False, 'device': device,
               'seed': seed, 'rotation_block_size': 64, 'group_size': 128,
               'data': {'fit': [r['input_ids'].tolist() for r in calibration],
                        'search_validation': [r['input_ids'].tolist() for r in validation],
                        'benchmark_answers_used': False},
               'coverage': 'same 25 decoder projections; endpoints BF16 in paired experiment',
               'endpoint_status': 'separate native invariance/gradient/export tests; not silently window-trained',
               'conditions': {}, 'torch': torch.__version__}
    for rotated in (False, True):
        label = 'rotated' if rotated else 'unrotated'
        model = copy.deepcopy(original)
        torch.manual_seed(seed)
        if rotated:
            names = replace_rotated_projections(model, block_size=64, lora_rank=4, seed=seed)
        else:
            names = replace_projections(model, lora_rank=4, factor_mapping='sigmoid')
        set_quant_enabled(model, False)
        with torch.no_grad():
            for row, expected in zip(validation, baseline_logits):
                torch.testing.assert_close(model(**row, use_cache=False).logits, expected, atol=0, rtol=0)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        training = train(model, calibration, config)
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        duration = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated() if device.startswith('cuda') else None
        set_quant_enabled(model, False)
        with torch.no_grad():
            for row, expected in zip(validation, baseline_logits):
                torch.testing.assert_close(model(**row, use_cache=False).logits, expected, atol=0, rtol=0)
        set_quant_enabled(model, True)
        set_progress(model, 1.)
        with torch.no_grad():
            hard_logits = [model(**r, use_cache=False).logits for r in validation]
        selected = {}
        for name, module in iter_ternary_modules(model):
            provenance = {'source_checkpoint': 'random-native-fixture',
                          'source_weight_sha256': source_weight_sha256(original.get_submodule(name).weight)}
            selected[name] = (module.mixed_candidate(provenance=provenance) if rotated else
                              ternary_candidate(module.packed_state(), provenance=provenance))
        with prepare_mixed_export(original, names, metadata={'source_checkpoint': 'random-native-fixture',
                                  'seed': seed, 'comparison': 'paired-rotation-hypothesis'}) as plan:
            budget = plan.fixed_bytes + sum(candidate_nbytes(c) for c in selected.values())
            manifest = save_mixed(plan, selected, root / label, budget_bytes=budget)
        for name, candidate in selected.items():
            model.set_submodule(name, MixedLinear(candidate, device=device))
        restored = load_mixed(root / label, device=device)
        with torch.no_grad():
            packed = [model(**r, use_cache=False).logits for r in validation]
            reloaded = [restored(**r, use_cache=False).logits for r in validation]
            for expected, actual in zip(packed, reloaded):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            # Warmup and synchronize: counts the reference transform/decode/matmul
            # path. This tiny timing is not a speed projection for 9B.
            restored(**validation[0], use_cache=False)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            start_inference = time.perf_counter()
            restored(**validation[0], use_cache=False)
            if device.startswith('cuda'):
                torch.cuda.synchronize()
            inference_seconds = time.perf_counter() - start_inference
        def nll(logits):
            return sum(float(causal_lm_loss(l, r['input_ids'], r['attention_mask'])) for l, r in zip(logits, validation)) / len(validation)
        results['conditions'][label] = {
            'training_seconds': duration, 'peak_allocated_bytes': peak,
            'teacher_bypass_exact_before_and_after': True,
            'reference_reload_exact': True,
            'fp16_export_rounding_max_logit_error': max(float((a-b).abs().max()) for a,b in zip(hard_logits, packed)),
            'original_nll': nll(baseline_logits), 'hard_nll': nll(hard_logits), 'packed_nll': nll(packed),
            'artifact_bytes': int(manifest['storage']['artifact_bytes']),
            'vision_file_bytes': int(manifest['storage']['vision_file_bytes']),
            'reference_forward_seconds': inference_seconds,
            'runtime': 'floating-point reconstruction with optional explicit transforms; no low-bit kernel',
            'training': training}
        del model, restored, selected, hard_logits, packed, reloaded
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
        (root / 'report.json').write_text(json.dumps(results, indent=2, allow_nan=False) + '\n')
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=5090)
    args = parser.parse_args()
    result = run(args.output, device=args.device, seed=args.seed)
    print(json.dumps({k: {p: v for p, v in r.items() if p != 'training'}
                      for k, r in result['conditions'].items()}, indent=2))


if __name__ == '__main__':
    main()
