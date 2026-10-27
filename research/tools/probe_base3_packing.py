"""Same ternary values, four codecs, complete native-model fresh-process reload.

Random BF16 hybrid Qwen fixture only. Decoder factors are fitted once; endpoint
values are uncalibrated random-fixture ternary values. This checks broad codec
coverage and exact preservation, not pretrained quality or inference speed.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(tensor):
    import torch
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def reload_check(artifact, inputs, reference, output):
    import torch
    from openbonsai.mixed_export import load_mixed
    torch.set_num_threads(2)
    model = load_mixed(artifact, device='cpu')
    samples = torch.load(inputs, weights_only=True)
    expected = torch.load(reference, weights_only=True)
    with torch.no_grad():
        actual = [model(**s, use_cache=False).logits for s in samples]
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    save(output, {'fresh_process': True, 'exact_logits': True,
                  'logits_sha256': [digest(value) for value in actual]})


def run(output):
    import torch
    from openbonsai.adapter import tiny_model, iter_ternary_modules, set_quant_enabled
    from openbonsai.candidates import source_weight_sha256
    from openbonsai.mixed_export import prepare_mixed_export, save_mixed, load_mixed
    from openbonsai.mixed_formats import candidate_nbytes, candidate_size_breakdown, repack_ternary_candidate
    from openbonsai.rotations import (replace_rotated_projections, RotatedTernaryEmbedding,
                                     RotatedTernaryLinear, fixed_signs)
    from openbonsai.trainer import TrainingConfig, train

    root = Path(output).absolute()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError('Use a new/empty output directory')
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    seed = 5090
    source = tiny_model(seed=seed).to(dtype=torch.bfloat16).eval()
    source_hashes = {name: digest(value) for name, value in source.state_dict().items()}
    model = copy.deepcopy(source)
    torch.manual_seed(seed)
    names = replace_rotated_projections(model, block_size=64, lora_rank=4, seed=seed)
    rng = torch.Generator().manual_seed(seed + 1)
    samples = [{'input_ids': torch.randint(3, 240, (1, 16), generator=rng),
                'attention_mask': torch.ones(1, 16, dtype=torch.long)} for _ in range(4)]
    training = train(model, samples[:2], TrainingConfig(epochs=2, batch_size=1,
        window_size=2, stride=2, seed=seed, learning_rate=.005, lora_learning_rate=.001,
        reconstruction_loss='huber', gradient_checkpointing=True))
    # Endpoints are included to check storage/runtime coverage only; decoder
    # window calibration does not silently train them.
    for name, cls in (('model.embed_tokens', RotatedTernaryEmbedding), ('lm_head', RotatedTernaryLinear)):
        native = model.get_submodule(name)
        module = cls(native, group_size=128, block_size=64, lora_rank=4,
                     signs=fixed_signs(native.weight.shape[1], seed=seed, name=name),
                     factor_mapping='sigmoid')
        module.set_progress(1.)
        model.set_submodule(name, module)
        names.append(name)
    set_quant_enabled(model, False)
    with torch.no_grad():
        for sample in samples[2:]:
            torch.testing.assert_close(model(**sample, use_cache=False).logits,
                                       source(**sample, use_cache=False).logits, atol=0, rtol=0)
    base = {name: module.mixed_candidate(packing='2bit', provenance={
        'source_checkpoint': 'random-native-fixture',
        'source_weight_sha256': source_weight_sha256(source.get_submodule(name).weight)})
        for name, module in iter_ternary_modules(model)}
    inputs, reference = root/'inputs.pt', root/'reference.pt'
    torch.save(samples[2:], inputs)
    report = {'format': 'openbonsai.base3-native-probe.v1', 'passed': False,
              'quality_claim': False, 'pretrained_weights_loaded': False, 'device': 'cpu',
              'seed': seed, 'training': training, 'source_sha256': source_hashes,
              'coverage': '25 rotated decoder projections plus rotated embedding/head; endpoints uncalibrated',
              'original_teacher_bypass_exact': True, 'codecs': {}}
    try:
        with prepare_mixed_export(source, names, metadata={
                'source_checkpoint': 'random-native-fixture', 'quality_claim': False,
                'purpose': 'same-values codec parity including endpoints'}) as plan:
            for codec in ('2bit', 'bitcos', 'base3', 'auto'):
                selected = {name: repack_ternary_candidate(candidate, packing=codec)
                            for name, candidate in base.items()}
                for name, candidate in selected.items():
                    restored = repack_ternary_candidate(candidate, packing='2bit')
                    assert restored.metadata == base[name].metadata
                    for key, value in base[name].tensors.items():
                        assert torch.equal(restored.tensors[key], value)
                        assert restored.tensors[key].dtype == value.dtype
                budget = plan.fixed_bytes + sum(candidate_nbytes(c) for c in selected.values())
                manifest = save_mixed(plan, selected, root/codec, budget_bytes=budget)
                actual = sum(p.stat().st_size for p in (root/codec).rglob('*') if p.is_file())
                assert actual == budget == int(manifest['storage']['artifact_bytes'])
                if codec == '2bit':
                    loaded = load_mixed(root/codec)
                    with torch.no_grad():
                        torch.save([loaded(**sample, use_cache=False).logits for sample in samples[2:]], reference)
                    del loaded
                receipt = root/f'{codec}-reload.json'
                command = [sys.executable, str(Path(__file__).absolute()), '--reload', str(root/codec),
                           '--inputs', str(inputs), '--reference', str(reference), '--output', str(receipt)]
                with (root/f'{codec}-reload.log').open('wb') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120, check=True)
                report['codecs'][codec] = {'actual_artifact_bytes': actual,
                    'candidate_bytes': int(manifest['storage']['candidate_bytes']),
                    'fixed_bytes': plan.fixed_bytes, 'exact_codes_and_metadata': True,
                    'candidates': {name: {'encoding': c.encoding, **candidate_size_breakdown(c)}
                                   for name, c in selected.items()},
                    'reload': json.loads(receipt.read_text())}
                save(root/'report.json', report)
        assert source_hashes == {name: digest(value) for name, value in source.state_dict().items()}
        report.update(passed=True, original_source_unchanged=True)
        return report
    finally:
        save(root/'report.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reload')
    parser.add_argument('--inputs')
    parser.add_argument('--reference')
    args = parser.parse_args()
    if args.reload:
        if not args.inputs or not args.reference:
            parser.error('--reload requires --inputs and --reference')
        reload_check(args.reload, args.inputs, args.reference, args.output)
    else:
        result = run(args.output)
        print(json.dumps({'passed': result['passed'], 'codecs': {name: {
            key: value for key, value in item.items() if key != 'candidates'}
            for name, item in result['codecs'].items()}}, indent=2))


if __name__ == '__main__':
    main()
