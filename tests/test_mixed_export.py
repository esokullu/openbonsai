import copy
import json

import pytest
import torch

from openbonsai.adapter import tiny_model
from openbonsai.candidates import source_weight_sha256
from openbonsai.mixed_formats import (bf16_candidate, affine_candidate, ternary_candidate,
                                      MixedLinear, MixedEmbedding, candidate_nbytes)
from openbonsai.mixed_export import prepare_mixed_export, save_mixed, load_mixed
from openbonsai.packing import pack_grouped_weight


def fixture(multimodal=False):
    torch.set_num_threads(2)
    model = tiny_model(seed=41, multimodal=multimodal).to(torch.bfloat16)
    prefix = 'model.language_model' if multimodal else 'model'
    generator = torch.Generator().manual_seed(8)
    names = [prefix + '.embed_tokens', 'lm_head', prefix + '.layers.0.mlp.down_proj',
             prefix + '.layers.3.self_attn.o_proj']
    assignment = {}
    for i, name in enumerate(names):
        source = model.get_submodule(name)
        kind = 'embedding' if i == 0 else 'linear'
        if i in (0, 1):
            bits = 4 if i == 0 else 6
            codes = torch.randint(0, 2**bits, source.weight.shape, generator=generator, dtype=torch.uint8)
            scales = torch.full((codes.shape[0], codes.shape[1] // 128), 0.001, dtype=torch.float32)
            assignment[name] = affine_candidate(codes, scales, torch.full(scales.shape, 2**(bits-1), dtype=torch.int32),
                bits=bits, group_size=128, module_kind=kind, source='test-fixture')
        elif i == 2:
            codes = torch.zeros(source.weight.shape, dtype=torch.int8)
            codes[:, ::8] = 1
            state = pack_grouped_weight(codes, torch.full((codes.numel() // 128,), 0.01),
                                       shape=codes.shape, group_size=128)
            assignment[name] = ternary_candidate(state)
        else:
            assignment[name] = bf16_candidate(source.weight)
        assignment[name].metadata['provenance']['source_weight_sha256'] = source_weight_sha256(source.weight)
    return model, assignment


@pytest.mark.parametrize('multimodal', [False, True])
def test_complete_mixed_artifact_exact_cost_and_native_reload(tmp_path, multimodal):
    original, assignment = fixture(multimodal)
    original.generation_config.eos_token_id = [2, 3]
    original.generation_config.max_new_tokens = 79
    tokenizer = tmp_path / 'source-tokenizer'
    tokenizer.mkdir()
    (tokenizer / 'tokenizer.json').write_text('{"test":true}')
    with prepare_mixed_export(original, assignment, metadata={'source_checkpoint': 'random-native-fixture'},
                              tokenizer_dir=tokenizer) as plan:
        total = plan.fixed_bytes + sum(candidate_nbytes(c) for c in assignment.values())
        output = tmp_path / 'model'
        report = save_mixed(plan, assignment, output, budget_bytes=total)
        assert total == sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        assert int(report['storage']['artifact_bytes']) == total
        assert (int(report['storage']['vision_file_bytes']) > 0) == multimodal
        assert int(report['storage']['tokenizer_bytes']) == len('{"test":true}')
        expected = copy.deepcopy(original)
        for name, c in assignment.items():
            cls = MixedEmbedding if c.module_kind == 'embedding' else MixedLinear
            expected.set_submodule(name, cls(c))
        loaded = load_mixed(output)
        assert loaded.generation_config.eos_token_id == [2, 3]
        assert loaded.generation_config.max_new_tokens == 79
        ids = torch.tensor([[4, 5, 6, 7, 8]])
        with torch.no_grad():
            torch.testing.assert_close(loaded(input_ids=ids).logits,
                                       expected(input_ids=ids).logits, rtol=0, atol=0)
        if multimodal:
            # Text logits alone do not establish vision quality; exact retained state does.
            for (n, t), (m, u) in zip(original.model.visual.state_dict().items(), loaded.model.visual.state_dict().items()):
                assert n == m
                torch.testing.assert_close(t, u, rtol=0, atol=0)
        with pytest.raises(ValueError, match='exceeds budget'):
            save_mixed(plan, assignment, tmp_path / 'too-small', budget_bytes=total - 1)
        with pytest.raises(FileExistsError):
            save_mixed(plan, assignment, output, budget_bytes=total)


def test_assignment_changes_candidate_cost_but_never_manifest_overhead(tmp_path):
    original, assignment = fixture()
    fallback = {n: bf16_candidate(original.get_submodule(n).weight, module_kind=c.module_kind,
                                  provenance=c.metadata['provenance'])
                for n, c in assignment.items()}
    with prepare_mixed_export(original, assignment, metadata={'source_checkpoint': 'fixture'}) as plan:
        results = []
        for i, selected in enumerate([assignment, fallback]):
            budget = plan.fixed_bytes + sum(candidate_nbytes(c) for c in selected.values())
            results.append(save_mixed(plan, selected, tmp_path / str(i), budget_bytes=budget))
        assert results[0]['storage']['fixed_bytes'] == results[1]['storage']['fixed_bytes']
        assert results[0]['storage']['manifest_bytes'] == results[1]['storage']['manifest_bytes']
        assert results[0]['storage']['artifact_bytes'] != results[1]['storage']['artifact_bytes']


@pytest.mark.parametrize('corruption', ['payload', 'extra', 'accounting', 'missing-state'])
def test_mixed_artifact_rejects_corruption(tmp_path, corruption):
    original, assignment = fixture()
    output = tmp_path / 'artifact'
    with prepare_mixed_export(original, assignment, metadata={'source_checkpoint': 'fixture'}) as plan:
        report = save_mixed(plan, assignment, output, budget_bytes=10**9)
    if corruption == 'payload':
        target = output / next(iter(report['modules'].values()))['file']
        content = bytearray(target.read_bytes()); content[-1] ^= 1; target.write_bytes(content)
    elif corruption == 'extra':
        (output / 'unaccounted').write_text('x')
    else:
        if corruption == 'accounting':
            report['storage']['artifact_bytes'] = '0' * 20
        else:
            report['retained_tensors'].pop(next(iter(report['retained_tensors'])))
        (output / 'manifest.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):
        load_mixed(output)


def test_export_plan_requires_original_bf16_and_source(tmp_path):
    model, selected = fixture()
    with pytest.raises(ValueError, match='provenance'):
        prepare_mixed_export(model, selected, metadata={})
    model.to(torch.float32)
    with pytest.raises(ValueError, match='BF16'):
        prepare_mixed_export(model, selected, metadata={'source_checkpoint': 'fixture'})


def test_candidate_from_different_original_checkpoint_is_rejected(tmp_path):
    model, selected = fixture()
    next(iter(selected.values())).metadata['provenance']['source_weight_sha256'] = '0' * 64
    with prepare_mixed_export(model, selected, metadata={'source_checkpoint': 'fixture'}) as plan:
        with pytest.raises(ValueError, match='provenance mismatch'):
            save_mixed(plan, selected, tmp_path / 'mismatch', budget_bytes=10**9)


def test_candidate_cannot_silently_drop_or_invent_original_bias(tmp_path):
    model, selected = fixture()
    head = model.lm_head
    head.bias = torch.nn.Parameter(torch.ones(head.out_features, dtype=torch.bfloat16), requires_grad=False)
    with prepare_mixed_export(model, selected, metadata={'source_checkpoint': 'fixture'}) as plan:
        with pytest.raises(ValueError, match='bias differs'):
            save_mixed(plan, selected, tmp_path / 'missing-bias', budget_bytes=10**9)
    model, selected = fixture()
    selected['lm_head'].tensors['bias'] = torch.ones(model.lm_head.out_features, dtype=torch.bfloat16)
    with prepare_mixed_export(model, selected, metadata={'source_checkpoint': 'fixture'}) as plan:
        with pytest.raises(ValueError, match='bias differs'):
            save_mixed(plan, selected, tmp_path / 'invented-bias', budget_bytes=10**9)
