import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from openbonsai.adapter import tiny_model, set_quant_enabled, set_progress, iter_ternary_modules
from openbonsai.candidates import source_weight_sha256
from openbonsai.mixed_export import prepare_mixed_export, save_mixed, load_mixed
from openbonsai.mixed_formats import MixedLinear, MixedEmbedding
from openbonsai.rotations import (RotatedTernaryLinear, RotatedTernaryEmbedding, fixed_signs,
                                 replace_rotated_projections)


@pytest.mark.parametrize('multimodal', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_native_hybrid_all_endpoints_gates_exact_original_bypass(multimodal, dtype):
    torch.set_num_threads(2)
    model = tiny_model(seed=13, multimodal=multimodal).to(dtype)
    ids = torch.tensor([[3, 5, 7, 9, 11, 13]])
    with torch.no_grad():
        expected = model(input_ids=ids).logits
    names = replace_rotated_projections(model, endpoints=True, aggressive_gates=True,
                                        block_size=64, lora_rank=2)
    assert len(names) == 33
    # All original computations are kept, including recurrence, head, embedding
    # and the small gates. Nonzero adaptation must not leak into the teacher.
    for _, module in iter_ternary_modules(model):
        with torch.no_grad():
            module.lora_B.fill_(0.1)
    with torch.no_grad():
        torch.testing.assert_close(model(input_ids=ids).logits, expected, atol=0, rtol=0)
        set_progress(model, 1)
        set_quant_enabled(model, False)
        torch.testing.assert_close(model(input_ids=ids).logits, expected, atol=0, rtol=0)


def test_native_actual_float_rotation_computation_without_quantization():
    torch.set_num_threads(2)
    model = tiny_model(seed=13).double()
    ids = torch.tensor([[3, 4, 5, 6]])
    with torch.no_grad():
        expected = model(input_ids=ids).logits
    replace_rotated_projections(model, endpoints=True, aggressive_gates=True, block_size=32, lora_rank=0)
    set_progress(model, 1)
    for _, module in iter_ternary_modules(model):
        # adapter.set_progress intentionally visits decoder layers only.
        # Endpoints must explicitly enter the float transform diagnostic too.
        module.set_progress(1)
        module.rotation_only = True
    assert all(module.progress == 1 and module.rotation_only for _, module in iter_ternary_modules(model))
    with torch.no_grad():
        actual = model(input_ids=ids).logits
    # Some native Qwen kernels explicitly accumulate/cast in FP32 even when
    # the fixture is FP64. Standalone Hadamard tests provide the FP64 oracle.
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('embedding', [False, True])
@pytest.mark.parametrize('mapping', ['exp', 'sigmoid'])
def test_rotation_soft_hard_gradients_and_export_reload(tmp_path, embedding, mapping):
    torch.manual_seed(7)
    source = nn.Embedding(32, 16) if embedding else nn.Linear(16, 32, bias=True)
    source.to(torch.bfloat16)
    cls = RotatedTernaryEmbedding if embedding else RotatedTernaryLinear
    module = cls(source, 16, signs=fixed_signs(16, seed=19), block_size=8,
                 lora_rank=4, factor_mapping=mapping)
    inputs = torch.tensor([[3, 9, 3, 15]]) if embedding else torch.randn(1, 4, 16, dtype=torch.bfloat16)
    target = source(inputs).detach()
    optimizer = torch.optim.Adam(module.trainable_parameters(), lr=0.01)
    first_a = module.lora_A.detach().clone()
    for progress in (.2, .5, .75, 1.):
        optimizer.zero_grad(set_to_none=True)
        module.set_progress(progress)
        loss = (module(inputs).float() - target.float()).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.trainable_parameters())
        optimizer.step()
    assert not torch.equal(first_a, module.lora_A)
    assert torch.count_nonzero(module.lora_B)
    candidate = module.mixed_candidate()
    runtime = (MixedEmbedding if embedding else MixedLinear)(candidate)
    from openbonsai.mixed_formats import encode_candidate, decode_candidate
    restored = (MixedEmbedding if embedding else MixedLinear)(decode_candidate(encode_candidate(candidate)))
    with torch.no_grad():
        torch.testing.assert_close(runtime(inputs), restored(inputs), rtol=0, atol=0)
        module.quant_enabled = False
        torch.testing.assert_close(module(inputs), target, rtol=0, atol=0)
    with pytest.raises(ValueError, match='legacy export'):
        module.packed_state()


def test_complete_rotated_native_artifact_parity_and_counted_overhead(tmp_path):
    torch.set_num_threads(2)
    original = tiny_model(seed=21, multimodal=True).to(torch.bfloat16)
    model = copy.deepcopy(original)
    names = replace_rotated_projections(model, endpoints=True, block_size=64, lora_rank=0)
    selected = {name: model.get_submodule(name).mixed_candidate(provenance={
        'source_weight_sha256': source_weight_sha256(original.get_submodule(name).weight)}) for name in names}
    with prepare_mixed_export(original, names, metadata={'source_checkpoint': 'tiny-fixture'}) as plan:
        manifest = save_mixed(plan, selected, tmp_path / 'rotated', budget_bytes=10**9)
    loaded = load_mixed(tmp_path / 'rotated')
    for name, candidate in selected.items():
        model.set_submodule(name, (MixedEmbedding if candidate.module_kind == 'embedding' else MixedLinear)(candidate))
    with torch.no_grad():
        ids = torch.tensor([[3, 4, 5, 6]])
        torch.testing.assert_close(model(input_ids=ids).logits, loaded(input_ids=ids).logits, rtol=0, atol=0)
    assert int(manifest['storage']['artifact_bytes']) == sum(p.stat().st_size for p in (tmp_path / 'rotated').rglob('*') if p.is_file())
    assert int(manifest['storage']['vision_file_bytes']) > 0


def test_nondivisible_rotation_refused_without_partially_mutating_model():
    model = tiny_model()
    with pytest.raises(ValueError, match='does not divide'):
        replace_rotated_projections(model, block_size=4096)
    assert not list(iter_ternary_modules(model))


@pytest.mark.parametrize('endpoints', [False, True])
def test_window_trainer_refuses_untrained_endpoints_and_float_probe(endpoints):
    from openbonsai.trainer import train, TrainingConfig
    model = tiny_model()
    replace_rotated_projections(model, endpoints=endpoints, block_size=64, lora_rank=0)
    if not endpoints:
        next(iter_ternary_modules(model))[1].rotation_only = True
    sample = {'input_ids': torch.tensor([[3,4,5]]), 'attention_mask': torch.ones(1,3,dtype=torch.long)}
    with pytest.raises(ValueError, match='endpoint|diagnostic'):
        train(model, [sample], TrainingConfig(epochs=2))
