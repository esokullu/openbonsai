"""Decoded caching must preserve row-GEMM arithmetic and artifact contracts."""
import hashlib

import pytest
import torch

from openbonsai import mixed_formats as mf
from openbonsai.adapter import tiny_model, replace_projections
from openbonsai.mixed_export import prepare_mixed_export, save_mixed, load_mixed
from openbonsai.packing import pack_grouped_weight
from openbonsai.rotations import fixed_signs, replace_rotated_projections, rotation_targets


def candidate(encoding='bitcos', rotated=False):
    generator = torch.Generator().manual_seed(53)
    shape = (129, 128)
    if encoding == 'bf16':
        result = mf.bf16_candidate(torch.randn(shape, generator=generator).bfloat16(),
                                   torch.randn(shape[0], generator=generator).bfloat16())
    elif encoding == 'affine':
        result = mf.affine_candidate(torch.randint(0, 16, shape, generator=generator),
            torch.rand((129, 4), generator=generator) + .01, torch.full((129, 4), 7),
            bits=4, group_size=32, bias=torch.randn(129, generator=generator))
    else:
        codes = torch.randint(-1, 2, shape, generator=generator, dtype=torch.int8)
        scales = torch.rand(129 * 4, generator=generator) + .01
        state = pack_grouped_weight(codes, scales, shape=shape, group_size=32,
                                    bias=torch.randn(129, generator=generator))
        result = mf.ternary_candidate(state, packing=encoding)
    return (mf.with_rotation(result, fixed_signs(128, seed=5090), 64) if rotated else result)


@pytest.mark.parametrize('encoding', ['2bit', 'bitcos', 'affine', 'bf16'])
@pytest.mark.parametrize('rotated', [False, True])
def test_cache_exact_weights_outputs_row_calls_and_no_serialization_change(monkeypatch, encoding, rotated):
    torch.set_num_threads(4)
    runtime = mf.MixedLinear(candidate(encoding, rotated), row_chunk_size=64)
    state_keys = set(runtime.state_dict())
    encoded = mf.encode_candidate(runtime._candidate())
    inputs = [torch.randn((1, length, 128), dtype=torch.bfloat16) for length in (512, 17, 1)]
    calls = []
    linear = mf.F.linear
    def observed(x, weight, bias):
        calls.append((tuple(x.shape), tuple(weight.shape), weight.dtype,
                      None if bias is None else (tuple(bias.shape), bias.dtype)))
        return linear(x, weight, bias)
    monkeypatch.setattr(mf.F, 'linear', observed)
    with torch.no_grad():
        expected = [runtime(x) for x in inputs]
        expected_calls = list(calls)
        expected_weight = runtime.weight
        assert runtime.prepare_decoded_weight_cache() is runtime
        assert torch.equal(runtime._decoded_weight_cache, expected_weight)
        assert runtime.decoded_weight_cache_bytes == 129 * 128 * 2
        first_pointer = runtime._decoded_weight_cache.data_ptr()
        runtime.prepare_decoded_weight_cache()
        assert runtime._decoded_weight_cache.data_ptr() == first_pointer
        calls.clear()
        def forbidden(*_args, **_kwargs):
            raise AssertionError('A warm forward must not decode rows')
        monkeypatch.setattr(mf, '_rows', forbidden)
        for x, wanted in zip(inputs, expected):
            assert torch.equal(runtime(x), wanted)
        assert calls == expected_calls
        assert [row[1][0] for row in calls] == [64, 64, 1] * 3
    assert set(runtime.state_dict()) == state_keys
    assert '_decoded_weight_cache' not in runtime.state_dict()
    assert mf.encode_candidate(runtime._candidate()) == encoded
    assert runtime.clear_decoded_weight_cache() is runtime
    assert runtime.decoded_weight_cache_bytes == 0


def test_cache_falls_back_for_input_dtype_and_invalidates_on_storage_updates():
    runtime = mf.MixedLinear(candidate(), row_chunk_size=64)
    x = torch.randn(2, 128)
    with torch.no_grad():
        expected = runtime(x)
        runtime.prepare_decoded_weight_cache()
        assert torch.equal(runtime(x), expected)  # FP32 decode, not BF16 cache cast to FP32.
        runtime.payload_scales.mul_(1.25)
        uncached = mf.MixedLinear(runtime._candidate(), row_chunk_size=64)
        assert torch.equal(runtime(x.bfloat16()), uncached(x.bfloat16()))
        assert runtime.decoded_weight_cache_bytes == 0
        runtime.prepare_decoded_weight_cache()
        state = runtime.state_dict()
        runtime.load_state_dict(state)
        assert runtime.decoded_weight_cache_bytes == 0
        runtime.prepare_decoded_weight_cache()
        runtime.to(dtype=torch.float32)
        assert runtime.decoded_weight_cache_bytes == 0
        with pytest.raises(ValueError, match='BF16'):
            runtime.prepare_decoded_weight_cache()


def test_cache_accepts_immutable_inference_created_payloads_without_version_counters():
    with torch.inference_mode():
        runtime = mf.MixedLinear(candidate(rotated=True))
        assert runtime.payload_scales.is_inference()
        x = torch.randn(1, 17, 128, dtype=torch.bfloat16)
        expected = runtime(x)
        runtime.prepare_decoded_weight_cache()
        assert torch.equal(runtime(x), expected)
        assert runtime.decoded_weight_cache_bytes > 0
        runtime.to('cpu')
        assert runtime.decoded_weight_cache_bytes == 0


@pytest.mark.parametrize('rotated', [False, True])
def test_native_cache_manifest_policy_cold_override_and_reload_are_exact(tmp_path, rotated):
    torch.set_num_threads(4)
    model = tiny_model(seed=83, multimodal=True).bfloat16()
    names = rotation_targets(model)
    with prepare_mixed_export(model, names, metadata={'source_checkpoint': 'tiny-original-BF16'},
                              row_chunk_size=64, cache_decoded_weights=True) as plan:
        if rotated:
            replace_rotated_projections(model, block_size=64, lora_rank=3)
        else:
            replace_projections(model, lora_rank=3, factor_mapping='sigmoid')
        assignment = {}
        for name in names:
            module = model.get_submodule(name)
            provenance = {'source_weight_sha256': plan.manifest['modules'][name]['source_weight_sha256']}
            assignment[name] = (module.mixed_candidate(provenance=provenance) if rotated else
                                mf.ternary_candidate(module.packed_state(), provenance=provenance))
        artifact = tmp_path / 'packed'
        budget = plan.fixed_bytes + sum(map(mf.candidate_nbytes, assignment.values()))
        manifest = save_mixed(plan, assignment, artifact, budget_bytes=budget)
    assert manifest['runtime_options']['cache_decoded_weights'] is True
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in artifact.iterdir() if p.is_file()}
    cold = load_mixed(artifact, cache_decoded_weights=False)
    cached = load_mixed(artifact)
    assert all(cold.get_submodule(name).decoded_weight_cache_bytes == 0 for name in names)
    assert all(cached.get_submodule(name).decoded_weight_cache_bytes > 0 for name in names)
    assert not any('_decoded_weight_cache' in name for name in cached.state_dict())
    with torch.no_grad():
        for length in (17, 1):
            ids = torch.randint(3, 200, (1, length))
            expected = cold(input_ids=ids, use_cache=False).logits
            assert torch.equal(cached(input_ids=ids, use_cache=False).logits, expected)
        for name in names:
            module = cold.get_submodule(name)
            module.prepare_decoded_weight_cache()
            assert torch.equal(module._decoded_weight_cache, cached.get_submodule(name)._decoded_weight_cache)
        assert torch.equal(cold(input_ids=ids, use_cache=False).logits, expected)
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in artifact.iterdir() if p.is_file()}


@pytest.mark.parametrize('value', [0, 1, 'true'])
def test_plan_rejects_nonboolean_cache_policy_before_allocating(value):
    with pytest.raises(ValueError, match='cache_decoded_weights'):
        prepare_mixed_export(None, [], metadata={}, cache_decoded_weights=value)
