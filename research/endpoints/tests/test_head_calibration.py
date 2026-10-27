import hashlib
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from openbonsai.adapter import tiny_model, replace_projections
from openbonsai.endpoint_quantizer import BoundedTernaryHead
from openbonsai.head_calibration import (
    HeadTrainingConfig, cache_original_head_inputs, load_head_input_cache,
    chunked_full_vocab_kl, evaluate_head_from_cache, fit_head_from_cache,
    _trainable_state_sha256,
)
from openbonsai.rotations import fixed_signs, RotatedTernaryLinear
from openbonsai.quantizer import TernaryLinear


SOURCE = {'source_checkpoint': 'native-random-fixture', 'source_revision': 'fixture-seed-31'}


def model_fixture(multimodal=False):
    torch.set_num_threads(2)
    return tiny_model(seed=31, multimodal=multimodal).to(torch.bfloat16)


def records(start=3):
    return [{'input_ids': torch.tensor([[start, start+1, start+2, start+3, 2]]),
             'attention_mask': torch.ones(1, 5, dtype=torch.long)},
            {'input_ids': torch.tensor([[start+4, start+5, 2, 0, 0]]),
             'attention_mask': torch.tensor([[1, 1, 1, 0, 0]])}]


@pytest.mark.parametrize('multimodal', [False, True])
def test_cache_is_original_final_norm_input_masked_shifted_and_bounded(tmp_path, monkeypatch, multimodal):
    model = model_fixture(multimodal)
    rows = [{'input_ids': torch.tensor([[0, 4, 5, 2, 0], [7, 8, 9, 2, 0]]),
             'attention_mask': torch.tensor([[0, 1, 1, 1, 0], [1, 1, 1, 1, 0]]),
             'labels': torch.tensor([[-100, 4, 5, 2, -100], [7, -100, 9, 2, -100]])}]
    with torch.no_grad():
        expected = model.model(input_ids=rows[0]['input_ids'], attention_mask=rows[0]['attention_mask'],
                               use_cache=False).last_hidden_state
    model.train()
    model.lm_head.eval()
    modes = [(module, module.training) for module in model.modules()]
    source = model.lm_head.weight.detach().clone()
    def no_logits(*args, **kwargs):
        raise AssertionError('cache must stop before vocabulary logits')
    monkeypatch.setattr(model.lm_head, 'forward', no_logits)
    cache = cache_original_head_inputs(model, rows, tmp_path/'cache', SOURCE, shard_positions=2)
    parts = list(cache.iter_record(0, token_microbatch=1))
    coordinates = torch.cat([part['positions'] for part in parts])
    assert coordinates.tolist() == [[0, 1], [0, 2], [1, 1], [1, 2]]
    assert torch.cat([part['labels'] for part in parts]).tolist() == [5, 2, 9, 2]
    torch.testing.assert_close(torch.cat([part['hidden'] for part in parts]),
                               expected[coordinates[:, 0], coordinates[:, 1]], atol=0, rtol=0)
    assert cache.positions == 4 and len(cache) == 1
    assert len(cache.manifest['files']) == 2
    assert all(part['hidden'].shape == (1, 128) for part in parts)
    assert all(module.training == was for module, was in modes)
    assert not model.lm_head._forward_pre_hooks
    assert torch.equal(model.lm_head.weight, source)
    assert cache.manifest['config_sha256'] and cache.manifest['records'][0]['input_sha256']
    with pytest.raises(FileExistsError):
        cache_original_head_inputs(model, rows, cache.directory, SOURCE)


def test_cache_max_positions_preserves_record_and_shard_accounting(tmp_path):
    cache = cache_original_head_inputs(model_fixture(), records(), tmp_path/'limited', SOURCE,
                                       max_positions=5, shard_positions=2)
    assert cache.positions == 5
    assert [r['positions'] for r in cache.manifest['records']] == [4, 1]
    assert [d['positions'] for d in cache.manifest['files'].values()] == [2, 2, 1]


def test_native_bf16_teacher_retains_fp32_recurrent_parameters(tmp_path):
    model = model_fixture()
    recurrent = model.model.layers[0].linear_attn
    recurrent.A_log.data = recurrent.A_log.data.float()
    recurrent.dt_bias.data = recurrent.dt_bias.data.float()
    before = {name: value.detach().clone() for name,value in model.state_dict().items()}
    row = records()[0]
    with torch.no_grad():
        expected = model.model(**row, use_cache=False).last_hidden_state[:, :-1].reshape(-1, 128)
    cache = cache_original_head_inputs(model, [row], tmp_path/'mixed-precision-original', SOURCE)
    actual = torch.cat([part['hidden'] for part in cache.iter_record(0, token_microbatch=2)])
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    counts = cache.manifest['source_parameter_dtypes']
    assert counts['float32']['tensors'] == 2
    assert counts['float32']['numel'] == recurrent.A_log.numel() + recurrent.dt_bias.numel()
    assert counts['bfloat16']['tensors'] > 0
    for name,value in model.state_dict().items():
        assert value.dtype == before[name].dtype and torch.equal(value, before[name])
    recurrent.dt_bias.data = recurrent.dt_bias.data.half()
    with pytest.raises(ValueError, match='BF16/FP32'):
        cache_original_head_inputs(model, [row], tmp_path/'unsupported-fp16', SOURCE)


@pytest.mark.parametrize('bad', ['padding', 'label', 'nonfinite', 'fail-forward'])
def test_cache_failure_restores_hooks_modes_and_does_not_publish(tmp_path, monkeypatch, bad):
    model = model_fixture()
    model.train()
    rows = records()
    if bad == 'padding':
        for row in rows:
            row['attention_mask'].zero_()
    elif bad == 'label':
        rows[0]['labels'] = torch.zeros_like(rows[0]['input_ids'])
    elif bad == 'nonfinite':
        def poison(_module, args):
            return (torch.full_like(args[0], float('nan')),)
        handle = model.lm_head.register_forward_pre_hook(poison)
    else:
        def fail(*args, **kwargs):
            raise RuntimeError('intentional capture failure')
        monkeypatch.setattr(model, 'forward', fail)
    hooks = len(model.lm_head._forward_pre_hooks)
    with pytest.raises((ValueError, RuntimeError)):
        cache_original_head_inputs(model, rows, tmp_path/'bad', SOURCE)
    assert len(model.lm_head._forward_pre_hooks) == hooks
    assert model.training
    assert not (tmp_path/'bad').exists()
    if bad == 'nonfinite':
        handle.remove()


def test_cache_rejects_nonoriginal_float_tied_and_missing_source(tmp_path):
    model = model_fixture()
    for missing in ({}, {'source_checkpoint': 'foo'}, {'source_checkpoint': 'foo', 'source_revision': 'main'}):
        with pytest.raises(ValueError):
            cache_original_head_inputs(model, records(), tmp_path/'bad', missing)
    model.float()
    with pytest.raises(ValueError, match='BF16'):
        cache_original_head_inputs(model, records(), tmp_path/'bad', SOURCE)
    model.to(torch.bfloat16)
    model.lm_head.weight = model.get_input_embeddings().weight
    with pytest.raises(ValueError, match='Tied'):
        cache_original_head_inputs(model, records(), tmp_path/'bad', SOURCE)
    model = model_fixture()
    replace_projections(model, lora_rank=0)
    with pytest.raises(ValueError, match='original model'):
        cache_original_head_inputs(model, records(), tmp_path/'bad', SOURCE)


def test_cache_detects_corruption_and_unaccounted_files(tmp_path):
    cache = cache_original_head_inputs(model_fixture(), records(), tmp_path/'cache', SOURCE)
    (cache.directory/'unaccounted').write_text('x')
    with pytest.raises(ValueError, match='Unaccounted'):
        load_head_input_cache(cache.directory)
    (cache.directory/'unaccounted').unlink()
    shard = cache.directory / next(iter(cache.manifest['files']))
    content = bytearray(shard.read_bytes()); content[-1] ^= 1; shard.write_bytes(content)
    with pytest.raises(ValueError, match='Checksum'):
        load_head_input_cache(cache.directory)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('chunk', [1, 4, 8, 19, 32])
@pytest.mark.parametrize('magnitude', [.2, 25.])
def test_global_kl_and_all_parameter_gradients_match_dense(dtype, chunk, magnitude):
    torch.manual_seed(5090)
    positions, width, vocabulary, rank = 7, 8, 19, 3
    hidden = torch.randn(positions, width, dtype=dtype)
    teacher_weight = torch.randn(vocabulary, width, dtype=dtype) * magnitude
    source = teacher_weight + torch.randn_like(teacher_weight) * .05
    initial = [torch.randn(rank, width, dtype=dtype)*.1, torch.randn(vocabulary, rank, dtype=dtype)*.1,
               torch.randn(vocabulary, 1, dtype=dtype)*.1, torch.randn(vocabulary, dtype=dtype)*.1]
    dense = [p.clone().requires_grad_() for p in initial]
    bounded = [p.clone().requires_grad_() for p in initial]
    mask = torch.tensor([True, False, True, True, False, True, True])
    def logits(params, start, end):
        a, b, scale, bias = params
        weight = (source[start:end] + b[start:end] @ a / rank) * scale[start:end].sigmoid() * 2
        return hidden @ weight.T + bias[start:end]
    teacher = lambda start, end: hidden @ teacher_weight[start:end].T
    target = teacher(0, vocabulary).double().log_softmax(-1)
    expected = ((target.exp() * (target - logits(dense, 0, vocabulary).double().log_softmax(-1))).sum(-1) * mask).sum()/mask.sum()
    expected.backward()
    actual = chunked_full_vocab_kl(lambda start, end: logits(bounded, start, end), teacher,
                                  vocabulary=vocabulary, row_chunk=chunk, mask=mask)
    tolerance = 2e-5 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(actual, expected.detach(), atol=tolerance, rtol=tolerance)
    for a, b in zip(dense, bounded):
        torch.testing.assert_close(a.grad, b.grad, atol=tolerance, rtol=tolerance)


def test_token_microbatch_global_denominator_matches_dense_and_no_backward_mode():
    torch.manual_seed(7)
    hidden = torch.randn(7, 4, dtype=torch.float64)
    teacher_weight = torch.randn(9, 4, dtype=torch.float64)
    full = torch.randn(9, 4, dtype=torch.float64, requires_grad=True)
    split = full.detach().clone().requires_grad_()
    target = (hidden @ teacher_weight.T).log_softmax(-1)
    expected = (target.exp()*(target-(hidden @ full.T).log_softmax(-1))).sum()/7
    expected.backward()
    actual = 0.
    for value in hidden.split(3):
        actual += chunked_full_vocab_kl(lambda a,b: value @ split[a:b].T,
                                       lambda a,b: value @ teacher_weight[a:b].T,
                                       vocabulary=9, row_chunk=4, mask=torch.ones(len(value), dtype=torch.bool), denominator=7)
    torch.testing.assert_close(actual, expected.detach(), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(split.grad, full.grad, atol=1e-12, rtol=1e-12)
    grad = split.grad.clone()
    evaluated = chunked_full_vocab_kl(lambda a,b: hidden @ split[a:b].T,
                                     lambda a,b: hidden @ teacher_weight[a:b].T,
                                     vocabulary=9, row_chunk=4, mask=torch.ones(7, dtype=torch.bool), backward=False)
    assert torch.equal(grad, split.grad)
    torch.testing.assert_close(evaluated, expected.detach(), atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError, match='valid KL'):
        chunked_full_vocab_kl(None, None, vocabulary=9, row_chunk=4, mask=torch.zeros(7, dtype=torch.bool))


@pytest.mark.parametrize('rotated', [False, True])
@pytest.mark.parametrize('hard', [False, True])
def test_full_vocab_kl_through_actual_soft_and_saved_hard_jacobian(rotated, hard):
    torch.manual_seed(28)
    source = torch.nn.Linear(8, 19, bias=True, dtype=torch.bfloat16)
    signs = fixed_signs(8, seed=3) if rotated else None
    bounded = BoundedTernaryHead(source, group_size=8, row_chunk_size=4, lora_rank=3,
                                 signs=signs, block_size=4, factor_mapping='sigmoid')
    dense = (RotatedTernaryLinear(source, group_size=8, lora_rank=3, signs=signs, block_size=4,
                                  factor_mapping='sigmoid') if rotated else
             TernaryLinear(source, group_size=8, lora_rank=3, factor_mapping='sigmoid'))
    with torch.no_grad():
        bounded.lora_B.normal_(std=.1)
        for name, parameter in dense.named_parameters():
            parameter.copy_(dict(bounded.named_parameters())[name])
    hidden = torch.randn(5, 8, dtype=torch.bfloat16)
    mask = torch.tensor([True, True, False, True, True])
    dense.set_progress(.4)
    bounded.begin_step(.4)
    if hard:
        # Capture the actual dense soft-forward point, then move current
        # parameters so using a current-point surrogate cannot pass this test.
        dense(hidden)
        with torch.no_grad():
            for name, parameter in dense.named_parameters():
                delta = torch.randn_like(parameter) * .01
                parameter.add_(delta)
                dict(bounded.named_parameters())[name].add_(delta)
        dense.set_progress(1.)
        bounded.begin_step(1.)
    with torch.no_grad():
        target = source(hidden).double().log_softmax(-1)
    expected = ((target.exp() * (target - dense(hidden).double().log_softmax(-1))).sum(-1) * mask).sum()/mask.sum()
    expected.backward()
    actual = chunked_full_vocab_kl(lambda a,b: bounded.forward_rows(hidden,a,b),
                                  lambda a,b: bounded.teacher_rows(hidden,a,b),
                                  vocabulary=19, row_chunk=4, mask=mask)
    torch.testing.assert_close(actual, expected.detach(), atol=1e-6, rtol=1e-5)
    for name, parameter in dense.named_parameters():
        torch.testing.assert_close(dict(bounded.named_parameters())[name].grad, parameter.grad,
                                   atol=2e-6, rtol=2e-4)


@pytest.mark.parametrize('rotated', [False, True])
def test_fit_head_real_cache_global_steps_snapshots_eval_and_source_immutability(tmp_path, monkeypatch, rotated):
    model = model_fixture()
    cache = cache_original_head_inputs(model, records(), tmp_path/'fit', SOURCE, shard_positions=1)
    validation = cache_original_head_inputs(model, records(40), tmp_path/'validation', SOURCE, role='search_validation')
    head = BoundedTernaryHead(model.lm_head, row_chunk_size=31, lora_rank=4,
                              signs=fixed_signs(128, seed=19) if rotated else None, block_size=64)
    weight = model.lm_head.weight.detach().clone()
    calls, optimizer_steps, callback_events = [], [], []
    begin, step = head.begin_step, torch.optim.AdamW.step
    def counted_begin(progress):
        calls.append(progress)
        return begin(progress)
    def counted_step(optimizer, *args, **kwargs):
        optimizer_steps.append(1)
        return step(optimizer, *args, **kwargs)
    monkeypatch.setattr(head, 'begin_step', counted_begin)
    monkeypatch.setattr(torch.optim.AdamW, 'step', counted_step)
    result = fit_head_from_cache(head, cache,
        HeadTrainingConfig(epochs=2, batch_size=2, token_microbatch=2, vocabulary_chunk=31, learning_rate=.005),
        source_provenance=SOURCE, validation_cache=validation, progress_callback=callback_events.append)
    assert calls == [.5, 1.] and len(optimizer_steps) == 2
    assert result.report['head_configuration']['group_size'] == 128
    assert result.report['head_configuration']['row_chunk_size'] == 31
    assert result.report['head_configuration']['factor_mapping'] == 'sigmoid'
    assert result.report['head_configuration']['lora_rank'] == 4
    assert result.report['head_configuration']['hard_phase'] == 'frozen_jacobian'
    assert result.report['head_configuration']['soft_fraction'] == .8
    assert result.report['head_configuration']['max_sharpness'] == 30
    rotation = result.report['head_configuration']['rotation']
    assert (rotation is not None) == rotated
    if rotated:
        assert rotation['convention'] == 'row-DH' and rotation['block_size'] == 64
        assert len(rotation['signs_sha256']) == 64
    assert 'caller initializes' in result.report['seed_scope']
    assert len(callback_events) == 2
    assert [row['positions'] for row in result.report['steps']] == [6, 6]
    assert all(torch.isfinite(torch.tensor(row['kl'])) for row in result.report['steps'])
    assert set(result.trainable_state) == {'raw_mu', 'raw_scale', 'raw_threshold', 'lora_A', 'lora_B'}
    assert all(value.device.type == 'cpu' and value.dtype == torch.float32 for value in result.trainable_state.values())
    assert bool(torch.count_nonzero(result.trainable_state['lora_B']))
    assert torch.equal(weight, model.lm_head.weight)
    state = {k:v.clone() for k,v in result.trainable_state.items()}
    evaluation = evaluate_head_from_cache(head, validation, source_provenance=SOURCE,
                                          token_microbatch=3, vocabulary_chunk=31)
    assert evaluation['role'] == 'search_validation' and evaluation['positions'] == 6
    assert calls == [.5, 1.] and len(optimizer_steps) == 2
    assert all(torch.equal(p, state[n]) for n,p in head.named_parameters())
    assert head._hard_snapshot_sharpness is None


def test_fit_rejects_noncalibration_source_mismatch_overlap_and_one_step(tmp_path):
    model = model_fixture()
    cache = cache_original_head_inputs(model, records(), tmp_path/'fit', SOURCE)
    heldout = cache_original_head_inputs(model, records(), tmp_path/'same', SOURCE, role='search_validation')
    head = BoundedTernaryHead(model.lm_head, row_chunk_size=31, lora_rank=0)
    config = HeadTrainingConfig(epochs=2, vocabulary_chunk=31)
    with pytest.raises(ValueError, match='calibration-role'):
        fit_head_from_cache(head, heldout, config, source_provenance=SOURCE)
    with pytest.raises(ValueError, match='provenance mismatch'):
        fit_head_from_cache(head, cache, config, source_provenance={**SOURCE, 'source_revision': 'different-decoder'})
    other = BoundedTernaryHead(model_fixture().lm_head, row_chunk_size=31, lora_rank=0)
    with torch.no_grad():
        other.weight[0,0].add_(1)
    with pytest.raises(ValueError, match='source mismatch'):
        fit_head_from_cache(other, cache, config, source_provenance=SOURCE)
    with pytest.raises(ValueError, match='overlapping'):
        fit_head_from_cache(head, cache, config, source_provenance=SOURCE, validation_cache=heldout)
    with pytest.raises(ValueError, match='at least one soft'):
        fit_head_from_cache(head, cache, HeadTrainingConfig(epochs=1, batch_size=2), source_provenance=SOURCE)
    assert head.progress == 0


def test_logical_batch_semantics_are_independent_of_disk_shard_size(tmp_path):
    model = model_fixture()
    caches = [cache_original_head_inputs(model, records(), tmp_path/f'cache-{size}', SOURCE, shard_positions=size)
              for size in (1, 128)]
    config = HeadTrainingConfig(epochs=2, batch_size=2, token_microbatch=1, vocabulary_chunk=31)
    results = []
    for cache in caches:
        torch.manual_seed(19)
        head = BoundedTernaryHead(model.lm_head, row_chunk_size=31, lora_rank=2)
        results.append(fit_head_from_cache(head, cache, config, source_provenance=SOURCE))
    a, b = results
    assert len(a.report['steps']) == len(b.report['steps']) == 2
    assert a.report['initial_trainable_state_sha256'] == b.report['initial_trainable_state_sha256']
    for key in a.trainable_state:
        torch.testing.assert_close(a.trainable_state[key], b.trainable_state[key], atol=0, rtol=0)


@pytest.mark.parametrize('flag,value', [('quant_enabled', False), ('rotation_only', True)])
def test_training_refuses_diagnostic_or_original_bypass_before_updates(tmp_path, flag, value):
    model = model_fixture()
    cache = cache_original_head_inputs(model, records(), tmp_path/'fit', SOURCE)
    head = BoundedTernaryHead(model.lm_head, row_chunk_size=31, lora_rank=0)
    setattr(head, flag, value)
    before = {n:p.detach().clone() for n,p in head.named_parameters()}
    with pytest.raises(ValueError, match='quantization enabled'):
        fit_head_from_cache(head, cache, HeadTrainingConfig(), source_provenance=SOURCE)
    assert head.progress == 0
    assert all(torch.equal(p, before[n]) for n,p in head.named_parameters())


def test_initial_state_hash_is_chunk_independent_bounded_and_excludes_source(monkeypatch):
    head = BoundedTernaryHead(model_fixture().lm_head, row_chunk_size=31, lora_rank=4)
    expected = _trainable_state_sha256(head)
    original_cpu = torch.Tensor.cpu
    seen = []
    def bounded_cpu(tensor, *args, **kwargs):
        # A complete original weight matrix or whole B/group-factor array
        # would exceed this limit. One A parameter row legitimately has 128.
        assert tensor.numel() <= 128
        seen.append(tensor.numel())
        return original_cpu(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'cpu', bounded_cpu)
    assert _trainable_state_sha256(head, chunk_elements=10) == expected
    assert len(seen) > 5
    with torch.no_grad():
        head.lora_A[0,0].add_(1)
    assert _trainable_state_sha256(head, chunk_elements=10) != expected


def test_malformed_late_shard_is_rejected_before_any_training_step(tmp_path):
    model = model_fixture()
    cache = cache_original_head_inputs(model, records(), tmp_path/'fit', SOURCE, shard_positions=2)
    last = next(reversed(cache.manifest['files']))
    path = cache.directory/last
    tensors = load_file(str(path))
    tensors['hidden'] = tensors['hidden'].float()
    save_file(tensors, str(path))
    manifest = cache.manifest
    manifest['files'][last].update(bytes=path.stat().st_size,
                                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (cache.directory/'manifest.json').write_text(json.dumps(manifest))
    head = BoundedTernaryHead(model.lm_head, row_chunk_size=31, lora_rank=0)
    with pytest.raises(ValueError, match='original BF16'):
        fit_head_from_cache(head, cache, HeadTrainingConfig(), source_provenance=SOURCE)
    assert head.progress == 0
