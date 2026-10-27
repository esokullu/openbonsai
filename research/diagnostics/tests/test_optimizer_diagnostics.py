"""Stored-moment arithmetic and noninterference, without a language model."""
import copy
import json
import math
import random

import pytest
import torch

from openbonsai.optimizer_diagnostics import summarize_adamw_coordinates
import openbonsai.optimizer_diagnostics as diagnostics


def optimizer_with_state(m, v, *, step=1, betas=(0., 0.), eps=1., lr=.01):
    parameter = torch.nn.Parameter(torch.zeros(len(m), dtype=torch.float32))
    optimizer = torch.optim.AdamW([parameter], betas=betas, eps=eps, lr=lr, weight_decay=0.)
    optimizer.state[parameter] = {'step': torch.tensor(float(step)),
        'exp_avg': torch.tensor(m, dtype=torch.float32), 'exp_avg_sq': torch.tensor(v, dtype=torch.float32)}
    return optimizer, parameter


def test_known_moments_thresholds_domains_and_energy_weighting():
    # r = 0,.5,1,9,99; strict '<' bins distinguish all three thresholds.
    optimizer, p = optimizer_with_state([2., 1., -3., 0., 99.], [0., .25, 1., 81., 9801.])
    result = summarize_adamw_coordinates(optimizer, [('layer.raw_scale', p)], chunk_elements=2)
    row = result['parameters']['layer.raw_scale']
    r = [0., .5, 1., 9., 99.]
    m = [2., 1., -3., 0., 99.]
    ue = [a / (b + 1.) for a, b in zip(m, r)]
    u0 = [m[i] / r[i] for i in range(1, 5)]
    assert row['stored_step'] == 1
    assert row['r_over_epsilon']['min'] == 0.
    assert row['r_over_epsilon']['max'] == 99.
    assert row['r_over_epsilon']['rms'] == pytest.approx(math.sqrt(sum(x*x for x in r)/5))
    assert row['r_over_epsilon']['fractions_below'] == {'1': .4, '9': .6, '99': .8}
    assert row['mhat_rms'] == pytest.approx(math.sqrt(sum(x*x for x in m)/5))
    assert row['undamped_direction']['rms'] == pytest.approx(math.sqrt(sum(x*x for x in u0)/4))
    assert row['epsilon_direction']['rms'] == pytest.approx(math.sqrt(sum(x*x for x in ue)/5))
    expected = math.sqrt(sum(x*x for x in ue[1:])/sum(x*x for x in u0))
    assert row['direction_energy_attenuation']['ratio'] == pytest.approx(expected)
    assert row['direction_energy_attenuation']['undefined_comparator_coordinates'] == 1
    assert row['direction_energy_attenuation']['domain_count'] == 4
    assert row['coordinates']['stored_nonzero_m_zero_v'] == 1
    assert row['coordinates']['stored_zero_m_positive_v'] == 1
    assert row['damping']['mean'] == pytest.approx(sum(x/(x+1) for x in r)/5)
    assert row['damping']['mean'] != pytest.approx(expected)
    assert result['families']['raw_scale']['mhat_rms'] == row['mhat_rms']
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('foreach', [False, True])
def test_actual_adamw_update_oracle_and_stored_step_not_next_step(foreach):
    p = torch.nn.Parameter(torch.tensor([.4, -.2, .7], dtype=torch.float32))
    optimizer = torch.optim.AdamW([p], lr=.03, betas=(.6, .8), eps=.02, weight_decay=.15, foreach=foreach)
    for gradient in ([.02, -.1, .03], [.05, .2, -.4], [.01, -.2, .1]):
        before = p.detach().clone()
        p.grad = torch.tensor(gradient)
        optimizer.step()
        row = summarize_adamw_coordinates(optimizer, {'block.raw_mu': p})['parameters']['block.raw_mu']
        state = optimizer.state[p]
        step = int(state['step'])
        mhat = state['exp_avg'].double() / (1-.6**step)
        r = (state['exp_avg_sq'].double() / (1-.8**step)).sqrt()
        direction = mhat / (r + .02)
        predicted = before.double() * (1-.03*.15) - .03*direction
        torch.testing.assert_close(p.double(), predicted, rtol=1e-6, atol=1e-7)
        assert row['stored_step'] == step
        assert row['bias_correction1'] == pytest.approx(1-.6**step, rel=1e-14)
        assert row['mhat_rms'] == pytest.approx(float(mhat.square().mean().sqrt()), rel=1e-14)
        assert row['epsilon_direction']['rms'] == pytest.approx(float(direction.square().mean().sqrt()), rel=1e-14)
    # A before-step call still observes step3: setting a new gradient is irrelevant.
    expected = summarize_adamw_coordinates(optimizer, {'block.raw_mu': p})
    p.grad = torch.full_like(p, 123.)
    assert summarize_adamw_coordinates(optimizer, {'block.raw_mu': p}) == expected


def test_heterogeneous_groups_and_steps_aggregate_coordinates_not_parameter_means():
    a = torch.nn.Parameter(torch.zeros(1)); b = torch.nn.Parameter(torch.zeros(3))
    optimizer = torch.optim.AdamW([
        {'params': [a], 'lr': .1, 'betas': (.5, .75), 'eps': .1},
        {'params': [b], 'lr': .003, 'betas': (.8, .9), 'eps': .01}])
    optimizer.state[a] = {'step': torch.tensor(2.), 'exp_avg': torch.tensor([.2]), 'exp_avg_sq': torch.tensor([.03])}
    optimizer.state[b] = {'step': torch.tensor(7.), 'exp_avg': torch.tensor([.1, -.2, .3]),
                          'exp_avg_sq': torch.tensor([.1, .2, .3])}
    result = summarize_adamw_coordinates(optimizer, [('first.lora_A', a), ('second.lora_A', b)], chunk_elements=2)
    assert [result['parameters'][name]['stored_step'] for name in ('first.lora_A', 'second.lora_A')] == [2, 7]
    expected_r, expected_m, u0, ue = [], [], [], []
    for p, group, step in [(a, optimizer.param_groups[0], 2), (b, optimizer.param_groups[1], 7)]:
        m = optimizer.state[p]['exp_avg'].double() / (1-group['betas'][0]**step)
        r = (optimizer.state[p]['exp_avg_sq'].double() / (1-group['betas'][1]**step)).sqrt()
        expected_r.extend((r/group['eps']).tolist()); expected_m.extend(m.tolist())
        u0.extend((m/r).tolist()); ue.extend((m/(r+group['eps'])).tolist())
    family = result['families']['lora_A']
    assert family['group_indices'] == [0, 1]
    assert family['r_over_epsilon']['rms'] == pytest.approx(math.sqrt(sum(x*x for x in expected_r)/4))
    assert family['mhat_rms'] == pytest.approx(math.sqrt(sum(x*x for x in expected_m)/4))
    assert family['direction_energy_attenuation']['ratio'] == pytest.approx(math.sqrt(sum(x*x for x in ue)/sum(x*x for x in u0)))
    assert family['direction_energy_attenuation']['includes_learning_rate_or_weight_decay'] is False
    subset = summarize_adamw_coordinates(optimizer, {'second.lora_A': b})
    assert subset['coverage'] == {'optimizer_parameters': 2, 'optimizer_elements': 4,
        'named_parameters': 1, 'named_elements': 3, 'complete_optimizer_coverage': False}


def test_uninitialized_zero_moments_and_undefined_comparators_are_distinct():
    a, b, c, d = (torch.nn.Parameter(torch.zeros(2)) for _ in range(4))
    optimizer = torch.optim.AdamW([a, b, c, d], eps=.1)
    for p, m, v in [(b, [0., 0.], [0., 0.]), (c, [1., -1.], [0., 0.]), (d, [0., 0.], [1., 2.])]:
        optimizer.state[p] = {'step': torch.tensor(4.), 'exp_avg': torch.tensor(m), 'exp_avg_sq': torch.tensor(v)}
    assert a not in optimizer.state
    result = summarize_adamw_coordinates(optimizer, {'a.raw_mu': a, 'b.raw_mu': b, 'c.raw_mu': c, 'd.raw_mu': d})
    assert a not in optimizer.state  # Reading must not trigger defaultdict initialization.
    assert result['parameters']['a.raw_mu']['state_status'] == 'uninitialized'
    assert result['parameters']['a.raw_mu']['mhat_rms'] is None
    assert result['parameters']['a.raw_mu']['stored_step'] is None
    assert result['parameters']['b.raw_mu']['state_status'] == 'initialized'
    assert result['parameters']['b.raw_mu']['mhat_rms'] == 0.
    assert result['parameters']['b.raw_mu']['direction_energy_attenuation']['ratio'] is None
    assert result['parameters']['c.raw_mu']['epsilon_direction']['rms'] > 0
    assert result['parameters']['c.raw_mu']['direction_energy_attenuation']['ratio'] is None
    assert result['parameters']['d.raw_mu']['undamped_direction']['count'] == 2
    assert result['parameters']['d.raw_mu']['direction_energy_attenuation']['ratio'] is None
    assert result['overall']['coordinates'] == {'total': 8, 'observed': 6, 'uninitialized': 2,
        'positive_r': 2, 'stored_zero_m_and_v': 2, 'stored_nonzero_m_zero_v': 2, 'stored_zero_m_positive_v': 2}
    json.dumps(result, allow_nan=False)


def test_chunk_invariance_and_bounded_slice_before_cpu_conversion(monkeypatch):
    n = 137
    optimizer, p = optimizer_with_state(torch.linspace(-.3, .4, n).tolist(),
        torch.linspace(0., .07, n).tolist(), step=17, betas=(.83, .97), eps=1e-5)
    baseline = summarize_adamw_coordinates(optimizer, {'projection.raw_threshold': p}, chunk_elements=n)
    actual = diagnostics._moment_chunk
    seen = []
    def checked(moment, start, end):
        assert 0 < end-start <= 7
        value = actual(moment, start, end)
        assert value.numel() <= 7 and value.dtype == torch.float64 and value.device.type == 'cpu'
        seen.append((start, end))
        return value
    monkeypatch.setattr(diagnostics, '_moment_chunk', checked)
    small = summarize_adamw_coordinates(optimizer, {'projection.raw_threshold': p}, chunk_elements=7)
    assert len(seen) == 2*math.ceil(n/7)
    def compare(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                if k != 'chunk_elements': compare(a[k], b[k])
        elif isinstance(a, list):
            assert len(a) == len(b)
            for x, y in zip(a, b): compare(x, y)
        elif isinstance(a, float): assert a == pytest.approx(b, rel=2e-14, abs=1e-15)
        else: assert a == b
    compare(small, baseline)


def test_observer_preserves_tensors_versions_gradients_state_rng_and_grad_mode():
    optimizer, p = optimizer_with_state([.2, -.1, 0.], [.03, .04, 0.], step=5)
    p.grad = torch.tensor([.7, -.2, 0.])
    uninitialized = torch.nn.Parameter(torch.ones(2))
    optimizer.add_param_group({'params': [uninitialized]})
    tensors = [p, p.grad, uninitialized, *optimizer.state[p].values()]
    copies = [x.detach().clone() for x in tensors]
    refs = [(id(x), x.data_ptr(), x._version, x.requires_grad) for x in tensors]
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    groups = [{k: copy.deepcopy(v) for k,v in g.items() if k != 'params'} for g in optimizer.param_groups]
    state_keys = list(optimizer.state)
    state_ids = {id(k): id(v) for k,v in optimizer.state.items()}
    assert torch.is_grad_enabled()
    summarize_adamw_coordinates(optimizer, [('x.raw_scale', p), ('y.raw_scale', uninitialized)], chunk_elements=1)
    assert torch.is_grad_enabled()
    assert random.getstate() == python_rng and torch.equal(torch.get_rng_state(), torch_rng)
    assert list(optimizer.state) == state_keys and state_ids == {id(k): id(v) for k,v in optimizer.state.items()}
    assert groups == [{k: v for k,v in g.items() if k != 'params'} for g in optimizer.param_groups]
    for x, before, properties in zip(tensors, copies, refs):
        assert torch.equal(x, before)
        assert (id(x), x.data_ptr(), x._version, x.requires_grad) == properties


@pytest.mark.parametrize('damage', ['amsgrad', 'differentiable', 'complex', 'bf16', 'noncontiguous', 'shape',
    'negative_v', 'nan_m', 'inf_v', 'partial', 'extra_state', 'zero_step', 'fractional_step', 'vector_step',
    'zero_eps', 'negative_eps', 'invalid_beta', 'nan_lr', 'negative_decay', 'grad_moment', 'duplicate_group', 'orphan'])
def test_unsupported_or_corrupt_state_rejected_without_mutation(damage):
    optimizer, p = optimizer_with_state([.1, .2, .3, .4], [1., 2., 3., 4.])
    state, group = optimizer.state[p], optimizer.param_groups[0]
    if damage == 'amsgrad': group['amsgrad'] = True
    elif damage == 'differentiable': group['differentiable'] = True
    elif damage == 'complex': state['exp_avg'] = state['exp_avg'].to(torch.complex64)
    elif damage == 'bf16': state['exp_avg'] = state['exp_avg'].bfloat16()
    elif damage == 'noncontiguous': state['exp_avg'] = torch.ones(8)[::2]
    elif damage == 'shape': state['exp_avg'] = state['exp_avg'].reshape(2,2)
    elif damage == 'negative_v': state['exp_avg_sq'][0] = -1
    elif damage == 'nan_m': state['exp_avg'][0] = float('nan')
    elif damage == 'inf_v': state['exp_avg_sq'][0] = float('inf')
    elif damage == 'partial': del state['exp_avg']
    elif damage == 'extra_state': state['max_exp_avg_sq'] = torch.ones(4)
    elif damage == 'zero_step': state['step'].fill_(0.)
    elif damage == 'fractional_step': state['step'].fill_(1.5)
    elif damage == 'vector_step': state['step'] = torch.tensor([1.])
    elif damage == 'zero_eps': group['eps'] = 0.
    elif damage == 'negative_eps': group['eps'] = -1.
    elif damage == 'invalid_beta': group['betas'] = (.9, 1.)
    elif damage == 'nan_lr': group['lr'] = float('nan')
    elif damage == 'negative_decay': group['weight_decay'] = -1.
    elif damage == 'grad_moment': state['exp_avg'].requires_grad_(True)
    elif damage == 'duplicate_group': group['params'].append(p)
    else: optimizer.state[torch.nn.Parameter(torch.ones(1))] = {}
    versions = {k: v._version for k,v in state.items() if torch.is_tensor(v)}
    with pytest.raises((ValueError, TypeError)):
        summarize_adamw_coordinates(optimizer, {'layer.raw_mu': p}, chunk_elements=2)
    assert versions == {k: v._version for k,v in state.items() if torch.is_tensor(v)}


@pytest.mark.parametrize('damage', ['duplicate_name', 'duplicate_param', 'foreign', 'empty', 'bad_name', 'bad_pair'])
def test_named_membership_rejected(damage):
    optimizer, p = optimizer_with_state([1.], [1.])
    q = torch.nn.Parameter(torch.ones(1))
    named = {'duplicate_name': [('a',p),('a',q)], 'duplicate_param': [('a',p),('b',p)],
             'foreign': [('a',q)], 'empty': [], 'bad_name': [('a.',p)], 'bad_pair': [('a',p,0)]}[damage]
    with pytest.raises(ValueError): summarize_adamw_coordinates(optimizer, named)


def test_duplicate_outside_named_subset_is_still_rejected():
    optimizer, p = optimizer_with_state([1.], [1.])
    q = torch.nn.Parameter(torch.ones(1))
    optimizer.add_param_group({'params': [q]})
    optimizer.param_groups[1]['params'].append(q)
    with pytest.raises(ValueError, match='Duplicate'):
        summarize_adamw_coordinates(optimizer, {'a':p})


@pytest.mark.parametrize('chunk', [0, -1, True, 1.5])
def test_invalid_chunks_and_non_adamw(chunk):
    optimizer, p = optimizer_with_state([1.], [1.])
    with pytest.raises(ValueError): summarize_adamw_coordinates(optimizer, {'p':p}, chunk_elements=chunk)
    with pytest.raises(TypeError): summarize_adamw_coordinates(torch.optim.Adam([p]), {'p':p})
