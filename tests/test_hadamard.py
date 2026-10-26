"""Rotation algebra only: random tensors, no ternary model quality claims."""
import math

import pytest
import torch
from torch.nn import functional as F

from openbonsai.hadamard import inverse_rotate, normalized_hadamard, rotate


def dense_hadamard(block):
    result = torch.ones(1, 1, dtype=torch.float64)
    while result.shape[0] < block:
        result = torch.cat((torch.cat((result, result), 1), torch.cat((result, -result), 1)), 0)
    return result / math.sqrt(block)


@pytest.mark.parametrize("block,blocks", [(1, 1), (8, 1), (16, 3)])
def test_independent_dense_oracle_and_wrong_inverse_negative_control(block, blocks):
    generator = torch.Generator().manual_seed(99)
    width = block * blocks
    signs = torch.randint(0, 2, (width,), generator=generator) * 2 - 1
    h = torch.block_diag(*([dense_hadamard(block)] * blocks))
    r = torch.diag(signs.double()) @ h
    x = torch.randn(2, 3, width, generator=generator, dtype=torch.float64)
    torch.testing.assert_close(rotate(x, signs, block), x @ r, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(inverse_rotate(x, signs, block), x @ r.T, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(inverse_rotate(rotate(x, signs, block), signs, block), x, atol=2e-13, rtol=2e-13)
    torch.testing.assert_close(r @ r.T, torch.eye(width, dtype=torch.float64), atol=2e-13, rtol=2e-13)
    if block > 1:
        assert (rotate(rotate(x, signs, block), signs, block) - x).square().mean().sqrt() > .1
        assert (rotate(x, signs, block) * math.sqrt(block) - x @ r).square().mean().sqrt() > .1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_4096_block_linear_and_embedding_values_and_gradients(dtype):
    generator = torch.Generator().manual_seed(17)
    width = 8192
    signs = torch.randint(0, 2, (width,), generator=generator) * 2 - 1
    x = torch.randn(2, width, generator=generator, dtype=dtype).requires_grad_()
    weight = torch.randn(7, width, generator=generator, dtype=dtype) / math.sqrt(width)
    bias = torch.randn(7, generator=generator, dtype=dtype)
    original = F.linear(x, weight, bias)
    rotated = F.linear(rotate(x, signs, 4096), rotate(weight, signs, 4096), bias)
    tolerance = 2e-5 if dtype == torch.float32 else 2e-11
    torch.testing.assert_close(rotated, original, atol=tolerance, rtol=tolerance)
    original_grad, = torch.autograd.grad(original.sum(), x)
    rotated_grad, = torch.autograd.grad(rotated.sum(), x)
    torch.testing.assert_close(rotated_grad, original_grad, atol=tolerance, rtol=tolerance)
    table = torch.randn(9, width, generator=generator, dtype=dtype).requires_grad_()
    ids = torch.tensor([[8, 0, 2, 2], [1, 8, 2, 0]])
    expected = F.embedding(ids, table)
    actual = inverse_rotate(F.embedding(ids, rotate(table, signs, 4096)), signs, 4096)
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    original_grad, = torch.autograd.grad(expected.square().sum(), table)
    rotated_grad, = torch.autograd.grad(actual.square().sum(), table)
    torch.testing.assert_close(rotated_grad, original_grad, atol=tolerance, rtol=tolerance)


def test_half_precision_accumulates_fp32_and_rounds_once():
    x = torch.linspace(-4, 5, 32).reshape(2, 16).bfloat16()
    expected = normalized_hadamard(x.float(), 8).bfloat16()
    assert torch.equal(normalized_hadamard(x, 8), expected)
    assert normalized_hadamard(torch.empty(2, 0, 16), 8).shape == (2, 0, 16)


@pytest.mark.parametrize("block,width", [(3, 12), (0, 16), (True, 16), (8, 15)])
def test_invalid_block_contract_rejected(block, width):
    with pytest.raises(ValueError):
        rotate(torch.ones(2, width), torch.ones(width), block)


def test_invalid_sign_contract_rejected():
    with pytest.raises(ValueError): rotate(torch.ones(2, 16), torch.ones(15), 8)
    with pytest.raises(ValueError): rotate(torch.ones(2, 16), torch.zeros(16), 8)
