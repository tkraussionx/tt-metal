# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import data_gen_with_range, compare_pcc
from models.utility_functions import skip_for_grayskull


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_square_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.square(input_tensor, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.square(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("exponent", [0.5, 2.0])
def test_unary_pow_ttnn(input_shapes, exponent, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)
    _, output_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    cq_id = 0
    ttnn.pow(input_tensor, exponent, output_tensor=output_tensor, queue_id=cq_id)
    golden_tensor = torch.pow(in_data, exponent)

    comp_pass = compare_pcc([output_tensor], [golden_tensor], pcc=0.9)
    assert comp_pass

def test_unary_tanhshrink_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.tanhshrink(input_tensor)
    golden_tensor = torch.nn.functional.tanhshrink(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_acosh_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.acosh(input_tensor)
    golden_tensor = torch.acosh(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_asinh_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.asinh(input_tensor)
    golden_tensor = torch.asinh(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_atanh_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.atanh(input_tensor)
    golden_tensor = torch.atanh(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_cbrt_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.cbrt(input_tensor)
    golden_tensor = in_data.sign() * in_data.abs().pow(1.0 / 3.0)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


# range is -9 to 9
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_cosh_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -9, 9, device)

    output_tensor = ttnn.cosh(input_tensor)
    golden_tensor = torch.cosh(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


# range limit 1, to 1000
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_digamma_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 1, 100, device)

    output_tensor = ttnn.digamma(input_tensor)
    golden_tensor = torch.digamma(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_lgamma_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 0.1, 1e32, device)

    output_tensor = ttnn.lgamma(input_tensor)
    golden_tensor = torch.lgamma(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_log1p_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -1, 1, device)

    output_tensor = ttnn.log1p(input_tensor)
    golden_tensor = torch.log1p(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@skip_for_grayskull()
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_mish_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.mish(input_tensor)
    golden_tensor = in_data * torch.tanh(softplus(in_data, beta=1.0, threshold=20.0))

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_multigammaln_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 1.6, 1e32, device)

    output_tensor = ttnn.multigammaln(input_tensor)
    golden_tensor = torch.special.multigammaln(in_data, 4)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_sinh_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -9, 9, device)

    output_tensor = ttnn.sinh(input_tensor)
    golden_tensor = torch.sinh(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_softsign_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, 1, 100, device)

    output_tensor = ttnn.softsign(input_tensor)
    golden_tensor = torch.nn.functional.softsign(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass


# 0.9714 pcc
@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_unary_swish_ttnn(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    output_tensor = ttnn.swish(input_tensor)
    golden_tensor = torch.nn.functional.silu(in_data)

    comp_pass = compare_pcc([output_tensor], [golden_tensor])
    assert comp_pass
