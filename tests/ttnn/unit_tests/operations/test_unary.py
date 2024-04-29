# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch
import tt_lib
import ttnn
from tests.tt_eager.python_api_testing.unit_testing.backward_ops.utility_funcs import (
    data_gen_with_range,
    compare_pcc,
    data_gen_with_val,
)
from tests.tt_eager.python_api_testing.sweep_tests import (
    comparison_funcs,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random, skip_for_grayskull, skip_for_wormhole_b0


def run_unary_test(device, h, w, pcc=0.9999):
    torch.manual_seed(0)
    input_shapes = [1, 1, 32, 32]
    in_data, input_tensor = data_gen_with_val(input_shapes, device, True, val=h)
    grad_data, grad_tensor = data_gen_with_val(input_shapes, device, val=w)

    golden_tensor = torch.remainder(in_data, grad_data)
    golden_tensor = torch.where(torch.isnan(golden_tensor), torch.tensor(float("inf")), golden_tensor)

    tt_output_tensor_on_device = tt_lib.tensor.atan2(input_tensor, grad_tensor)
    tt_out_tensor = tt_output_tensor_on_device.cpu().to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch()

    differing_elements = torch.ne(golden_tensor, tt_out_tensor)
    total_differences = differing_elements.sum().item()
    print("Total differing elements --> ", total_differences)
    diff = torch.abs(golden_tensor - tt_out_tensor)
    if True:
        print("Inputs for which the outputs differ by more than 0:")
        indices = torch.nonzero(diff)
        iter = 0
        for idx in indices:
            if iter < 2:
                input1_val = in_data[idx[0], idx[1], idx[2], idx[3]]
                input2_val = grad_data[idx[0], idx[1], idx[2], idx[3]]
                expected_output_val = golden_tensor[idx[0], idx[1], idx[2], idx[3]]
                actual_output_val = tt_out_tensor[idx[0], idx[1], idx[2], idx[3]]
                print(
                    f"Input 1 value: {input1_val}, Input 2 value: {input2_val}, Expected output: {expected_output_val}, Actual output: {actual_output_val}"
                )
                print("diff ", torch.abs(expected_output_val) - torch.abs(actual_output_val))
                iter += 1

    comp_pass, comp_out = comparison_funcs.comp_pcc(golden_tensor, tt_out_tensor)
    comp_all, _ = comparison_funcs.comp_allclose(golden_tensor, tt_out_tensor, atol=4, rtol=1e-1)
    print(comp_pass)
    print(comp_all)
    print(comp_out)
    # print(tt_out_tensor)
    # print(golden_tensor)
    if total_differences > 0:
        assert False


@pytest.mark.parametrize(
    "h",
    [
        -50,
        -49.875,
        -49.75,
        -49.625,
        -49.5,
        -49.375,
        -49.25,
        -49.125,
        -49.0,
        -48.875,
        -48.75,
        -48.625,
        -48.5,
        -48.375,
        -48.25,
        -48.125,
        -48.0,
        -47.875,
        -47.75,
        -47.625,
        -47.5,
        -47.375,
        -47.25,
        -47.125,
        -47.0,
        -46.875,
        -46.75,
        -46.625,
        -46.5,
        -46.375,
        -46.25,
        -46.125,
        -46.0,
        -45.875,
        -45.75,
        -45.625,
        -45.5,
        -45.375,
        -45.25,
        -45.125,
        -45.0,
        -44.875,
        -44.75,
        -44.625,
        -44.5,
        -44.375,
        -44.25,
        -44.125,
        -44.0,
        -43.875,
        -43.75,
        -43.625,
        -43.5,
        -43.375,
        -43.25,
        -43.125,
        -43.0,
        -42.875,
        -42.75,
        -42.625,
        -42.5,
        -42.375,
        -42.25,
        -42.125,
        -42.0,
        -41.875,
        -41.75,
        -41.625,
        -41.5,
        -41.375,
        -41.25,
        -41.125,
        -41.0,
        -40.875,
        -40.75,
        -40.625,
        -40.5,
        -40.375,
        -40.25,
        -40.125,
        -40.0,
        -39.875,
        -39.75,
        -39.625,
        -39.5,
        -39.375,
        -39.25,
        -39.125,
        -39.0,
        -38.875,
        -38.75,
        -38.625,
        -38.5,
        -38.375,
        -38.25,
        -38.125,
        -38.0,
        -37.875,
        -37.75,
        -37.625,
        -37.5,
        -37.375,
        -37.25,
        -37.125,
        -37.0,
        -36.875,
        -36.75,
        -36.625,
        -36.5,
        -36.375,
        -36.25,
        -36.125,
        -36.0,
        -35.875,
        -35.75,
        -35.625,
        -35.5,
        -35.375,
        -35.25,
        -35.125,
        -35.0,
        -34.875,
        -34.75,
        -34.625,
        -34.5,
        -34.375,
        -34.25,
        -34.125,
        -34.0,
        -33.875,
        -33.75,
        -33.625,
        -33.5,
        -33.375,
        -33.25,
        -33.125,
        -33.0,
        -32.875,
        -32.75,
        -32.625,
        -32.5,
        -32.375,
        -32.25,
        -32.125,
        -32.0,
        -31.875,
        -31.75,
        -31.625,
        -31.5,
        -31.375,
        -31.25,
        -31.125,
        -31.0,
        -30.875,
        -30.75,
        -30.625,
        -30.5,
        -30.375,
        -30.25,
        -30.125,
    ],
)
@pytest.mark.parametrize("w", [63.5])
def test_fp32_uint32(device, h, w):
    run_unary_test(device, h, w)


"""
@pytest.mark.parametrize("h", [11.0, 12.125, 13.25, 14.375, 15.5, 16.625, 17.75, 18.875, 19.0, 0, -11.0, -12.125, -13.25, -14.375, -15.5, -16.625, -17.75, -18.875, -19.0])
@pytest.mark.parametrize("w", [-89.0, -88.125, -87.25, -86.375, -85.5, -84.625, -83.75, -82.875, -81.0, 0, 79.0, 78.125, 77.25, 76.375, 75.5, 74.625, 73.75, 72.875, 71.0])
@pytest.mark.parametrize("h", [11.0, 12.125, 13.25, 14.375, 15.5, 16.625, 17.75, 18.875, 19.0, 0, -11.0, -12.125, -13.25, -14.375, -15.5, -16.625, -17.75, -18.875, -19.0, -89.0, -88.125, -87.25, -86.375, -85.5, -84.625, -83.75, -82.875, -81.0, 79.0, 78.125, 77.25, 76.375, 75.5, 74.625, 73.75, 72.875, 71.0])
@pytest.mark.parametrize("w", [-89.0, -88.125, -87.25, -86.375, -85.5, -84.625, -83.75, -82.875, -81.0, 0, 79.0, 78.125, 77.25, 76.375, 75.5, 74.625, 73.75, 72.875, 71.0, 11.0, 12.125, 13.25, 14.375, 15.5, 16.625, 17.75, 18.875, 19.0, -11.0, -12.125, -13.25, -14.375, -15.5, -16.625, -17.75, -18.875, -19.0])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_exp(device, h, w):
    run_unary_test(device, h, w, ttnn.exp, torch.exp, pcc=0.9998)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_tanh(device, h, w):
    run_unary_test(device, h, w, ttnn.tanh, torch.tanh, pcc=0.993)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_gelu(device, h, w):
    run_unary_test(device, h, w, ttnn.gelu, torch.nn.functional.gelu, pcc=0.9996)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_relu(device, h, w):
    run_unary_test(device, h, w, ttnn.relu, torch.relu)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_rsqrt(device, h, w):
    run_unary_test(device, h, w, ttnn.rsqrt, torch.rsqrt)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_silu(device, h, w):
    run_unary_test(device, h, w, ttnn.silu, torch.nn.functional.silu)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_log(device, h, w):
    run_unary_test(device, h, w, ttnn.log, torch.log)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sin(device, h, w):
    run_unary_test(device, h, w, ttnn.sin, torch.sin)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_asin(device, h, w):
    run_unary_test(device, h, w, ttnn.asin, torch.asin, pcc=0.999)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_cos(device, h, w):
    run_unary_test(device, h, w, ttnn.cos, torch.cos, pcc=0.999)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_acos(device, h, w):
    run_unary_test(device, h, w, ttnn.acos, torch.acos, pcc=0.999)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_tan(device, h, w):
    run_unary_test(device, h, w, ttnn.tan, torch.tan)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_atan(device, h, w):
    run_unary_test(device, h, w, ttnn.atan, torch.atan)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_sinh(device, h, w):
    run_unary_test(device, h, w, ttnn.sinh, torch.sinh)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_asinh(device, h, w):
    run_unary_test(device, h, w, ttnn.asinh, torch.asinh, pcc=0.9997)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_cosh(device, h, w):
    run_unary_test(device, h, w, ttnn.cosh, torch.cosh, pcc=0.999)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@skip_for_wormhole_b0("Issue #6991: Failing on wormhole_b0 PCC issue")
def test_acosh(device, h, w):
    run_unary_test(device, h, w, ttnn.acosh, torch.acosh)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_atanh(device, h, w):
    run_unary_test(device, h, w, ttnn.atanh, torch.atanh)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_logical_not(device, h, w):
    run_unary_test(device, h, w, ttnn.logical_not, torch.logical_not)


def run_unary_test_range(device, h, w, ttnn_function, torch_function, pcc=0.9999):
    torch.manual_seed(0)
    low = -100
    high = 100

    torch_input_tensor = torch_random((h, w), low, high, dtype=torch.bfloat16)
    torch_output_tensor = torch_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_signbit(device, h, w):
    run_unary_test_range(device, h, w, ttnn.signbit, torch.signbit, pcc=0.99)


def run_unary_test_with_float(device, h, w, scalar, ttnn_function, torch_function, pcc=0.9999):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch_function(torch_input_tensor, scalar)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor, scalar)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("scalar", [1, 2])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@skip_for_wormhole_b0("Issue #6991: Failing on wormhole_b0 PCC issue")
def test_logit(device, h, w, scalar):
    run_unary_test_with_float(device, h, w, scalar, ttnn.logit, torch.logit)
"""
