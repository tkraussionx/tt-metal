# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random, is_wormhole_b0

from ttnn.model_preprocessing import preprocess_linear_weight, preprocess_linear_bias


@pytest.mark.parametrize("batch_sizes", [(1,)])
@pytest.mark.parametrize("m_size", [384])
@pytest.mark.parametrize("k_size", [1024])
@pytest.mark.parametrize("n_size", [1024])
@pytest.mark.parametrize("use_bias", [True, False])
def test_linear(
    batch_sizes,
    m_size,
    k_size,
    n_size,
    use_bias,
    *,
    device,
):
    input_shape_a = (*batch_sizes, m_size, k_size)
    input_shape_b = (k_size, n_size)

    torch_input_tensor_a = torch_random(input_shape_a, -0.1, 0.1, dtype=torch.float32)
    torch_input_tensor_b = torch_random(input_shape_b, -0.1, 0.1, dtype=torch.float32)
    if use_bias:
        torch_bias = torch_random((n_size,), -0.1, 0.1, dtype=torch.float32)
    else:
        torch_bias = None
    torch_output_tensor = torch.nn.functional.linear(
        torch_input_tensor_a, torch_input_tensor_b.T.contiguous(), bias=torch_bias
    )

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    if use_bias:
        bias = ttnn.from_torch(
            torch_bias.reshape((1, n_size)),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
    else:
        bias = None

    output_tensor = ttnn.linear(
        input_tensor_a,
        input_tensor_b,
        bias=bias,
    )
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("m_size", [384])
@pytest.mark.parametrize("k_size", [1024])
@pytest.mark.parametrize("n_size", [1024])
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.parametrize("core_grid", [False])
def test_linear_with_core_grid(
    batch_size,
    m_size,
    k_size,
    n_size,
    use_bias,
    core_grid,
    *,
    device,
):
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")
    input_shape_a = (batch_size, 1, m_size, k_size)
    input_shape_b = (k_size, n_size)

    torch_input_tensor_a = torch_random(input_shape_a, -0.1, 0.1, dtype=torch.float32)
    torch_input_tensor_b = torch_random(input_shape_b, -0.1, 0.1, dtype=torch.float32)
    if use_bias:
        torch_bias = torch_random((n_size,), -0.1, 0.1, dtype=torch.float32)
    else:
        torch_bias = None
    torch_output_tensor = torch.nn.functional.linear(
        torch_input_tensor_a, torch_input_tensor_b.T.contiguous(), bias=torch_bias
    )

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.bfloat16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    if use_bias:
        bias = ttnn.from_torch(
            torch_bias.reshape((1, n_size)),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
    else:
        bias = None

    output_tensor = ttnn.linear(
        input_tensor_a,
        input_tensor_b,
        bias=bias,
        core_grid=ttnn.CoreGrid(y=batch_size, x=6),
    )

    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("m_size", [32, 64])
@pytest.mark.parametrize("k_size", [1024, 2048])
@pytest.mark.parametrize("n_size", [1024, 2048])
@pytest.mark.parametrize("activation", [None, "relu", "silu"])
def test_wide_linear_with_argument_for_core_grid_set_to_device_grid(
    device, batch_size, m_size, k_size, n_size, activation
):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((batch_size, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b
    if activation == "relu":
        torch_output_tensor = torch.relu(torch_output_tensor)
    elif activation == "silu":
        torch_output_tensor = torch.nn.functional.silu(torch_output_tensor)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.linear(input_tensor_a, input_tensor_b, core_grid=device.core_grid, activation=activation)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.997)


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("m_size", [32, 64])
@pytest.mark.parametrize("k_size", [1024, 2048])
@pytest.mark.parametrize("n_size", [1024, 2048])
@pytest.mark.parametrize("activation", [None, "relu"])
def test_linear_by_passing_in_1D_systolic_array_program_config(device, batch_size, m_size, k_size, n_size, activation):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((batch_size, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b
    if activation == "relu":
        torch_output_tensor = torch.relu(torch_output_tensor)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.linear(
        input_tensor_a,
        input_tensor_b,
        activation=activation,
        core_grid=device.core_grid,
    )

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.997)


@pytest.mark.parametrize("m_size", [32, 512])
@pytest.mark.parametrize("k_size", [1024, 2048])
@pytest.mark.parametrize("n_size", [1024, 2048])
def test_linear_fp32_acc(device, m_size, k_size, n_size):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((1, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    if is_wormhole_b0():
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
    else:
        # Grayskull doesn't support fp32 but test passing a GS config is ok
        compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
        )

    output_tensor = ttnn.linear(
        input_tensor_a,
        input_tensor_b,
        core_grid=device.core_grid,
        compute_kernel_config=compute_kernel_config,
    )

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.997)


def test_bloom_ff2_linear(device):
    torch_input_tensor = torch_random((8, 384, 4096), -0.1, 0.1, dtype=torch.float32)
    torch_weight = torch_random((4096, 1024), -0.1, 0.1, dtype=torch.float32)
    torch_bias = torch_random((1024,), -0.01, 0.01, dtype=torch.float32)

    torch_output = torch_input_tensor @ torch_weight + torch_bias

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    weights = ttnn.from_torch(
        torch_weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    bias = ttnn.from_torch(
        torch_bias.reshape((1, -1)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    output = ttnn.linear(
        input_tensor,
        weights,
        bias=bias,
        core_grid=device.core_grid,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
    )

    assert ttnn.pearson_correlation_coefficient(torch_output, output) >= 0.9992


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "input_channels, input_height, input_width, filter_height, filter_width, has_bias ",
    (
        (100, 49, 96, 96, 96, True),
        (100, 49, 96, 96, 96, True),
        (100, 49, 96, 96, 96, True),
        (100, 49, 96, 96, 96, False),
        (1, 4096, 96, 384, 96, True),
        (1, 4096, 384, 96, 384, True),
        (100, 49, 96, 96, 96, True),
        (100, 49, 96, 96, 96, False),
        (100, 49, 96, 96, 96, True),
        (100, 49, 96, 96, 96, True),
        (1, 4096, 96, 384, 96, True),
        (1, 4096, 384, 96, 384, False),
        (1, 1024, 384, 192, 384, True),
        (25, 49, 192, 192, 192, True),
        (25, 49, 192, 192, 192, True),
    ),
)
def test_swin_s_linear(device, input_channels, input_height, input_width, filter_height, filter_width, has_bias):
    input_shape = [input_channels, input_height, input_width]
    weights_shape = [filter_height, filter_width]
    linear_bias = None
    if has_bias:
        bias_shape = [384]
        torch_bias_tensor = torch.randn(bias_shape, dtype=torch.bfloat16).float()
        linear_bias = preprocess_linear_bias(torch_bias_tensor, dtype=ttnn.bfloat16)
        linear_bias = ttnn.to_device(linear_bias, device)

    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16).float()
    torch_weights_tensor = torch.randn(weights_shape, dtype=torch.bfloat16).float()

    tt_input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_input_tensor = ttnn.to_memory_config(tt_input_tensor, ttnn.L1_MEMORY_CONFIG)

    linear_weight = preprocess_linear_weight(torch_weights_tensor, dtype=ttnn.bfloat16)
    linear_weight = ttnn.to_device(linear_weight, device)

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    linear_weight = ttnn.to_memory_config(linear_weight, ttnn.L1_MEMORY_CONFIG)

    tt_output_tensor = ttnn.linear(
        tt_input_tensor,
        linear_weight,
        bias=linear_bias,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        compute_kernel_config=compute_kernel_config,
    )
