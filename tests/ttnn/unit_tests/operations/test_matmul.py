# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
import os

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull, is_wormhole_b0, is_grayskull, is_blackhole


# fmt: off
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("m_size,k_size,n_size", [
    (1, 2, 2),
    (1, 2, 4),
    (1, 4, 2),
    (1, 4, 4),
    (3, 2, 2),
    (3, 2, 4),
    (3, 4, 2),
    (3, 4, 4),
])
# fmt: on
def test_matmul_with_matched_width_height(device, m_size, k_size, n_size):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = input_tensor_a @ input_tensor_b
    output = ttnn.to_torch(output)

    assert len(output.shape) == len(torch_output_tensor.shape)
    assert output.shape == torch_output_tensor.shape
    assert_with_pcc(torch_output_tensor, output, 0.99981)


# fmt: off
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("k_size, n_size", [
    (2, 4),
    (4, 2),
    (2, 4),
    (4, 2),
    (2, 4),
    (4, 2),
    (4, 4),
    ])
# fmt: on
def test_matmul_with_matched_width_height_from_1D(device, k_size, n_size):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = input_tensor_a @ input_tensor_b
    output = ttnn.to_torch(output, torch_rank=1)

    assert len(output.shape) == len(torch_output_tensor.shape)
    assert output.shape == torch_output_tensor.shape
    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.skip(reason="ttnn.reshape doesn't support reshaping the input tensors used in this test")
@pytest.mark.parametrize("w", [(4), (2)])
def test_matmul_does_dot_product(device, w):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.zeros((w,), dtype=torch.bfloat16).uniform_(-0.1, 0.1)
    torch_input_tensor_b = torch.zeros((w,), dtype=torch.bfloat16).uniform_(-0.1, 0.1)
    torch_output_tensor = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b)
    input_tensor_a = ttnn.to_device(input_tensor_a, device)
    input_tensor_b = ttnn.to_device(input_tensor_b, device)
    output = ttnn.matmul(input_tensor_a, input_tensor_b)
    output = ttnn.from_device(output)

    output = ttnn.to_torch(output)

    assert torch_output_tensor.shape == ()
    assert output.shape == (32,)
    assert torch.allclose(torch_output_tensor, output[0], atol=1e-2)


# fmt: off
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("n_size,c,h,w", [
    (1, 1, 2, 4),
    (1, 1, 4, 2),
    (3, 3, 2, 4),
    (3, 3, 4, 2),
    (1, 3, 2, 4),
    (3, 1, 4, 2),
    ])
# fmt: on
def test_matmul_with_matched_width_height_4D(device, n_size, c, h, w):
    torch_input_tensor_a = torch.rand((n_size, c, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((n_size, c, w, h), dtype=torch.bfloat16)
    torch_output_tensor = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.matmul(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    assert len(output.shape) == len(torch_output_tensor.shape)
    assert output.shape == torch_output_tensor.shape
    assert_with_pcc(torch_output_tensor, output, 0.999599)


# fmt: off
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("n_size,c,h,w", [
    (1, 1, 2, 2),
    (1, 1, 4, 4),
    (3, 3, 4, 4),
    (3, 1, 4, 4),
    (1, 3, 4, 4)
    ])
# fmt: on
def test_matmul_same_shape_and_valid(device, n_size, c, h, w):
    torch_input_tensor_a = torch.rand((n_size, c, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((n_size, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.matmul(input_tensor_a, input_tensor_b)
    output = ttnn.to_torch(output)

    assert len(output.shape) == len(torch_output_tensor.shape)
    assert output.shape == torch_output_tensor.shape
    assert_with_pcc(torch_output_tensor, output, 0.9997)


# fmt: off
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("input_a,input_b", [
        ([1.0,2.0,3.0],[3.0,4.0,5.0])
    ])
# fmt: on
def test_matmul_same_shape_but_invalid(device, input_a, input_b):
    # pad the lists with zeros to make it 32 so that it fits nicely on the device.
    input_a += [0.0] * (32 - len(input_a))
    input_b += [0.0] * (32 - len(input_b))

    torch_input_tensor_a = torch.as_tensor(input_a, dtype=torch.bfloat16).reshape((1, 1, 1, len(input_a)))
    torch_input_tensor_b = torch.as_tensor(input_b, dtype=torch.bfloat16).reshape((1, 1, 1, len(input_b)))

    with pytest.raises(RuntimeError) as exception:
        torch.matmul(torch_input_tensor_a, torch_input_tensor_b)
    assert "Expected size for first two dimensions of batch2 tensor to be: [1, 32] but got: [1, 1]." in str(
        exception.value
    )

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    with pytest.raises(RuntimeError) as exception:
        ttnn.matmul(input_tensor_a, input_tensor_b)
    assert "The width of the first tensor must be equal to the height of the second tensor" in str(exception.value)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
def test_tutorial_matmul(device):
    torch.manual_seed(0)

    m_size = 1024
    k_size = 1024
    n_size = 512

    torch_input_tensor_a = torch.randn((m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output = input_tensor_a @ input_tensor_b
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, pcc=0.999)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
def test_tutorial_matmul_inputs_and_output_in_l1_memory(device):
    torch.manual_seed(0)

    m_size = 1024
    k_size = 1024
    n_size = 512

    torch_input_tensor_a = torch.randn((m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    output = ttnn.matmul(input_tensor_a, input_tensor_b, memory_config=ttnn.L1_MEMORY_CONFIG)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, pcc=0.999)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
def test_tutorial_matmul_with_inputs_and_output_in_l1_memory_and_user_specified_core_grid(device):
    torch.manual_seed(0)

    m_size = 1024
    k_size = 1024
    n_size = 512

    torch_input_tensor_a = torch.randn((m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    output = ttnn.matmul(
        input_tensor_a, input_tensor_b, memory_config=ttnn.L1_MEMORY_CONFIG, core_grid=ttnn.CoreGrid(y=4, x=4)
    )

    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, pcc=0.999)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize(
    "batch_size_0, batch_size_1, m_size, k_size, n_size, bcast_batch, input_a_sharded_memory_config_args, input_b_sharded_memory_config_args",
    [
        (
            2,
            3,
            1600,
            224,
            896,
            True,
            dict(core_grid=ttnn.CoreGrid(y=5, x=7), strategy=ttnn.ShardStrategy.BLOCK),
            None,
        ),  # mcast 2d
        (
            2,
            3,
            1600,
            224,
            896,
            True,
            dict(
                core_grid=ttnn.CoreGrid(y=7, x=5),
                strategy=ttnn.ShardStrategy.BLOCK,
                orientation=ttnn.ShardOrientation.COL_MAJOR,
            ),
            None,
        ),  # mcast 2d transposed
        (
            2,
            1,
            128,
            256,
            512,
            True,
            dict(core_grid=ttnn.CoreGrid(y=2, x=2), strategy=ttnn.ShardStrategy.BLOCK),
            None,
        ),  # mcast 2d with shard width > 1 TILE
        (
            2,
            3,
            64,
            32 * 7,
            1024,
            True,
            dict(
                core_grid=ttnn.CoreGrid(y=1, x=7),
                strategy=ttnn.ShardStrategy.WIDTH,
            ),
            None,
        ),  # mcast in0
        (
            2,
            3,
            160 * 7,
            64,
            64,
            True,
            dict(
                core_grid=ttnn.CoreGrid(y=7, x=1),
                strategy=ttnn.ShardStrategy.HEIGHT,
            ),
            None,
        ),  # mcast in1
        (
            7,
            7,
            384,
            64,
            384,
            False,
            dict(
                core_grid=ttnn.CoreGrid(y=7, x=7),
                strategy=ttnn.ShardStrategy.HEIGHT,
                use_height_and_width_as_shard_shape=True,
            ),
            dict(
                core_grid=ttnn.CoreGrid(y=7, x=7),
                strategy=ttnn.ShardStrategy.HEIGHT,
                use_height_and_width_as_shard_shape=True,
            ),
        ),  # bmm
    ],
    ids=["mcast_2d", "mcast_2d_transposed", "mcast_2d_shard_width_gt_1", "mcast_in0", "mcast_in1", "bmm"],
)
def test_sharded_matmul(
    device,
    batch_size_0,
    batch_size_1,
    m_size,
    k_size,
    n_size,
    bcast_batch,
    input_a_sharded_memory_config_args,
    input_b_sharded_memory_config_args,
):
    torch.manual_seed(0)

    input_a_shape = [batch_size_0, batch_size_1, m_size, k_size]
    if bcast_batch:
        input_b_shape = [k_size, n_size]
    else:
        input_b_shape = [batch_size_0, batch_size_1, k_size, n_size]

    torch_input_tensor_a = torch.randn(input_a_shape)
    torch_input_tensor_b = torch.randn(input_b_shape)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a.to(torch.bfloat16))
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b.to(torch.bfloat16))

    input_tensor_a = ttnn.to_device(input_tensor_a, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    input_tensor_b = ttnn.to_device(input_tensor_b, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.TILE_LAYOUT)
    input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.TILE_LAYOUT)

    input_a_sharded_memory_config = ttnn.create_sharded_memory_config(
        input_a_shape, **input_a_sharded_memory_config_args
    )
    input_tensor_a = ttnn.to_memory_config(input_tensor_a, input_a_sharded_memory_config)

    if input_b_sharded_memory_config_args:
        input_b_sharded_memory_config = ttnn.create_sharded_memory_config(
            input_b_shape, **input_b_sharded_memory_config_args
        )
        input_tensor_b = ttnn.to_memory_config(input_tensor_b, input_b_sharded_memory_config)

    output = ttnn.matmul(input_tensor_a, input_tensor_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.from_device(output)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, pcc=0.999)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("batch_size", [1, 7])
def test_matmul_with_core_grid(device, batch_size):
    torch.manual_seed(0)

    m_size = 384
    k_size = 1024
    n_size = 1024

    torch_input_tensor_a = torch.randn((batch_size, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        core_grid=ttnn.CoreGrid(y=batch_size, x=8),
    )

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.999)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("m_size", [30, 61])
@pytest.mark.parametrize("k_size", [1023, 2048])
@pytest.mark.parametrize("n_size", [1021, 2048])
def test_wide_matmul_with_argument_for_core_grid_set_to_device_grid(device, batch_size, m_size, k_size, n_size):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((batch_size, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        core_grid=device.core_grid,
    )

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, 0.997)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("m_size", [1024, 2048])
@pytest.mark.parametrize("k_size", [1023, 2048])
@pytest.mark.parametrize("n_size", [32, 61])
def test_tall_matmul_with_argument_for_core_grid_set_to_device_grid(device, batch_size, m_size, k_size, n_size):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((batch_size, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        core_grid=device.core_grid,
    )

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.997)


@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("m_size", [31, 63])
@pytest.mark.parametrize("k_size", [1024, 2048])
@pytest.mark.parametrize("n_size", [1023, 2047])
def test_matmul_by_passing_in_1D_systolic_array_program_config(device, batch_size, m_size, k_size, n_size):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((batch_size, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        core_grid=device.core_grid,
    )

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.997)


@pytest.mark.parametrize(
    "n_size, c, m, k, n",
    [
        (1, 1, 2, 3, 4),
        (1, 1, 1024, 64, 512),
    ],
)
@pytest.mark.parametrize("transpose_b", [True, False])
@pytest.mark.parametrize("transpose_a", [True, False])
def test_matmul_with_transpose_a_or_b(device, n_size, c, m, k, n, transpose_a, transpose_b):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((n_size, c, m, k), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((n_size, c, k, n), dtype=torch.bfloat16)
    torch_output_tensor = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    if transpose_a:
        torch_input_tensor_a = torch_input_tensor_a.transpose(-1, -2)
    if transpose_b:
        torch_input_tensor_b = torch_input_tensor_b.transpose(-1, -2)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.matmul(input_tensor_a, input_tensor_b, transpose_a=transpose_a, transpose_b=transpose_b)
    output = ttnn.to_torch(output)

    assert len(output.shape) == len(torch_output_tensor.shape)
    assert output.shape == torch_output_tensor.shape
    assert_with_pcc(torch_output_tensor, output, 0.999)


##########################
# MODEL SPECIFIC MATMULS #
##########################
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("m_size", [128])
@pytest.mark.parametrize("k_size", [4544])
@pytest.mark.parametrize("n_size", [4672])
@pytest.mark.parametrize("core_grid", [None, ttnn.CoreGrid(y=7, x=8)])
def test_falcon_query_key_value_matmul(device, batch_size, m_size, k_size, n_size, core_grid):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((batch_size, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.matmul(
        input_tensor_a,
        input_tensor_b,
        core_grid=core_grid,
    )

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=0.996)


# @skip_for_grayskull()
@pytest.mark.skipif(is_wormhole_b0() or is_blackhole(), reason="Unsupported on WH and BH")
@pytest.mark.parametrize(
    "batch_size, channel_a, channel_b, m_size, k_size, n_size, has_bias",
    [
        (1, 2, 1, 1024, 640, 2560, False),
        (2, 8, 8, 64, 96, 160, False),
        (1, 2, 1, 4096, 320, 1280, False),
        (1, 2, 1, 64, 1280, 5120, False),
        (2, 8, 8, 64, 64, 160, False),
        (1, 2, 1, 1024, 640, 768, False),
        (2, 8, 8, 96, 160, 96, False),
        (2, 8, 8, 1024, 1024, 96, False),
        (1, 2, 1, 96, 768, 1024, False),
        (1, 1, 1, 32, 1280, 1280, True),
        (2, 8, 8, 4096, 96, 64, False),
        (1, 2, 1, 64, 5120, 1280, True),
        (2, 8, 8, 4096, 64, 96, False),
        (1, 2, 1, 1024, 768, 640, True),
        (1, 2, 1, 256, 1280, 1280, True),
        (2, 8, 8, 1024, 96, 96, False),
        (1, 2, 1, 1024, 640, 2304, False),
        (1, 1, 1, 32, 1280, 320, True),
        (1, 2, 1, 96, 768, 2560, False),
        (1, 2, 1, 4096, 1280, 320, True),
        (1, 2, 1, 1024, 2560, 640, True),
        (1, 2, 1, 256, 1280, 3840, False),
        (1, 1, 1, 32, 320, 1280, True),
        (1, 2, 1, 4096, 512, 320, True),
        (1, 2, 1, 64, 1280, 1280, True),
        (1, 2, 1, 256, 5120, 1280, True),
        (1, 2, 1, 256, 1280, 1280, False),
        (2, 8, 8, 256, 160, 96, False),
        (2, 8, 8, 256, 256, 160, False),
        (1, 2, 1, 96, 768, 1536, False),
        (1, 2, 1, 64, 1280, 3840, False),
        (2, 8, 8, 1024, 96, 1024, False),
        (2, 8, 8, 256, 96, 160, False),
        (1, 2, 1, 64, 1280, 1280, False),
        (2, 8, 8, 4096, 64, 4096, False),
        (1, 1, 1, 32, 1280, 640, True),
        (2, 8, 8, 64, 160, 64, False),
        (1, 2, 1, 4096, 320, 1536, False),
        (1, 2, 1, 256, 1280, 5120, False),
        (2, 8, 8, 4096, 4096, 64, False),
        (2, 8, 8, 256, 160, 256, False),
        (1, 2, 1, 4096, 320, 512, False),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b])
def test_sd_matmul(device, batch_size, channel_a, channel_b, m_size, k_size, n_size, has_bias, dtype):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip("Issue #6984: Compute Grid size too small")
    core_grid = ttnn.CoreGrid(x=8, y=8)
    TILE_HEIGHT = 32

    if batch_size == 2:
        if (m_size == 1024 and k_size == 96 and n_size == 1024) or (m_size == 4096 and k_size == 64 and n_size == 4096):
            # NOTE: matmul errors out with OOM otherwise
            core_grid = None

    # if batch_size == 2:
    #     if m_size == 1024 and k_size == 96 and n_size == 1024 and (dtype == ttnn.bfloat16 or is_grayskull()):
    #         pytest.skip("skip: Raises OOM")
    #     if m_size == 4096 and k_size == 64 and n_size == 4096:
    #         pytest.skip("skip: Raises OOM without decomposition")
    #     if is_grayskull():
    #         if m_size == 4096 and (
    #             (k_size == 96 and n_size == 64) or (k_size == 64 and n_size == 96) or (k_size == 4096 and n_size == 64)
    #         ):
    #             pytest.skip("skip: Raises OOM on GS")

    torch_input_tensor_a = torch.randn((batch_size, channel_a, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((batch_size, channel_b, k_size, n_size), dtype=torch.bfloat16)
    torch_output_tensor = torch_input_tensor_a @ torch_input_tensor_b
    if has_bias:
        torch_input_tensor_c = torch.randn((1, 1, TILE_HEIGHT, n_size), dtype=torch.bfloat16)
        _torch_input_tensor_c = torch.repeat_interleave(
            torch_input_tensor_c, torch_output_tensor.shape[2] // TILE_HEIGHT, dim=2
        )
        torch_output_tensor = torch_output_tensor + _torch_input_tensor_c

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    input_tensor_c = (
        ttnn.from_torch(torch_input_tensor_c, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype) if has_bias else None
    )
    pcc = 0.94 if dtype == ttnn.bfloat8_b else 0.98

    if has_bias:
        output_tensor = ttnn.linear(
            input_tensor_a,
            input_tensor_b,
            bias=input_tensor_c,
            core_grid=core_grid,
        )
    else:
        output_tensor = ttnn.matmul(
            input_tensor_a,
            input_tensor_b,
            core_grid=core_grid,
        )

    output_tensor = ttnn.to_torch(output_tensor)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)


# ------------------------ GH ISSUE #12453 ------------------------------------
# The following tests are an interactive explanation of the
# issue raised by GH#12453 which questions the resulsts of HiFi2/HiFi4
# -----------------------------------------------------------------------------


# This test creates 2 constant input vecotors and performs a matmul
# in HiFi2/HiFi4 and then compares the results.
@pytest.mark.skipif(is_grayskull(), reason="Unsupported on GS")
@pytest.mark.parametrize("batch_size", [int(os.getenv("BATCH_SIZE", 1))])
# Let's keep this dimension fixed when testing, it's easier to understand what's
# going on
@pytest.mark.parametrize("m_size", [int(os.getenv("M_SIZE", 32))])
# The inner dimension dictates the size of the difference between HiFi2/HiFi4
# k_size = 1024, diff = 8
# k_size = 512, diff = 4
# k_size = 128, diff = 1
@pytest.mark.parametrize("k_size", [int(os.getenv("K_SIZE", 1024))])
# This dimension dictates whether or not the error will be visible. It is due to
# nature of bfloat16 data storage - operands are brought to common exponent so
# small value + big value =~ big value. If we make this 512 or smaller, the
# difference between HiFi2 and HiFi4 will not be visible
@pytest.mark.parametrize("n_size", [int(os.getenv("N_SIZE", 1024))])
# Considering the k_size, this will be the difference between HiFi2 and HiFi4
# We want atol_th to be the same or larger than the difference for test to pass.
# If you play around and make this just a bit less than the actuall difference,
# the test will fail
@pytest.mark.parametrize("atol_th", [int(os.getenv("ATOL", 8))])
# This is number ensures that the least significant bit of the mantissa is '1'.
# If you make this a really small number, e.g. 0.0001, you can't see the difference
# even for really large tensors because both fidelities errors will be the same.
@pytest.mark.parametrize("scalar", [float(os.getenv("SCALAR", 1.0078125))])
def test_matmul_fidelity_constant_tensors(device, batch_size, m_size, k_size, n_size, atol_th, scalar):
    torch_input_tensor_a = torch.full((batch_size, m_size, k_size), scalar, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.full((k_size, n_size), scalar, dtype=torch.bfloat16)
    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    hifi2 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    hifi4 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    output_hifi2 = ttnn.to_torch(
        ttnn.matmul(
            input_tensor_a,
            input_tensor_b,
            compute_kernel_config=hifi2,
        )
    )

    output_hifi4 = ttnn.to_torch(
        ttnn.matmul(
            input_tensor_a,
            input_tensor_b,
            compute_kernel_config=hifi4,
        )
    )

    print(f"Max HiFi4 vs HiFi2 difference: {(output_hifi4.abs() - output_hifi2.abs()).abs().max().item():.8f}")
    count = 0

    for hf4, hf2 in zip(output_hifi4.flatten(), output_hifi2.flatten()):
        if abs(abs(hf4.item()) - abs(hf2.item())) >= atol_th and count < 10:
            count = count + 1
            print(f"HiFi2 value is {hf2.item():.8f} and HiFi4 value is {hf4.item():.8f}")

    assert torch.allclose(output_hifi2, output_hifi4, atol=atol_th)


# This test creates 2 input vecotors with uniformly distributed values and performs
# a matmul in HiFi2/HiFi4 and then compares the results.
@pytest.mark.skipif(is_grayskull(), reason="Unsupported on GS")
@pytest.mark.parametrize("batch_size", [int(os.getenv("BATCH_SIZE", 1))])
# Let's keep this dimension fixed when testing, it's easier to understand what's
# going on
@pytest.mark.parametrize("m_size", [int(os.getenv("M_SIZE", 32))])
# The inner dimension dictates the size of the difference between HiFi2/HiFi4
# This time around, it's a bit harder to predict the differnce
@pytest.mark.parametrize("k_size", [int(os.getenv("K_SIZE", 1024))])
# This dimension dictates whether or not the error will be visible. It is due to
# nature of bfloat16 data storage - operands are brought to common exponent so
# small value + big value =~ big value. If we make this 512 or smaller, the
# difference between HiFi2 and HiFi4 will not be visible
@pytest.mark.parametrize("n_size", [int(os.getenv("N_SIZE", 1024))])
# Note that now atol_th can be smaller for test to pass compared to the previous example.
# This is because the values are from 0 to 1 now, so overall the difference is smaller.
@pytest.mark.parametrize("atol_th", [int(os.getenv("ATOL", 2))])
def test_matmul_fidelity_uniform_tensors(device, batch_size, m_size, k_size, n_size, atol_th):
    torch_input_tensor_a = torch.rand((batch_size, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((k_size, n_size), dtype=torch.bfloat16)
    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    hifi2 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    hifi4 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    output_hifi2 = ttnn.to_torch(
        ttnn.matmul(
            input_tensor_a,
            input_tensor_b,
            compute_kernel_config=hifi2,
        )
    )

    output_hifi4 = ttnn.to_torch(
        ttnn.matmul(
            input_tensor_a,
            input_tensor_b,
            compute_kernel_config=hifi4,
        )
    )

    print(f"Max HiFi4 vs HiFi2 difference: {(output_hifi4.abs() - output_hifi2.abs()).abs().max().item():.8f}")
    count = 0

    for hf4, hf2 in zip(output_hifi4.flatten(), output_hifi2.flatten()):
        if abs(abs(hf4.item()) - abs(hf2.item())) >= atol_th and count < 10:
            count = count + 1
            print(f"HiFi2 value is {hf2.item():.8f} and HiFi4 value is {hf4.item():.8f}")

    assert torch.allclose(output_hifi2, output_hifi4, atol=atol_th)


# This test creates 2 input vecotors with standard normal distribution of values and
# performs a matmul in HiFi2/HiFi4 and then compares the results.
@pytest.mark.skipif(is_grayskull(), reason="Unsupported on GS")
@pytest.mark.parametrize("batch_size", [int(os.getenv("BATCH_SIZE", 1))])
# Let's keep this dimension fixed when testing, it's easier to understand what's
# going on
@pytest.mark.parametrize("m_size", [int(os.getenv("M_SIZE", 32))])
# The inner dimension dictates the size of the difference between HiFi2/HiFi4
# This time around, it's a bit harder to predict the differnce
@pytest.mark.parametrize("k_size", [int(os.getenv("K_SIZE", 1024))])
# This dimension dictates whether or not the error will be visible. It is due to
# nature of bfloat16 data storage - operands are brought to common exponent so
# small value + big value =~ big value. If we make this 512 or smaller, the
# difference between HiFi2 and HiFi4 will not be visible
@pytest.mark.parametrize("n_size", [int(os.getenv("N_SIZE", 1024))])
# This is the original example - atol can be even smaller because the biggest differences
# in fidelity calculations are around plus/minus 2/3. This is because these two numbers need 2
# bits to represent the whole part of the number, so less space is left for mantissa. However,
# the distribution is bipolar, so the differences largely cancel each other out, leading to
# the smallest atol_th among this and previous two tests.
@pytest.mark.parametrize("atol_th", [int(os.getenv("ATOL", 1))])
def test_matmul_fidelity_std_norm_tensors(device, batch_size, m_size, k_size, n_size, atol_th):
    torch_input_tensor_a = torch.randn((batch_size, m_size, k_size), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((k_size, n_size), dtype=torch.bfloat16)
    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    hifi2 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    hifi4 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    output_hifi2 = ttnn.to_torch(
        ttnn.matmul(
            input_tensor_a,
            input_tensor_b,
            compute_kernel_config=hifi2,
        )
    )

    output_hifi4 = ttnn.to_torch(
        ttnn.matmul(
            input_tensor_a,
            input_tensor_b,
            compute_kernel_config=hifi4,
        )
    )

    print(f"Max HiFi4 vs HiFi2 difference: {(output_hifi4.abs() - output_hifi2.abs()).abs().max().item():.8f}")
    count = 0

    for hf4, hf2 in zip(output_hifi4.flatten(), output_hifi2.flatten()):
        if abs(abs(hf4.item()) - abs(hf2.item())) >= atol_th and count < 10:
            count = count + 1
            print(f"HiFi2 value is {hf2.item():.8f} and HiFi4 value is {hf4.item():.8f}")

    assert torch.allclose(output_hifi2, output_hifi4, atol=atol_th)


# This test creates an input vecotor with constant value and performs a
# matmul in HiFi2/HiFi4 with unary tensor and then compares the results.
@pytest.mark.skipif(is_grayskull(), reason="Unsupported on GS")
@pytest.mark.parametrize("batch_size", [int(os.getenv("BATCH_SIZE", 1))])
# Let's keep this dimension fixed when testing, it's easier to understand what's
# going on
@pytest.mark.parametrize("m_size", [int(os.getenv("M_SIZE", 32))])
# Here the inner dimension doesn't influence the size of the error because there's
# no accumulation, or there is but only with zeroes. Note that because we're multiplying
# with a unary tensor, n_size is equal to k_size
@pytest.mark.parametrize("k_size", [int(os.getenv("K_SIZE", 4096))])
# Now here we get to the smallest difference, i.e. only the last bit of the second
# operand will not be included, meaning that the error has to be 0.0000001 binary,
# or 0.0078125 decimal. Run the test and watch what happens
@pytest.mark.parametrize("atol_th", [float(os.getenv("ATOL", 0.0078125))])
# This is number ensures that the least significant bit of the mantissa is '1'.
# If you make this a really small number, e.g. 0.0001, you can't see the difference
# even for really large tensors because both fidelities errors will be the same.
@pytest.mark.parametrize("scalar", [float(os.getenv("SCALAR", 1.0078125))])
def test_matmul_fidelity_const_unary_tensors(device, batch_size, m_size, k_size, atol_th, scalar):
    torch_input_tensor_a = torch.full((batch_size, m_size, k_size), scalar, dtype=torch.bfloat16)
    torch_input_tensor_b = (
        torch.eye(k_size, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1, 1).to(torch.bfloat16)
    )
    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    hifi2 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    hifi4 = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    output_hifi2 = ttnn.to_torch(
        ttnn.matmul(
            input_tensor_a,
            input_tensor_b,
            compute_kernel_config=hifi2,
        )
    )

    output_hifi4 = ttnn.to_torch(
        ttnn.matmul(
            input_tensor_a,
            input_tensor_b,
            compute_kernel_config=hifi4,
        )
    )

    print(f"Max HiFi4 vs HiFi2 difference: {(output_hifi4.abs() - output_hifi2.abs()).abs().max().item():.8f}")
    count = 0

    for hf4, hf2 in zip(output_hifi4.flatten(), output_hifi2.flatten()):
        if abs(abs(hf4.item()) - abs(hf2.item())) >= atol_th and count < 10:
            count = count + 1
            print(f"HiFi2 value is {hf2.item():.8f} and HiFi4 value is {hf4.item():.8f}")

    assert torch.allclose(output_hifi2, output_hifi4, atol=atol_th)
