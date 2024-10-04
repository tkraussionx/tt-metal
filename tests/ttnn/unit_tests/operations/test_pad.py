# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_wormhole_b0


@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [230])
@pytest.mark.parametrize("w", [224])
@pytest.mark.parametrize("padding,torch_padding", [(((0, 1), (3, 25), (32, 32)), (32, 32, 3, 25, 0, 1))])
@pytest.mark.parametrize("value", [0])
def test_pad_rm(device, n, c, h, w, padding, torch_padding, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


def run_pad_rm_with_program_cache(device, n, c, h, w, padding, torch_padding, value, use_program_cache):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)
    output_tensor = ttnn.to_torch(output_tensor)

    assert output_tensor.shape == torch_output_tensor.shape
    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [224])
@pytest.mark.parametrize("w", [224])
@pytest.mark.parametrize("padding,torch_padding", [(((0, 1), (0, 32), (0, 32)), (0, 32, 0, 32, 0, 1))])
@pytest.mark.parametrize("value", [0])
def test_pad_rm_with_program_cache(device, n, c, h, w, padding, torch_padding, value, use_program_cache):
    for _ in range(2):
        run_pad_rm_with_program_cache(device, n, c, h, w, padding, torch_padding, value, use_program_cache)
        # dummy tensor to change tensor alloc
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.from_torch(
            py_dummy_tensor,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    assert device.num_program_cache_entries() == 1


def run_pad_rm_sharded(dtype, device, n, c, h, w, padding, torch_padding, value, shard_strategy):
    torch.manual_seed(0)

    ttnn_dtype = {
        torch.bfloat16: ttnn.DataType.BFLOAT16,
        torch.float32: ttnn.DataType.FLOAT32,
    }[dtype]

    torch_input_tensor = torch.rand((n, c, h, w), dtype=dtype)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    def factorize_core_grid(core_grid, total_size_shards):
        for x in range(core_grid.x, 0, -1):
            for y in range(core_grid.y, 0, -1):
                if total_size_shards % (x * y) == 0:
                    return ttnn.CoreGrid(x=x, y=y)
        pytest.skip("Couldn't find a core grid that can shard the tensor")

    total_size_shards = n * c * h if shard_strategy == ttnn.ShardStrategy.HEIGHT else n * c * w
    core_grid = factorize_core_grid(device.core_grid, total_size_shards)

    input_shard_config = ttnn.create_sharded_memory_config([n, c, h, w], core_grid, shard_strategy)

    try:
        tt_input_tensor = ttnn.from_torch(
            torch_input_tensor,
            dtype=ttnn_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=input_shard_config,
        )
    except RuntimeError as e:
        if "Invalid sharding core grid" in str(e):
            pytest.skip("Invalid core grid for this sharding configuration")
        raise

    def safe_sum_padding(padding, i):
        try:
            return sum(padding[i])
        except IndexError:
            return 0

    n_padded = n + safe_sum_padding(padding, 0)
    c_padded = c + safe_sum_padding(padding, 1)
    h_padded = h + safe_sum_padding(padding, 2)
    w_padded = w + safe_sum_padding(padding, 3)

    # output shard config
    total_size_shards = (
        n_padded * c_padded * h_padded
        if shard_strategy == ttnn.ShardStrategy.HEIGHT
        else n_padded * c_padded * w_padded
    )
    core_grid = factorize_core_grid(device.core_grid, total_size_shards)
    output_shard_config = ttnn.create_sharded_memory_config(
        [n_padded, c_padded, h_padded, w_padded], core_grid, shard_strategy
    )

    tt_output_tensor = ttnn.pad(tt_input_tensor, padding=padding, value=value, memory_config=output_shard_config)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    assert tt_output_tensor.shape == torch_output_tensor.shape
    assert_with_pcc(torch_output_tensor, tt_output_tensor, 0.9999)


def padding_and_torch_padding(padding):
    return (padding, tuple(sum(reversed(padding), ())))


@pytest.mark.parametrize(
    "dtype_shape_and_padding",
    [
        [torch.bfloat16, (20, 3, 224, 256), ((1, 1), (2, 32), (0, 0))],
        [torch.float32, (1, 1, 337920, 4), ((0, 0), (0, 0), (0, 0), (0, 12))],
        [torch.bfloat16, (20, 3, 224, 256), ((1, 1), (1, 1), (1, 1), (1, 1))],
    ],
)
@pytest.mark.parametrize("value", [0])
@pytest.mark.parametrize("shard_strategy", [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH])
def test_pad_rm_sharded(device, dtype_shape_and_padding, value, shard_strategy, use_program_cache):
    dtype, shape, padding = dtype_shape_and_padding
    _, torch_padding = padding_and_torch_padding(padding)
    n, c, h, w = shape
    if device.core_grid.y < 8:
        pytest.skip("n300 does not have 8x8 grid")
    for _ in range(2):
        run_pad_rm_sharded(dtype, device, n, c, h, w, padding, torch_padding, value, shard_strategy)
        # dummy tensor to change tensor alloc
        dummy_shape = [1, 1, 32, 32]
        py_dummy_tensor = torch.randn(dummy_shape)
        tt_dummy_tensor = ttnn.from_torch(
            py_dummy_tensor,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
    assert device.num_program_cache_entries() == 2


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("padding,torch_padding", [(((0, 64),), (0, 64)), (((16, 16), (0, 32)), (0, 32, 0, 32))])
@pytest.mark.parametrize("value", [0])
def test_pad(device, h, w, padding, torch_padding, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)

    output_tensor = ttnn.to_torch(output_tensor)
    assert output_tensor.shape == torch_output_tensor.shape

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.parametrize("h", [2, 30])
@pytest.mark.parametrize("w", [128, 60])
@pytest.mark.parametrize("padding", [((0, 32), (0, 32)), ((0, 32), (0, 64))])
@pytest.mark.parametrize("value", [0])
def test_pad_any_input_shape(device, h, w, padding, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)

    output_tensor = ttnn.to_torch(output_tensor)
    tilezed_input_shape = input_tensor.shape.with_tile_padding()
    th = tilezed_input_shape[-2]
    tw = tilezed_input_shape[-1]
    assert output_tensor.shape == ttnn.Shape((th + padding[0][0] + padding[0][1], tw + padding[1][0] + padding[1][1]))


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("padding,torch_padding", [(((32, 32),), (32, 32))])
@pytest.mark.parametrize("value", [0])
def test_pad_padding_validation_front_pad_not_supported(device, h, w, padding, torch_padding, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    with pytest.raises(RuntimeError) as e:
        ttnn.pad(input_tensor, padding=padding, value=value)
    assert "ttnn.pad: on device padding currently supports front padding only for row major tensors" in str(e.value)
    return


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("padding,torch_padding", [(((0, 32), (0, 32), (0, 32)), (0, 32, 0, 32, 0, 32))])
@pytest.mark.parametrize("value", [0])
def test_pad_padding_validation_length(device, h, w, padding, torch_padding, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    with pytest.raises(RuntimeError) as e:
        ttnn.pad(input_tensor, padding=padding, value=value)
    assert "ttnn.pad: padding len can't be larger than input tensor rank" in str(e.value)
    return


@pytest.mark.skip(reason="ttnn.pad does not support row_major tensors because the kernel currently causes a PCC error")
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("padding,torch_padding", [(((0, 1), (0, 2)), (0, 2, 0, 1)), (((1, 1), (4, 2)), (4, 2, 1, 1))])
@pytest.mark.parametrize("value", [0])
def test_pad(device, h, w, padding, torch_padding, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)

    assert output_tensor.shape == ttnn.Shape((h + padding[0][0] + padding[0][1], w + padding[1][0] + padding[1][1]))

    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.skip(reason="ttnn.pad does not support row_major tensors because the kernel currently causes a PCC error")
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("padding,torch_padding", [(((0, 1), (0, 2)), (0, 2, 0, 1)), (((1, 1), (4, 2)), (4, 2, 1, 1))])
@pytest.mark.parametrize("value", [0])
def test_pad_back_to_back(device, h, w, padding, torch_padding, value):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor)
    input_tensor = ttnn.to_device(input_tensor, device)
    output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)
    output_tensor = ttnn.pad(output_tensor, padding=padding, value=value)

    assert output_tensor.shape == ttnn.Shape(
        (h + (padding[0][0] + padding[0][1]) * 2, w + (padding[1][0] + padding[1][1]) * 2)
    )

    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)


@pytest.mark.skip(reason="ttnn.pad requires pad to start at 0")
@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
@pytest.mark.parametrize("padding", [((0, 32), (0, 32)), ((1, 64), (0, 96)), ((0, 64), (0, 43)), ((32, 64), (64, 96))])
@pytest.mark.parametrize("value", [0])
def test_pad_for_tensor_in_tile_layout(device, h, w, padding, value):
    torch.manual_seed(0)
    torch_padding = (padding[1][0], padding[1][1], padding[0][0], padding[0][1])

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.pad(torch_input_tensor, torch_padding, mode="constant", value=value)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT)
    input_tensor = ttnn.to_device(input_tensor, device)
    if (
        padding[0][0] % ttnn.TILE_SIZE != 0
        or padding[0][1] % ttnn.TILE_SIZE != 0
        or padding[1][0] % ttnn.TILE_SIZE != 0
        or padding[1][1] % ttnn.TILE_SIZE != 0
    ):
        with pytest.raises(RuntimeError) as e:
            output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)
        assert "must be a multiple of the tile size on height and width" in str(e.value)
        return
    else:
        output_tensor = ttnn.pad(input_tensor, padding=padding, value=value)

    assert output_tensor.shape == ttnn.Shape((h + padding[0][0] + padding[0][1], w + padding[1][0] + padding[1][1]))

    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, 0.9999)
