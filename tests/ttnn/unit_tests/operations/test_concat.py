# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("height", [20, 32])
@pytest.mark.parametrize("width", [4, 32])
@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("async_mode", [True, False], ids=["async_on", "async_off"])
def test_concat(device, height, width, dim, async_mode):
    device.enable_async(async_mode)
    torch_input_tensor_a = torch.rand((height, width), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((height, width), dtype=torch.bfloat16)
    torch_output_tensor = torch.concat([torch_input_tensor_a, torch_input_tensor_b], dim=dim)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    if ttnn.has_tile_padding(input_tensor_a, dim=dim) or ttnn.has_tile_padding(input_tensor_b, dim=dim):
        pytest.skip("Cannot concat tensors with tile padding")

    output = ttnn.concat([input_tensor_a, input_tensor_b], dim=dim)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize(
    "input_shape_a, shard_shape_a, input_shape_b, shard_shape_b, output_shard_shape, shard_grid",
    (
        (
            (1, 1, 16, 16),
            (8, 16),
            (1, 1, 16, 16),
            (8, 16),
            (8, 32),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
        ),
        (
            (1, 1, 160, 32),
            (80, 32),
            (1, 1, 160, 32),
            (80, 32),
            (80, 64),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
        ),
        (
            (1, 1, 160, 32),
            (80, 32),
            (1, 1, 160, 16),
            (80, 16),
            (80, 48),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 1))}),
        ),
        (
            (1, 1, 25600, 64),
            (512, 64),
            (1, 1, 25600, 64),
            (512, 64),
            (512, 128),
            ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5)),
                    ttnn.CoreRange(ttnn.CoreCoord(0, 6), ttnn.CoreCoord(1, 6)),
                }
            ),
        ),
    ),
)
@pytest.mark.parametrize("async_mode", [True, False], ids=["async_on", "async_off"])
def test_sharded_concat(
    device, input_shape_a, shard_shape_a, input_shape_b, shard_shape_b, output_shard_shape, shard_grid, async_mode
):
    device.enable_async(async_mode)
    input_a_sharded_memory_config = ttnn.create_sharded_memory_config(
        shard_shape_a,
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    input_b_sharded_memory_config = ttnn.create_sharded_memory_config(
        shard_shape_b,
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    output_sharded_memory_config = ttnn.create_sharded_memory_config(
        output_shard_shape,
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    torch_input_tensor_a = torch.rand(input_shape_a, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(input_shape_b, dtype=torch.bfloat16)
    torch_output_tensor = torch.concat([torch_input_tensor_a, torch_input_tensor_b], dim=3)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    input_tensor_a = ttnn.to_memory_config(input_tensor_a, input_a_sharded_memory_config)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    input_tensor_b = ttnn.to_memory_config(input_tensor_b, input_b_sharded_memory_config)

    output = ttnn.concat([input_tensor_a, input_tensor_b], dim=3, memory_config=output_sharded_memory_config)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)
    assert_with_pcc(torch_output_tensor, output)


@pytest.mark.parametrize(
    "input1, input2",
    (
        ([1, 128, 80, 80], [1, 128, 80, 80]),
        ([1, 256, 40, 40], [1, 256, 40, 40]),
        ([1, 512, 20, 20], [1, 512, 20, 20]),
    ),
)
@pytest.mark.parametrize("dim", [1])
@pytest.mark.parametrize("async_mode", [True, False], ids=["async_on", "async_off"])
def test_concat_yolov7_2inputs(device, input1, input2, dim, async_mode):
    device.enable_async(async_mode)
    torch_input_tensor_a = torch.rand(input1, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(input2, dtype=torch.bfloat16)
    torch_output_tensor = torch.concat([torch_input_tensor_a, torch_input_tensor_b], dim=dim)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    if ttnn.has_tile_padding(input_tensor_a, dim=dim) or ttnn.has_tile_padding(input_tensor_b, dim=dim):
        pytest.skip("Cannot concat tensors with tile padding")

    output = ttnn.concat([input_tensor_a, input_tensor_b], dim=dim)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize(
    "input1, input2, input3",
    (
        ([1, 128, 40, 40], [1, 128, 40, 40], [1, 256, 40, 40]),
        ([1, 256, 20, 20], [1, 256, 20, 20], [1, 512, 20, 20]),
    ),
)
@pytest.mark.parametrize("dim", [1])
@pytest.mark.parametrize("async_mode", [True, False], ids=["async_on", "async_off"])
def test_concat_yolov7_3inputs(device, input1, input2, input3, dim, async_mode):
    device.enable_async(async_mode)
    torch_input_tensor_a = torch.rand(input1, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(input2, dtype=torch.bfloat16)
    torch_input_tensor_c = torch.rand(input3, dtype=torch.bfloat16)
    torch_output_tensor = torch.concat([torch_input_tensor_a, torch_input_tensor_b, torch_input_tensor_c], dim=dim)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_c = ttnn.from_torch(torch_input_tensor_c, layout=ttnn.TILE_LAYOUT, device=device)

    if (
        ttnn.has_tile_padding(input_tensor_a, dim=dim)
        or ttnn.has_tile_padding(input_tensor_b, dim=dim)
        or ttnn.has_tile_padding(input_tensor_c, dim=dim)
    ):
        pytest.skip("Cannot concat tensors with tile padding")

    output = ttnn.concat([input_tensor_a, input_tensor_b, input_tensor_c], dim=dim)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize(
    "input1, input2, input3, input4",
    (
        ([1, 64, 160, 160], [1, 64, 160, 160], [1, 64, 160, 160], [1, 64, 160, 160]),
        ([1, 128, 80, 80], [1, 128, 80, 80], [1, 128, 80, 80], [1, 128, 80, 80]),
        ([1, 256, 40, 40], [1, 256, 40, 40], [1, 256, 40, 40], [1, 256, 40, 40]),
        ([1, 256, 20, 20], [1, 256, 20, 20], [1, 256, 20, 20], [1, 256, 20, 20]),
    ),
)
@pytest.mark.parametrize("dim", [1])
@pytest.mark.parametrize("async_mode", [True, False], ids=["async_on", "async_off"])
def test_concat_yolov7_4inputs(device, input1, input2, input3, input4, dim, async_mode):
    device.enable_async(async_mode)
    torch_input_tensor_a = torch.rand(input1, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(input2, dtype=torch.bfloat16)
    torch_input_tensor_c = torch.rand(input3, dtype=torch.bfloat16)
    torch_input_tensor_d = torch.rand(input4, dtype=torch.bfloat16)
    torch_output_tensor = torch.concat(
        [torch_input_tensor_a, torch_input_tensor_b, torch_input_tensor_c, torch_input_tensor_d], dim=dim
    )

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_c = ttnn.from_torch(torch_input_tensor_c, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_d = ttnn.from_torch(torch_input_tensor_d, layout=ttnn.TILE_LAYOUT, device=device)

    if (
        ttnn.has_tile_padding(input_tensor_a, dim=dim)
        or ttnn.has_tile_padding(input_tensor_b, dim=dim)
        or ttnn.has_tile_padding(input_tensor_c, dim=dim)
        or ttnn.has_tile_padding(input_tensor_d, dim=dim)
    ):
        pytest.skip("Cannot concat tensors with tile padding")

    output = ttnn.concat([input_tensor_a, input_tensor_b, input_tensor_c, input_tensor_d], dim=dim)
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)


@pytest.mark.parametrize(
    "input1, input2, input3, input4, input5, input6",
    (
        ([1, 128, 40, 40], [1, 128, 40, 40], [1, 128, 40, 40], [1, 128, 40, 40], [1, 256, 40, 40], [1, 256, 40, 40]),
        ([1, 64, 80, 80], [1, 64, 80, 80], [1, 64, 80, 80], [1, 64, 80, 80], [1, 128, 80, 80], [1, 128, 80, 80]),
        ([1, 256, 20, 20], [1, 256, 20, 20], [1, 256, 20, 20], [1, 256, 20, 20], [1, 512, 20, 20], [1, 512, 20, 20]),
    ),
)
@pytest.mark.parametrize("dim", [1])
@pytest.mark.parametrize("async_mode", [True, False], ids=["async_on", "async_off"])
def test_concat_yolov7_6inputs(device, input1, input2, input3, input4, input5, input6, dim, async_mode):
    device.enable_async(async_mode)
    torch_input_tensor_a = torch.rand(input1, dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand(input2, dtype=torch.bfloat16)
    torch_input_tensor_c = torch.rand(input3, dtype=torch.bfloat16)
    torch_input_tensor_d = torch.rand(input4, dtype=torch.bfloat16)
    torch_input_tensor_e = torch.rand(input5, dtype=torch.bfloat16)
    torch_input_tensor_f = torch.rand(input6, dtype=torch.bfloat16)
    torch_output_tensor = torch.concat(
        [
            torch_input_tensor_a,
            torch_input_tensor_b,
            torch_input_tensor_c,
            torch_input_tensor_d,
            torch_input_tensor_e,
            torch_input_tensor_f,
        ],
        dim=dim,
    )

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_c = ttnn.from_torch(torch_input_tensor_c, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_d = ttnn.from_torch(torch_input_tensor_d, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_e = ttnn.from_torch(torch_input_tensor_e, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_f = ttnn.from_torch(torch_input_tensor_f, layout=ttnn.TILE_LAYOUT, device=device)

    if (
        ttnn.has_tile_padding(input_tensor_a, dim=dim)
        or ttnn.has_tile_padding(input_tensor_b, dim=dim)
        or ttnn.has_tile_padding(input_tensor_c, dim=dim)
        or ttnn.has_tile_padding(input_tensor_d, dim=dim)
        or ttnn.has_tile_padding(input_tensor_e, dim=dim)
        or ttnn.has_tile_padding(input_tensor_f, dim=dim)
    ):
        pytest.skip("Cannot concat tensors with tile padding")

    output = ttnn.concat(
        [input_tensor_a, input_tensor_b, input_tensor_c, input_tensor_d, input_tensor_e, input_tensor_f], dim=dim
    )
    output = ttnn.to_torch(output)

    assert_with_pcc(torch_output_tensor, output, 0.9999)
