# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import math
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull, is_wormhole_b0, is_grayskull, is_blackhole, run_for_wormhole_b0


def find_max_subblock(out_block_h, out_block_w):
    max_product = 0
    best_h = 1
    best_w = 1

    for h in range(1, out_block_h + 1):
        if out_block_h % h == 0:  # h is a divisor of out_block_h
            for w in range(1, out_block_w + 1):
                if out_block_w % w == 0 and h * w <= 8:  # w is a divisor and product condition met
                    if h * w > max_product:
                        max_product = h * w
                        best_h = h
                        best_w = w
    if out_block_w > best_w:
        best_h = 1
    return best_h, best_w, max_product


# @pytest.mark.parametrize("n", [2])
# @pytest.mark.parametrize("c", [3])
# @pytest.mark.parametrize("h", [96])
# @pytest.mark.parametrize("w", [128])
# @pytest.mark.parametrize("tile_h", [16, 32])
# @pytest.mark.parametrize("tile_w", [16, 32])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("c", [1])
@pytest.mark.parametrize("h", [16])
@pytest.mark.parametrize("w", [16])
@pytest.mark.parametrize("tile_h", [16])
@pytest.mark.parametrize("tile_w", [16])
def test_tiny_tiles_transposed_tile(device, n, c, h, w, tile_h, tile_w):
    torch.manual_seed(0)
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        tile=ttnn.Tile((tile_h, tile_w), transpose_tile=True),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    output_tensor = ttnn.to_torch(input_tensor)

    flattened_tensor = output_tensor.flatten()
    tile_hw = tile_h * tile_w
    face_h = 16
    face_w = 16
    face_hw = face_h * face_w
    num_tiles_x = w // tile_w
    num_tiles_y = n * c * h // tile_h
    num_faces_y = tile_h // face_h
    num_faces_x = tile_w // face_w
    result = []

    if tile_h == 16 and tile_w == 32:
        for tile_y in range(num_tiles_y):
            tile_start = tile_y * w * tile_h
            for col in range(face_w):
                start = tile_start + col
                for tile_x in range(num_tiles_x):
                    start_tile_x = start + tile_x * tile_w
                    for face_x in range(num_faces_x):
                        offset = start_tile_x + face_x * face_w
                        for row in range(face_h):
                            result.append(flattened_tensor[offset + row * w])
        transposed_output_tensor = torch.tensor(result)
        transposed_output_tensor = transposed_output_tensor.view(n, c, h, w)
    elif tile_h == 32 and tile_w == 16:
        for tile_y in range(num_tiles_y):
            tile_start = tile_y * w * tile_h
            for face_y in range(num_faces_y):
                start = tile_start + face_y * face_h * w
                for col in range(face_w):
                    start_col = start + col
                    for tile_x in range(num_tiles_x):
                        offset = start_col + tile_x * tile_w
                        for row in range(face_h):
                            result.append(flattened_tensor[offset + row * w])
        transposed_output_tensor = torch.tensor(result)
        transposed_output_tensor = transposed_output_tensor.view(n, c, h, w)
    else:
        output_tensor = output_tensor.view(n, c, h // tile_h, tile_h, w // tile_w, tile_w)
        output_tensor = output_tensor.permute(0, 1, 2, 4, 3, 5).transpose(-1, -2)
        output_tensor = output_tensor.permute(0, 1, 2, 4, 3, 5)
        transposed_output_tensor = output_tensor.contiguous().view(n, c, h, w)

    assert_with_pcc(torch_input_tensor, transposed_output_tensor, 1)


@pytest.mark.parametrize("n", [2])
@pytest.mark.parametrize("c", [5])
@pytest.mark.parametrize("h", [384])
@pytest.mark.parametrize("w", [768])
@pytest.mark.parametrize("tile_h", [16, 32])
@pytest.mark.parametrize("tile_w", [16, 32])
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_tiny_tiles_bfloat(device, n, c, h, w, tile_h, tile_w, dtype):
    torch.manual_seed(0)
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        tile=ttnn.Tile((tile_h, tile_w)),
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    output_tensor = ttnn.to_torch(input_tensor)
    if dtype == ttnn.bfloat8_b:
        expected_pcc = 0.9999
    elif dtype == ttnn.bfloat4_b:
        expected_pcc = 0.989
    assert_with_pcc(torch_input_tensor, output_tensor, expected_pcc)


@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("c", [2])
@pytest.mark.parametrize("h", [71])
@pytest.mark.parametrize("w", [35])
@pytest.mark.parametrize("tile_h", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("tile_w", [16, 32])
def test_tiny_tiles(device, n, c, h, w, tile_h, tile_w):
    torch.manual_seed(0)
    torch_input_tensor = torch.rand((n, c, h, w), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        tile=ttnn.Tile((tile_h, tile_w)),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    output_tensor = ttnn.to_torch(input_tensor)
    assert_with_pcc(torch_input_tensor, output_tensor, 1)


@run_for_wormhole_b0()
# @pytest.mark.parametrize("b", [8])
# @pytest.mark.parametrize("h", [4])
# @pytest.mark.parametrize("m", [256])
# @pytest.mark.parametrize("k", [256])
# @pytest.mark.parametrize("n", [256])
# @pytest.mark.parametrize("tile_h", [16, 32])
# @pytest.mark.parametrize("tile_w", [16, 32])
# @pytest.mark.parametrize("in0_sharded", [True, False])
# @pytest.mark.parametrize("in1_sharded", [True, False])
# @pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("b", [1])
@pytest.mark.parametrize("h", [1])
@pytest.mark.parametrize("m", [16])
@pytest.mark.parametrize("k", [32])
@pytest.mark.parametrize("n", [16])
@pytest.mark.parametrize("in0_sharded", [True])
@pytest.mark.parametrize("in1_sharded", [True])
@pytest.mark.parametrize("out_sharded", [True])
@pytest.mark.parametrize("tile_h", [16])
@pytest.mark.parametrize("tile_w", [16])
@pytest.mark.parametrize("transpose_tile", [False])
def test_matmul_reuse_config_sharded_tiny_tile(
    device, b, h, m, k, n, tile_h, tile_w, in0_sharded, in1_sharded, out_sharded, transpose_tile
):
    torch.manual_seed(0)

    grid_size = (b, h)

    in0 = torch.ones([b, h, m, k]).bfloat16().float()
    in1 = torch.rand([b, h, k, n]).bfloat16().float()

    # print(in1)

    if in0_sharded:
        in0_memory_config = ttnn.create_sharded_memory_config(
            (b, h, m, k),
            core_grid=ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
    else:
        in0_memory_config = ttnn.L1_MEMORY_CONFIG
    in0_t = ttnn.from_torch(
        in0,
        tile=ttnn.Tile((tile_h, 32)),
        dtype=ttnn.bfloat8_b,
        # dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_memory_config,
    )

    if in1_sharded:
        in1_memory_config = ttnn.create_sharded_memory_config(
            (b, h, k, n),
            core_grid=ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
    else:
        in1_memory_config = ttnn.L1_MEMORY_CONFIG
    in1_t = ttnn.from_torch(
        in1,
        tile=ttnn.Tile((32, tile_w), transpose_tile=transpose_tile),
        dtype=ttnn.bfloat8_b,
        # dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in1_memory_config,
    )

    out_block_h = m // tile_h
    out_block_w = n // tile_w
    out_subblock_h, out_subblock_w, _ = find_max_subblock(out_block_h, out_block_w)

    program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=k // 32,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
    )
    if out_sharded:
        out_mem_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type=ttnn.BufferType.L1,
        )
    else:
        out_mem_config = ttnn.L1_MEMORY_CONFIG
    # override the tile width for later ops
    # if out_sharded and tile_h <= 16:
    #     output_tile = ttnn.Tile([tile_h, 32])
    # else:
    #     output_tile = ttnn.Tile([tile_h, tile_w])
    # output_tile = ttnn.Tile([tile_h, tile_w])
    output_t = ttnn.matmul(
        # in0_t, in1_t, program_config=program_config, memory_config=out_mem_config, output_tile=output_tile
        in0_t,
        in1_t,
        dtype=ttnn.bfloat8_b,
        # dtype=ttnn.bfloat16,
        program_config=program_config,
        memory_config=out_mem_config,
    )
    output_tensor = ttnn.to_torch(output_t)
    pt_out = in0 @ in1

    print(pt_out)
    print(output_tensor)

    assert_with_pcc(pt_out, output_tensor, 0.999)


def pad_to_dram_banks(num, tile_w, lcm=32 * 12):
    remainder = num % lcm
    if remainder == 0:
        return num
    padding_needed = lcm - remainder
    padded_number = num + padding_needed
    return padded_number


@run_for_wormhole_b0()
@pytest.mark.parametrize("k", [8192])
@pytest.mark.parametrize("n", [1280])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("grid_size", [(8, 1)])
@pytest.mark.parametrize("tile_h", [16, 32])
@pytest.mark.parametrize("tile_w", [16, 32])
def test_matmul_in1_dram_sharded_tiny_tile(device, k, n, has_bias, grid_size, tile_h, tile_w):
    # PCC issue when height not equal to tile height
    m = tile_h
    if is_grayskull():
        n_padded = n
        num_banks = 8
    else:
        num_banks = 12
        n_padded = pad_to_dram_banks(n, tile_w, tile_w * num_banks)

    in0_shape = [1, 1, m, k]
    in1_shape = [1, 1, k, n]
    in1_shard_shape = [k, n_padded // num_banks]
    bias_shape = [1, 1, n]
    bias_shard_shape = [tile_h, n_padded // num_banks]
    num_cores = grid_size[0] * grid_size[1]

    in0_block_w = k // num_cores // 32
    out_block_h = m // tile_h
    out_block_w = n // num_cores // tile_w

    sharded_mem_config = ttnn.MemoryConfig(
        memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        buffer_type=ttnn.BufferType.L1,
    )

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()

    in0_memory_config = ttnn.create_sharded_memory_config(
        (1, 1, m, k),
        core_grid=ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    in0_t = ttnn.from_torch(
        in0,
        tile=ttnn.Tile((tile_h, 32)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_memory_config,
    )
    in1_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
    in1_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), in1_shard_grid)})
    in1_shard_spec = ttnn.ShardSpec(in1_shard_grid, in1_shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
    in1_memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec)
    in1_t = ttnn.from_torch(
        in1,
        tile=ttnn.Tile((32, tile_w)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in1_memory_config,
    )

    if has_bias:
        bias = torch.randn(bias_shape).bfloat16().float()
        bias_padded = bias.unsqueeze(2)
        bias_padded = torch.nn.functional.pad(bias_padded, (0, 0, 0, tile_h - bias_padded.size(2)), "constant", 0)
        bias_shard_grid = ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1)
        bias_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), bias_shard_grid)})
        bias_shard_spec = ttnn.ShardSpec(bias_shard_grid, bias_shard_shape, ttnn.ShardOrientation.ROW_MAJOR, False)
        bias_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, bias_shard_spec
        )
        bias_t = ttnn.from_torch(
            bias_padded,
            tile=ttnn.Tile((tile_h, tile_w)),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=bias_mem_config,
        )

    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w // 4,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fused_activation=None,
    )

    if is_grayskull():
        compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
        )
    else:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    if has_bias:
        output_t = ttnn.linear(
            in0_t,
            in1_t,
            bias=bias_t,
            program_config=program_config,
            memory_config=sharded_mem_config,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute_kernel_config,
            output_tile=ttnn.Tile([tile_h, 32]) if tile_h <= 16 else ttnn.Tile([tile_h, tile_w]),
        )
    else:
        output_t = ttnn.matmul(
            in0_t,
            in1_t,
            program_config=program_config,
            memory_config=sharded_mem_config,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute_kernel_config,
            output_tile=ttnn.Tile([tile_h, 32]) if tile_h <= 16 else ttnn.Tile([tile_h, tile_w]),
        )
    output_tensor = ttnn.to_torch(output_t)
    pt_out = in0 @ in1
    if has_bias:
        pt_out += bias

    assert_with_pcc(pt_out, output_tensor, 0.999)


@run_for_wormhole_b0()
@pytest.mark.parametrize("m", [1536])
@pytest.mark.parametrize("k", [1024])
@pytest.mark.parametrize("n", [3072])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("grid_size", [(8, 4)])
@pytest.mark.parametrize("tile_h", [16, 32])
@pytest.mark.parametrize("tile_w", [16, 32])
@pytest.mark.parametrize("in0_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
def test_matmul_2d_tiny_tile(device, m, k, n, has_bias, grid_size, tile_h, tile_w, in0_sharded, out_sharded):
    in0_shape = [1, 1, m, k]
    in1_shape = [1, 1, k, n]
    bias_shape = [1, 1, n]

    in0_block_w = k // grid_size[0] // 32
    out_block_h = m // grid_size[1] // tile_h
    out_block_w = n // grid_size[0] // tile_w
    out_subblock_h, out_subblock_w, _ = find_max_subblock(out_block_h, out_block_w)

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()

    if in0_sharded:
        in0_memory_config = ttnn.create_sharded_memory_config(
            (1, 1, m, k),
            core_grid=ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
    else:
        in0_memory_config = ttnn.L1_MEMORY_CONFIG
    in0_t = ttnn.from_torch(
        in0,
        tile=ttnn.Tile((tile_h, 32)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_memory_config,
    )
    in1_t = ttnn.from_torch(
        in1,
        tile=ttnn.Tile((32, tile_w)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if has_bias:
        bias = torch.randn(bias_shape).bfloat16().float()
        bias_padded = bias.unsqueeze(2)
        bias_padded = torch.nn.functional.pad(bias_padded, (0, 0, 0, tile_h - bias_padded.size(2)), "constant", 0)
        bias_t = ttnn.from_torch(
            bias_padded,
            tile=ttnn.Tile((tile_h, tile_w)),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        transpose_mcast=False,
        fused_activation=None,
    )

    if is_grayskull():
        compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
        )
    else:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
    if out_sharded:
        out_mem_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            buffer_type=ttnn.BufferType.L1,
        )
    else:
        out_mem_config = ttnn.L1_MEMORY_CONFIG
    if out_sharded:
        output_tile = ttnn.Tile([tile_h, 32]) if tile_h <= 16 else ttnn.Tile([tile_h, tile_w])
    else:
        output_tile = ttnn.Tile([tile_h, tile_w])
    if has_bias:
        output_t = ttnn.linear(
            in0_t,
            in1_t,
            bias=bias_t,
            program_config=program_config,
            memory_config=out_mem_config,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute_kernel_config,
            output_tile=output_tile,
        )
    else:
        output_t = ttnn.matmul(
            in0_t,
            in1_t,
            program_config=program_config,
            memory_config=out_mem_config,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute_kernel_config,
            output_tile=output_tile,
        )
    output_tensor = ttnn.to_torch(output_t)
    pt_out = in0 @ in1
    if has_bias:
        pt_out += bias

    assert_with_pcc(pt_out, output_tensor, 0.999)


@run_for_wormhole_b0()
@pytest.mark.parametrize("m", [256])
@pytest.mark.parametrize("k", [1024])
@pytest.mark.parametrize("n", [1024])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("grid_size", [(8, 4)])
@pytest.mark.parametrize("tile_h", [16, 32])
@pytest.mark.parametrize("tile_w", [16, 32])
@pytest.mark.parametrize("in0_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
def test_matmul_1d_tiny_tile(device, m, k, n, has_bias, grid_size, tile_h, tile_w, in0_sharded, out_sharded):
    in0_shape = [1, 1, m, k]
    in1_shape = [1, 1, k, n]
    bias_shape = [1, 1, n]

    num_cores = grid_size[0] * grid_size[1]

    in0_block_w = k // num_cores // 32
    out_block_h = m // tile_h
    out_block_w = n // num_cores // tile_w
    out_subblock_h, out_subblock_w, _ = find_max_subblock(out_block_h, out_block_w)

    in0 = torch.randn(in0_shape).bfloat16().float()
    in1 = torch.randn(in1_shape).bfloat16().float()

    if in0_sharded:
        in0_memory_config = ttnn.create_sharded_memory_config(
            (1, 1, m, k),
            core_grid=ttnn.CoreGrid(y=grid_size[1], x=grid_size[0]),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        )
    else:
        in0_memory_config = ttnn.L1_MEMORY_CONFIG
    in0_t = ttnn.from_torch(
        in0,
        tile=ttnn.Tile((tile_h, 32)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in0_memory_config,
    )
    in1_t = ttnn.from_torch(
        in1,
        tile=ttnn.Tile((32, tile_w)),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if has_bias:
        bias = torch.randn(bias_shape).bfloat16().float()
        bias_padded = bias.unsqueeze(2)
        bias_padded = torch.nn.functional.pad(bias_padded, (0, 0, 0, tile_h - bias_padded.size(2)), "constant", 0)
        bias_t = ttnn.from_torch(
            bias_padded,
            tile=ttnn.Tile((tile_h, tile_w)),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )

    if is_grayskull():
        compute_kernel_config = ttnn.GrayskullComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
        )
    else:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
    if out_sharded:
        out_mem_config = ttnn.MemoryConfig(
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            buffer_type=ttnn.BufferType.L1,
        )
    else:
        out_mem_config = ttnn.L1_MEMORY_CONFIG
    if out_sharded:
        output_tile = ttnn.Tile([tile_h, 32]) if tile_h <= 16 else ttnn.Tile([tile_h, tile_w])
    else:
        output_tile = ttnn.Tile([tile_h, tile_w])
    if has_bias:
        output_t = ttnn.linear(
            in0_t,
            in1_t,
            bias=bias_t,
            program_config=program_config,
            memory_config=out_mem_config,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute_kernel_config,
            output_tile=output_tile,
        )
    else:
        output_t = ttnn.matmul(
            in0_t,
            in1_t,
            program_config=program_config,
            memory_config=out_mem_config,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute_kernel_config,
            output_tile=output_tile,
        )
    output_tensor = ttnn.to_torch(output_t)
    pt_out = in0 @ in1
    if has_bias:
        pt_out += bias

    assert_with_pcc(pt_out, output_tensor, 0.999)


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


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "in0_dtype, in1_dtype, num_activation_cores, num_compute_cores, has_bias, config, M, K, N",
    [
        (ttnn.bfloat16, ttnn.bfloat4_b, 24, 24, False, "tg_llama_FF1", None, None, None),
        (ttnn.bfloat16, ttnn.bfloat4_b, 24, 24, True, "tg_llama_FF1", None, None, None),
        (ttnn.bfloat16, ttnn.bfloat4_b, 2, 2, True, None, 32, 1024, 2048),
    ],
)
def test_matmul_in0_in1_bias_sharded(
    device, in0_dtype, in1_dtype, num_activation_cores, num_compute_cores, has_bias, config, M, K, N
):
    torch.manual_seed(0)

    def padded_size_per_device_for_num_cores(size, num_devices, num_cores):
        padded_size = math.ceil(size / num_devices / num_cores / TILE_SIZE) * num_cores * TILE_SIZE
        return padded_size

    def core_grid_size_for_num_cores(num_cores):
        assert num_cores < 8 or num_cores % 8 == 0
        x = min(num_cores, 8)
        y = max(1, num_cores // 8)
        core_grid = (x, y)
        return core_grid

    def core_range_for_num_cores(num_cores):
        core_grid = core_grid_size_for_num_cores(num_cores)
        core_range = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(core_grid[0] - 1, core_grid[1] - 1),
                ),
            }
        )
        return core_range

    TILE_SIZE = 32

    if config == "tg_llama_FF1":
        assert M is None and K is None and N is None, "Cannot specify config and any of M, K, N"
        cluster_size = (4, 8)
        hidden_size = 8192
        ff_size = 28 * 1024
        M = 32
        K = padded_size_per_device_for_num_cores(hidden_size, cluster_size[0], num_activation_cores)
        N = ff_size // cluster_size[1]
        N_padded = padded_size_per_device_for_num_cores(ff_size, cluster_size[1], num_compute_cores)
    else:
        assert M is not None and K is not None and N is not None, "Must specify M, K, N"
        N_padded = N

    # Weights
    mem_config_weights = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            core_range_for_num_cores(num_compute_cores),
            [
                K,
                N_padded // num_compute_cores,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    if has_bias:
        mem_config_bias = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                core_range_for_num_cores(num_compute_cores),
                [
                    32,
                    N_padded // num_compute_cores,
                ],
                ttnn.ShardOrientation.ROW_MAJOR,
                False,
            ),
        )

    # Input
    mem_config_input = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            core_range_for_num_cores(num_activation_cores),
            [
                M,
                K // num_activation_cores,
            ],
            ttnn.ShardOrientation.ROW_MAJOR,
            False,
        ),
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    input_shape = [1, 1, M, K]
    input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    tt_input_tensor = ttnn.as_tensor(
        input_tensor,
        dtype=in0_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config_input,
    )

    weights_tensor = torch.randn([1, 1, K, N], dtype=torch.bfloat16)
    weight_tt = ttnn.as_tensor(
        weights_tensor,
        dtype=in1_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config_weights,
    )

    if has_bias:
        bias_tensor = torch.randn([1, 1, 1, N], dtype=torch.bfloat16) * 2.0
        bias_tt = ttnn.as_tensor(
            bias_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=mem_config_bias,
        )

    mm_core_grid = core_grid_size_for_num_cores(num_compute_cores)
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=mm_core_grid,
        in0_block_w=K // num_compute_cores // 32,  # K // num_cores // 32; how much inner dim you take each time
        out_subblock_h=1,  # Must be divisible by per_core_M
        out_subblock_w=1,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4; per_core_N = 5 so can only use 1 here
        per_core_M=M // 32,  # M / TILE_HEIGHT / Grid_Size
        per_core_N=N_padded // 32 // (mm_core_grid[0] * mm_core_grid[1]),
        mcast_in0=True,
        fused_activation=None,
        fuse_batch=True,
    )

    if has_bias:
        tt_matmul_out_tensor = ttnn.linear(
            tt_input_tensor,
            weight_tt,
            bias=bias_tt,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )
    else:
        tt_matmul_out_tensor = ttnn.matmul(
            tt_input_tensor,
            weight_tt,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
        )

    tt_mm_out = ttnn.from_device(tt_matmul_out_tensor)
    tt_mm_out = ttnn.to_torch(tt_mm_out)

    # Torch reference
    matmul_output = torch.matmul(input_tensor, weights_tensor)
    if has_bias:
        matmul_output = matmul_output + bias_tensor

    assert_with_pcc(matmul_output, tt_mm_out, pcc=0.993)


@pytest.mark.parametrize("M", [32, 128])
@pytest.mark.parametrize("K", [32, 128])
@pytest.mark.parametrize("N", [32, 128])
def test_alternating_dst_sync_mode_matmul(device, M, K, N):
    torch.manual_seed(0)
    torch_input_tensor_a = torch.randn([1, 1, M, K], dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn([1, 1, K, N], dtype=torch.bfloat16)
    torch_output_tensor = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    # Half sync mode
    output1 = ttnn.matmul(input_tensor_a, input_tensor_b, core_grid=ttnn.CoreGrid(y=4, x=4))
    # Full sync mode
    output2 = ttnn.matmul(input_tensor_a, input_tensor_b)
    # Half sync mode
    output3 = ttnn.matmul(input_tensor_a, input_tensor_b, core_grid=ttnn.CoreGrid(y=4, x=4))

    pcc = 0.99
    output_tensor = ttnn.to_torch(output1)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
    output_tensor = ttnn.to_torch(output2)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
    output_tensor = ttnn.to_torch(output3)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)


def test_interleaved_input_sharded_output_matmul(device):
    torch.manual_seed(0)
    pcc = 0.99
    # Width sharded
    torch_input_tensor_a = torch.randn([1, 1, 32, 32], dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn([1, 1, 32, 256], dtype=torch.bfloat16)
    torch_output_tensor = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    out_mem_config = ttnn.create_sharded_memory_config(
        shape=(32, 256),
        core_grid=ttnn.CoreGrid(x=1, y=8),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    output1 = ttnn.matmul(input_tensor_a, input_tensor_b, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output1)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)

    # Block sharded
    out_mem_config = ttnn.create_sharded_memory_config(
        shape=(32, 256),
        core_grid=ttnn.CoreGrid(x=1, y=8),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    output2 = ttnn.matmul(input_tensor_a, input_tensor_b, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output2)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)

    # Height sharded
    torch_input_tensor_a = torch.randn([1, 1, 256, 32], dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn([1, 1, 32, 32], dtype=torch.bfloat16)
    torch_output_tensor = torch.matmul(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    out_mem_config = ttnn.create_sharded_memory_config(
        shape=(256, 32),
        core_grid=ttnn.CoreGrid(x=8, y=1),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    output3 = ttnn.matmul(input_tensor_a, input_tensor_b, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output3)
    assert_with_pcc(torch_output_tensor, output_tensor, pcc=pcc)
