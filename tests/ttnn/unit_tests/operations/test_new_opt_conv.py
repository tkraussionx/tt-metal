from loguru import logger

import torch
import pytest
from models.utility_functions import skip_for_wormhole_b0, skip_for_grayskull, is_grayskull, is_wormhole_b0
from tests.ttnn.utils_for_testing import (
    assert_with_pcc,
    check_with_pcc,
    check_with_pcc_without_tensor_printout,
    update_process_id,
)
from tt_lib.utils import (
    _nearest_y,
)
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores
import ttnn
import tt_lib
import math
import os


@pytest.mark.parametrize("device_l1_small_size", [16384], indirect=True)
def test_new_conv(device):
    print("OptimizedConvNew1 Python Side")

    tt_bias_tensor = None
    reader_patterns_cache = {}
    use_shallow_conv_variant = (False,)
    activations_dtype = ttnn.bfloat16
    weights_dtype = ttnn.bfloat8_b
    # update_process_id()
    batch_size, input_channel, input_height, input_width = 2, 1152, 16, 8
    weight_batch, ncores = 1152, 36
    filter_height, filter_width = 3, 3
    input_shape = [1, 1, batch_size * input_height * input_width, input_channel]
    weight_shape = [1, 1, weight_batch * filter_height * filter_width, input_channel]
    torch_input = torch.rand(input_shape, dtype=torch.bfloat16) * 2 - 1
    torch_weight = torch.rand(weight_shape, dtype=torch.bfloat16) * 2 - 1
    for i in range(batch_size * input_height * input_width):
        for j in range(input_channel):
            torch_input[0, 0, i, j] = j + 0.01 * i
    tt_input = ttnn.from_torch(torch_input, device=device, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tt_weight = ttnn.from_torch(
        torch_weight,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_2d_height = batch_size * input_height * input_width
    shard_height = batch_size * input_height * input_width
    print("init complted")
    input_2d_width_padded = _nearest_y(input_channel, ncores * 32)
    shard_width = math.ceil(input_2d_width_padded / ncores)
    shard_orientation = ttnn.experimental.tensor.ShardOrientation.ROW_MAJOR
    tensor_memory_layout = ttnn.types.TensorMemoryLayout.WIDTH_SHARDED
    # tensor_memory_layout = ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED
    shard_grid = get_shard_grid_from_num_cores(ncores, device)
    shard_spec = ttnn.experimental.tensor.ShardSpec(shard_grid, (shard_height, shard_width), shard_orientation, False)
    in_sharded_mem_config = ttnn.MemoryConfig(tensor_memory_layout, ttnn.types.BufferType.L1, shard_spec)
    logger.debug(f"shard_memory_layout={in_sharded_mem_config}")
    input_tensor = ttnn.to_memory_config(tt_input, memory_config=in_sharded_mem_config)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    ##op computation
    # ttnn.dump_device_memory_state(device, prefix='pre_')
    stride_h, stride_w, pad_h, pad_w = 1, 1, 1, 1

    # output = ttnn.optimized_conv_new(
    #     input_tensor,
    #     tt_weight,
    #     device,
    #     [3, 3, 1, 1, 1, 1],
    #     weight_batch,
    #     input_height,
    #     input_width,
    #     True,
    #     True,
    #     ttnn.MathFidelity.HiFi4,
    #     10,
    # )

    # return
    # ttnn.dump_device_memory_state(device, prefix='shwe_')
    # print(output)
    # ttnn.opimized_abc([1, 1, 1, 1], 1024, True, True)
    # ttnn.opimized_conv_new_1(tt_input, tt_weight, [1, 1, 1, 1], 1024, True, True)

    conv_config = ttnn.Conv2dConfig(
        dtype=activations_dtype,
        weights_dtype=weights_dtype,
        math_fidelity=ttnn.MathFidelity.HiFi4,
        activation=None,
        conv_shard_scheme="WIDTH",
        input_channels_alignment=(16 if use_shallow_conv_variant else 32),
        deallocate_activation=False,
    )

    [tt_output_tensor_on_device, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=tt_input,
        weight_tensor=tt_weight,
        in_channels=input_channel,
        out_channels=weight_batch,
        device=device,
        bias_tensor=tt_bias_tensor,
        kernel_size=(filter_height, filter_width),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        batch_size=batch_size,
        input_height=input_height,
        input_width=input_width,
        conv_config=conv_config,
        conv_op_cache=reader_patterns_cache,
        reshard_if_not_optimal=False,
        debug=False,
    )

    print(tt_output_tensor_on_device)
