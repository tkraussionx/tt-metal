# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from loguru import logger
import math

from tt_eager.tt_dnn.op_library.sliding_window_op_infra.halo_config_generation_for_sliding_window_op import (
    trace_sliding_window_op_to_generate_data_top_left_indices_and_pad_metadata,
    decompose_sliding_window_op_into_shards_and_generate_tensor_metadata,
    generate_untilize_with_halo_kernel_configs,
)
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.halo_config_validation_for_conv_op import (
    construct_input_padded_tensor,
    validate_input_padded_tensor_and_data_top_left_indices_and_pad_metadata,
    construct_utwh_output_shards,
    validate_utwh_output_shards_and_req_ds_input_shard_start_end,
    validate_tensor_metadata,
    validate_untilize_with_halo_kernel_configs,
)
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.sliding_window_op_config_generation_and_validation import (
    generate_sliding_window_op_sharded_input_top_left_indices,
    validate_downsample_sharded_input_top_left_indices,
    validate_max_pool_sharded_input_top_left_indices,
)
from tt_eager.tt_dnn.op_library.sliding_window_op_infra.tt_py_untilize_with_halo import TTPyUntilizeWithHalo
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_allclose_and_pcc, comp_pcc
from tt_lib.utils import _nearest_y
import tt_lib as ttl


@pytest.mark.parametrize(
    "batch_size, stride_h, stride_w, input_c, input_h, input_w, num_cores_nhw",
    (
        # resnet50 downsamples
        # batch size 8
        (8, 2, 2, 256, 56, 56, 98),
        (8, 2, 2, 512, 28, 28, 10),
        (8, 2, 2, 1024, 14, 14, 7),
        # batch size 16
        (16, 2, 2, 256, 56, 56, 98),
        (16, 2, 2, 512, 28, 28, 11),
        (16, 2, 2, 1024, 14, 14, 9),
        # batch size 20
        (20, 2, 2, 256, 56, 56, 98),
        (20, 2, 2, 512, 28, 28, 12),
        (20, 2, 2, 1024, 14, 14, 11),
    ),
)
def test_generate_all_configs_and_references_for_downsample(
    batch_size, stride_h, stride_w, input_c, input_h, input_w, num_cores_nhw
):
    torch.set_printoptions(threshold=10000, edgeitems=50, linewidth=400)
    assert stride_h == stride_w
    # Construct input tensor
    input_tensor = []
    input_nchw_shape = [batch_size, input_c, input_h, input_w]
    input_volume = np.prod(input_nchw_shape)
    input_nhw_size = batch_size * input_h * input_w
    output_h = math.ceil(input_h / stride_h)
    output_w = math.ceil(input_w / stride_h)
    output_nhw_size = batch_size * output_h * output_w

    input_nhw_size_to_shard_evenly = _nearest_y(input_nhw_size, num_cores_nhw * 32)
    input_shard_nhw_size = (int)(input_nhw_size_to_shard_evenly / num_cores_nhw)
    output_nhw_size_to_shard_evenly = _nearest_y(output_nhw_size, num_cores_nhw * 32)
    output_shard_nhw_size = (int)(output_nhw_size_to_shard_evenly / num_cores_nhw)

    logger.info(f"downsample input shard nhw size={input_shard_nhw_size}")
    logger.info(f"downsample output shard nhw size={output_shard_nhw_size}")

    # Initialize tensor with data
    # Inserting sequential integer data
    # for val in range(1, input_volume + 1):
    #     input_tensor.append(val)
    # input_pyt_tensor = torch.tensor(input_tensor)
    input_pyt_tensor = torch.rand(input_volume, dtype=torch.bfloat16)
    input_tensor = input_pyt_tensor.reshape(-1).tolist()
    input_pyt_tensor = torch.reshape(input_pyt_tensor, input_nchw_shape)

    # run downsample in pytorch with a maxpool op with 1x1 window
    out_golden_pyt_tensor = torch.nn.functional.max_pool2d(input_pyt_tensor, 1, stride=stride_h)

    # Generate following configs by tracing op with sliding window -
    logger.info("Trace sliding window op and generate following configs - pad_metadata and data_top_left_indices.")
    pad_metadata, data_top_left_indices = trace_sliding_window_op_to_generate_data_top_left_indices_and_pad_metadata(
        (input_c, input_c, 1, 1, stride_h, stride_w, 0, 0, 1, 1), input_nchw_shape
    )
    # sanity check - no padding in downsample
    assert not any(pad_metadata)

    # Generate more configs -
    logger.info(
        "Decompose sliding window op into shards and generate the required sliding window op input shard start/end stick indices and tensor metadata."
    )
    (
        req_sliding_window_op_input_shard_start_end,
        tensor_metadata,
    ) = decompose_sliding_window_op_into_shards_and_generate_tensor_metadata(
        data_top_left_indices,
        pad_metadata,
        input_w,
        output_shard_nhw_size,
        input_shard_nhw_size,
        num_cores_nhw,
        1,
        1,
    )

    # Permute input tensor from nchw shape to nhwc shape
    input_tensor_nchw = np.reshape(input_tensor, input_nchw_shape)
    input_tensor_nhwc = np.transpose(input_tensor_nchw, (0, 2, 3, 1))
    input_tensor_nhwc = np.reshape(input_tensor_nhwc, (np.prod(input_nchw_shape)))
    logger.info("Validate required downsample input shard start/end stick indices")
    golden_untilize_with_halo_output_shards = construct_utwh_output_shards(
        input_tensor_nhwc, input_nchw_shape, req_sliding_window_op_input_shard_start_end
    )

    validate_utwh_output_shards_and_req_ds_input_shard_start_end(
        input_nchw_shape,
        out_golden_pyt_tensor,
        data_top_left_indices,
        golden_untilize_with_halo_output_shards,
        req_sliding_window_op_input_shard_start_end,
    )

    logger.info("Validate tensor metadata")
    untilize_with_halo_input_shards = validate_tensor_metadata(
        input_tensor_nchw.reshape(-1).tolist(),
        input_nchw_shape,
        input_shard_nhw_size,
        tensor_metadata,
        req_sliding_window_op_input_shard_start_end,
        golden_untilize_with_halo_output_shards,
    )

    # Generate and validate the final untilize with halo configs here
    logger.info("Generate untilize with halo kernel configs")
    (
        local_data,
        local_pad,
        ll_data,
        l_data,
        r_data,
        rr_data,
        src_start_idx,
        local_data_nsegments_per_core,
        local_pad_nsegments_per_core,
        ll_data_nsegments_per_core,
        l_data_nsegments_per_core,
        r_data_nsegments_per_core,
        rr_data_nsegments_per_core,
        max_out_nsticks_per_core,
    ) = generate_untilize_with_halo_kernel_configs(tensor_metadata, req_sliding_window_op_input_shard_start_end)

    logger.info("Validate reshards")
    validate_untilize_with_halo_kernel_configs(
        golden_untilize_with_halo_output_shards,
        untilize_with_halo_input_shards,
        req_sliding_window_op_input_shard_start_end,
        local_data,
        local_pad,
        ll_data,
        l_data,
        r_data,
        rr_data,
        src_start_idx,
        local_data_nsegments_per_core,
        local_pad_nsegments_per_core,
        ll_data_nsegments_per_core,
        l_data_nsegments_per_core,
        r_data_nsegments_per_core,
        rr_data_nsegments_per_core,
        max_out_nsticks_per_core,
    )

    # Generate sliding window op config -
    logger.info("Generate downsample configs - top left positioned indices for input shards")
    sliding_window_op_sharded_input_top_left_indices = generate_sliding_window_op_sharded_input_top_left_indices(
        data_top_left_indices, req_sliding_window_op_input_shard_start_end
    )
    logger.info("Validate config indices for downsample")
    validate_downsample_sharded_input_top_left_indices(
        golden_untilize_with_halo_output_shards,
        out_golden_pyt_tensor,
        sliding_window_op_sharded_input_top_left_indices,
    )
