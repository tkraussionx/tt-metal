// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "max_pool.hpp"

namespace ttnn::operations {
namespace pool {

// maxpool macro-op
Tensor max_pool2d_new(const Tensor& input_tensor,
                             uint32_t batch_size,
                             uint32_t input_h,
                             uint32_t input_w,
                             uint32_t channels,
                             std::array<uint32_t, 2> kernel_size,
                             std::array<uint32_t, 2> stride,
                             std::array<uint32_t, 2> padding,
                             std::array<uint32_t, 2> dilation,
                             ttnn::Device& device) {

    TT_FATAL(input_tensor.storage_type() == ttnn::DEVICE_STORAGE_TYPE, "Input tensor must be on device.");
    MemoryConfig memory_config = input_tensor.memory_config();

    // const auto shard_grid = memory_config.shard_spec.value().grid;
    // const auto shard_scheme = memory_config.memory_layout;
    // const auto shard_orientation = memory_config.shard_spec.value().orientation;

    // TT_FATAL(shard_scheme == TensorMemoryLayout::HEIGHT_SHARDED, "Only height sharded tensors are supported.");
    // TT_FATAL(shard_orientation == ShardOrientation::ROW_MAJOR, "Only row major orientation is supported.");

    ParallelConfig parallel_config = conv2d::determine_parallel_config(
                                        true,       // height sharding
                                        batch_size,
                                        0,          // in_channels -- not used
                                        input_h,
                                        input_w,
                                        0,          // out_channels -- not used
                                        device,
                                        ShardOrientation::ROW_MAJOR);
    uint32_t num_cores_nhw = conv2d::get_num_cores_nhw_from_parallel_config(parallel_config);

    auto sharded_memory_config = memory_config;
    auto sharded_input_tensor = input_tensor;
    if (!memory_config.shard_spec.has_value()) {
        TT_FATAL(memory_config.shard_spec.has_value(), "Shard spec is not set for the input tensor.");
        // sharded_memory_config = conv2d::create_sharded_memory_config_from_parallel_config(input_tensor.shape(), parallel_config, 1);
        // sharded_input_tensor = ttnn::to_memory_config(input_tensor, sharded_memory_config);
    }

    SlidingWindowConfig sliding_window_config = SlidingWindowConfig(batch_size,
                                                                    input_h, input_w,
                                                                    kernel_size.at(0), kernel_size.at(1),
                                                                    stride.at(0), stride.at(1),
                                                                    padding.at(0), padding.at(1),
                                                                    dilation.at(0), dilation.at(1),
                                                                    num_cores_nhw,
                                                                    parallel_config.grid);
    uint32_t neg_inf_pad_val = 0xf7ff;

    auto haloed_tensor = ttnn::operations::halo::halo_op(sharded_input_tensor, sliding_window_config, neg_inf_pad_val, false, parallel_config.shard_orientation == ShardOrientation::COL_MAJOR, 0, memory_config);
    return max_pool2d_internal(haloed_tensor, sliding_window_config, memory_config);
}

}  // namespace pool
}  // namespace ttnn::operations
