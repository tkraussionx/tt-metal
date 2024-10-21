// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/pool/avgpool/device/avg_pool2d_device_op.hpp"

namespace ttnn::operations::experimental::pool {

AvgPool2D::MultiCore::cached_program_t AvgPool2D::MultiCore::create(
    const AvgPool2D::operation_attributes_t& op_attrs,
    const AvgPool2D::tensor_args_t& inputs,
    AvgPool2D::tensor_return_value_t& output_tensor) {
    const auto& input = inputs.input_tensor_;
    const auto& out_mem_config = output_tensor.memory_config();
    const auto& sliding_window_config = op_attrs.sliding_window_config_;

    tt::tt_metal::Program program{};

    const auto parallel_config = sliding_window::ParallelConfig{
        .grid = input.shard_spec().value().grid,
        .shard_scheme = input.memory_config().memory_layout,
        .shard_orientation = input.shard_spec().value().orientation,
    };

    const auto out_shape = sliding_window_config.get_output_shape();
    const uint32_t out_h = out_shape[1];
    const uint32_t out_w = out_shape[2];

    bool is_block_sharded = input.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED;

    const auto pad_metadata = sliding_window::generate_pad_metadata(sliding_window_config);
    const auto op_trace_metadata = sliding_window::generate_op_trace_metadata(sliding_window_config);
    const auto shard_boundaries = sliding_window::generate_shard_boundaries(sliding_window_config, op_trace_metadata);
    const auto top_left_indices =
        sliding_window::generate_sliding_window_op_config(op_trace_metadata, shard_boundaries, false, false);
    const auto reader_indices =
        sliding_window::construct_on_host_config_tensor(top_left_indices, sliding_window_config, parallel_config);
    const auto reader_indices_on_device =
        sliding_window::move_config_tensor_to_device(reader_indices, parallel_config, is_block_sharded, input.device());
    tt::tt_metal::detail::AddConfigBuffer(program, reader_indices_on_device.device_buffer());

    tt::log_info("reader_indices shape: {}", reader_indices.shape());

    return {std::move(program), {}};
}

void AvgPool2D::MultiCore::override_runtime_arguments(
    AvgPool2D::MultiCore::cached_program_t& cached_program,
    const AvgPool2D::operation_attributes_t& op_attrs,
    const AvgPool2D::tensor_args_t& inputs,
    AvgPool2D::tensor_return_value_t& output_tensor) {}

}  // namespace ttnn::operations::experimental::pool
