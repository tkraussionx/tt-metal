// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/pool/avgpool/device/avg_pool2d_device_op.hpp"

namespace ttnn::operations::experimental::pool {

AvgPool2D::program_factory_t AvgPool2D::select_program_factory(
    const AvgPool2D::operation_attributes_t& op_attr, const AvgPool2D::tensor_args_t& tensors) {
    return MultiCore{};
}

void AvgPool2D::validate_on_program_cache_miss(
    const AvgPool2D::operation_attributes_t& op_attr, const AvgPool2D::tensor_args_t& tensors) {}

void AvgPool2D::validate_on_program_cache_hit(
    const AvgPool2D::operation_attributes_t& op_attr, const AvgPool2D::tensor_args_t& tensors) {}

AvgPool2D::shape_return_value_t AvgPool2D::compute_output_shapes(
    const AvgPool2D::operation_attributes_t& op_attr, const AvgPool2D::tensor_args_t& inputs) {
    const auto& input = inputs.input_tensor_;
    const auto& sliding_window_config = op_attr.sliding_window_config_;
    const bool is_out_tiled = op_attr.output_dtype_ == DataType::BFLOAT8_B;

    std::vector<uint32_t> out_dims = sliding_window_config.get_output_shape().logical_shape().as_vector();
    out_dims[3] = input.get_logical_shape()[3];

    const auto num_cores_nhw = sliding_window_config.num_cores_nhw;
    const auto out_nhw_dims = std::vector<uint32_t>({1, 1, out_dims[0] * out_dims[1] * out_dims[2], out_dims[3]});
    const auto out_padded_dims = std::vector<uint32_t>(
        {1,
         1,
         tt::round_up(out_nhw_dims[2], num_cores_nhw * (is_out_tiled ? tt::constants::TILE_HEIGHT : 1)),
         tt::round_up(out_nhw_dims[3], (out_nhw_dims[3] <= 16) ? 16 : tt::constants::TILE_WIDTH)});

    // TODO: padding value
    const auto out_padding = Padding(
        {{0, 0}, {0, 0}, {0, out_padded_dims[2] - out_nhw_dims[2]}, {0, out_padded_dims[3] - out_nhw_dims[3]}},
        Padding::PadValue::Zero);

    return Shape(out_padded_dims, out_padding);
}

Tensor AvgPool2D::create_output_tensors(
    const AvgPool2D::operation_attributes_t& op_attr, const AvgPool2D::tensor_args_t& inputs) {
    const auto& input = inputs.input_tensor_;
    const auto& sliding_window_config = op_attr.sliding_window_config_;
    const auto& out_dtype = op_attr.output_dtype_;
    const auto& out_shape = compute_output_shapes(op_attr, inputs);

    MemoryConfig out_mem_config = input.memory_config();
    // update the shard spec to match the output shape
    const auto& out_padded_shape = out_shape.padded_shape();
    const auto& shard_spec = out_mem_config.shard_spec.value();
    const uint32_t out_shard_height_padded = out_padded_shape[2] / sliding_window_config.num_cores_nhw;
    const uint32_t out_shard_width_padded = out_padded_shape[3];
    out_mem_config.shard_spec = ShardSpec(
        shard_spec.grid,
        {out_shard_height_padded, out_shard_width_padded},
        ShardOrientation::ROW_MAJOR,
        /*halo= */ false);

    tt::log_info("output shape {} memory config {}", out_shape, out_mem_config);

    return create_device_tensor(out_shape, out_dtype, input.get_layout(), input.device(), out_mem_config);
}

tt::stl::hash::hash_t AvgPool2D::compute_program_hash(
    const AvgPool2D::operation_attributes_t& op_attr, const AvgPool2D::tensor_args_t& tensors) {
    const auto& input = tensors.input_tensor_;
    return operation::hash_operation<AvgPool2D>(
        op_attr.sliding_window_config_.get_hash(),
        op_attr.output_dtype_,
        input.get_shape(),
        input.memory_config(),
        input.dtype());
}

operation::OpPerformanceModel AvgPool2D::create_op_performance_model(
    const AvgPool2D::operation_attributes_t& op_attr, const AvgPool2D::tensor_args_t& inputs, const Tensor& output) {
    const auto& input = inputs.input_tensor_;
    return operation::OpPerformanceModel{{input}, {output}, 1};
}

std::tuple<AvgPool2D::operation_attributes_t, AvgPool2D::tensor_args_t> AvgPool2D::invoke(
    const Tensor& input_tensor,
    const sliding_window::SlidingWindowConfig& sliding_window_config,
    DataType output_dtype) {
    return {
        operation_attributes_t{.sliding_window_config_ = sliding_window_config, .output_dtype_ = output_dtype},
        tensor_args_t{input_tensor}};
}

}  // namespace ttnn::operations::experimental::pool
