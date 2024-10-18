// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/pool/avgpool/device/avg_pool2d_device_op.hpp"

namespace ttnn::operations::experimental::pool {

AvgPool2D::MultiCore::cached_program_t AvgPool2D::MultiCore::create(
    const AvgPool2D::operation_attributes_t& op_attrs,
    const AvgPool2D::tensor_args_t& inputs,
    AvgPool2D::tensor_return_value_t& output_tensor) {
    tt::tt_metal::Program program{};
    const auto& input = inputs.input_tensor_;
    return {std::move(program), {}};
}

void AvgPool2D::MultiCore::override_runtime_arguments(
    AvgPool2D::MultiCore::cached_program_t& cached_program,
    const AvgPool2D::operation_attributes_t& op_attrs,
    const AvgPool2D::tensor_args_t& inputs,
    AvgPool2D::tensor_return_value_t& output_tensor) {}

}  // namespace ttnn::operations::experimental::pool
