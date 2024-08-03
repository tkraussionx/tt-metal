
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/moreh_relu_device_operation.hpp"

namespace ttnn::operations::moreh_eltwise {

struct MorehRelu {
  static Tensor operator()(uint8_t queue_id, const Tensor &input_tensor,
                           const std::optional<Tensor> &output_tensor,
                           const uint8_t which_relu, const float bound) {
    return ttnn::device_operation::run<MorehReluDeviceOperation>(
        queue_id,
        MorehReluDeviceOperation::operation_attributes_t{
            .which_relu = which_relu, .bound = bound},
        MorehReluDeviceOperation::tensor_args_t{input_tensor, output_tensor});
  }
};

} // namespace ttnn::operations::moreh_eltwise

namespace ttnn {

constexpr auto moreh_relu = ttnn::register_operation_with_auto_launch_op<
    "ttnn::moreh_relu", operations::moreh_eltwise::MorehRelu>();

} // namespace ttnn
