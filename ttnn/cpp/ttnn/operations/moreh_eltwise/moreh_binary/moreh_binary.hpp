// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/moreh_binary_device_operation.hpp"

namespace ttnn::operations::moreh_eltwise {

struct MorehBinary {
  static Tensor operator()(uint8_t queue_id, const Tensor &input_tensor0,
                           const Tensor &input_tensor1,
                           const std::optional<Tensor> &output_tensor,
                           const float scalar0, const float scalar1,
                           const uint8_t program_selector) {
    return ttnn::device_operation::run<MorehBinaryDeviceOperation>(
        queue_id,
        MorehBinaryDeviceOperation::operation_attributes_t{
            .scalar0 = scalar0, .scalar1 = scalar1, .program_selector = program_selector},
        MorehBinaryDeviceOperation::tensor_args_t{input_tensor0, input_tensor1,
                                                  output_tensor});
  }
};

} // namespace ttnn::operations::moreh_eltwise

namespace ttnn {

constexpr auto moreh_binary = ttnn::register_operation_with_auto_launch_op<
    "ttnn::moreh_binary", operations::moreh_eltwise::MorehBinary>();

} // namespace ttnn
