
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/moreh_relu_device_operation.hpp"

namespace ttnn::operations::moreh_eltwise {

struct MorehRelu {
  static Tensor operator()(uint8_t queue_id, const Tensor &input_tensor,
                           const std::optional<const Tensor> &output_tensor,
                           const bool do_max_relu, const uint32_t max) {
    return ttnn::device_operation::run<MorehReluDeviceOperation>(
        queue_id,
        MorehReluDeviceOperation::operation_attributes_t{
            .do_max_relu = do_max_relu, .max = max},
        MorehReluDeviceOperation::tensor_args_t{input_tensor, output_tensor});
  }
};

} // namespace ttnn::operations::moreh_eltwise

namespace ttnn {

constexpr auto moreh_relu =
    ttnn::register_operation<"ttnn::moreh_relu",
                             operations::moreh_eltwise::MorehRelu>();

} // namespace ttnn
