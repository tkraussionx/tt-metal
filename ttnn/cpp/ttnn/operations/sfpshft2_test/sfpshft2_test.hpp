// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/sfpshft2_test/device/sfpshft2_test_device_operation.hpp"

namespace ttnn {
namespace operations {
namespace sfpshft2_test {

struct SFPSHFT2Test {
    static Tensor invoke(
        uint8_t queue_id, const Tensor& input, const Tensor& output) {
        return ttnn::prim::sfpshft2_test(queue_id, input, output);
    }
};

}  // namespace sfpshft2_test
}  // namespace operations

constexpr auto sfpshft2_test = ttnn::
    register_operation_with_auto_launch_op<"ttnn::sfpshft2_test", ttnn::operations::sfpshft2_test::SFPSHFT2Test>();

}  // namespace ttnn
