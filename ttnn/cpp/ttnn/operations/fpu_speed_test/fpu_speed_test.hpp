// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/fpu_speed_test/device/fpu_speed_test_device_operation.hpp"

namespace ttnn {
namespace operations {
namespace fpu_speed_test {

struct FPUSpeedTest {
    static Tensor invoke(
        uint8_t queue_id, uint32_t num_tiles, bool fp32_dest_acc_en, const Tensor& dummy) {
        return ttnn::prim::fpu_speed_test(queue_id, num_tiles, fp32_dest_acc_en, dummy);
    }
};

}  // namespace fpu_speed_test
}  // namespace operations

constexpr auto fpu_speed_test = ttnn::
    register_operation_with_auto_launch_op<"ttnn::fpu_speed_test", ttnn::operations::fpu_speed_test::FPUSpeedTest>();

}  // namespace ttnn
