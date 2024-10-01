// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations {
namespace data_movement {

struct IndexedFill {
    static Tensor invoke(
        const Tensor& batch_id,
        const Tensor& input_tensor_a,
        const Tensor& input_tensor_b,
        int64_t dim,
        const std::optional<MemoryConfig>& memory_config);
};

}  // namespace indexed_fill
}  // namespace operations

constexpr auto indexed_fill =
    ttnn::register_operation_with_auto_launch_op<"ttnn::indexed_fill", ttnn::operations::data_movement::IndexedFill>();

}  // namespace ttnn
