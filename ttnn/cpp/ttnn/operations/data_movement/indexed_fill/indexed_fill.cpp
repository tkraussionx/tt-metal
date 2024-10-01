// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/indexed_fill/device/indexed_fill_device_operation.hpp"
#include "indexed_fill.hpp"

namespace ttnn::operations::data_movement{

Tensor IndexedFill::invoke(
    const Tensor& batch_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    int64_t dim,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::indexed_fill(batch_id, input_tensor_a, input_tensor_b, dim, memory_config);
}

}  // namespace ttnn::operations::data_movement
