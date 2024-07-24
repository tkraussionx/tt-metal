// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/tilize_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct ExecuteTilize {
    static ttnn::Tensor execute_on_worker_thread(
        QueueId queue_id,
        const ttnn::Tensor &input_tensor,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        std::optional<DataType> output_dtype = std::nullopt,
        bool use_multicore = false) {
        return operation::run(
                   Tilize{
                       memory_config.value_or(input_tensor.memory_config()),
                       output_dtype.value_or(input_tensor.get_dtype()),
                       use_multicore},
                   {input_tensor},
                   {},
                   {},
                   queue_id)
            .at(0);
    }
};

}  // namespace operations::data_movement

constexpr auto tilize = ttnn::register_operation<ttnn::operations::data_movement::ExecuteTilize>("ttnn::tilize");

}  // namespace ttnn
