// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// #include "tt_eager/tt_dnn/op_library/layernorm/layernorm_op.hpp"
#include "ttnn/cpp/ttnn/operations/normalization/layernorm/device/layernorm_op.hpp"

namespace ttnn {
namespace operations::normalization {

struct RMSNorm {

    static inline ttnn::Tensor execute_on_worker_thread(
        const ttnn::Tensor& input_tensor,
        float epsilon = 1e-12,
        const std::optional<const ttnn::Tensor>& weight = std::nullopt,
        const std::optional<const ttnn::Tensor>& bias = std::nullopt,
        const std::optional<const ttnn::Tensor>& residual_input_tensor = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<const LayerNormProgramConfig>& program_config = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) {

        if (residual_input_tensor.has_value()) {
            return ttnn::operations::normalization::add_rmsnorm(
                input_tensor, residual_input_tensor.value(), epsilon, weight, bias, memory_config.value_or(input_tensor.memory_config()), program_config.value_or(LayerNormDefaultProgramConfig{}));
        } else {
            return ttnn::operations::normalization::rmsnorm(
                input_tensor, epsilon, weight, bias, memory_config.value_or(input_tensor.memory_config()), program_config.value_or(LayerNormDefaultProgramConfig{}));
        }
    }
};

}  // namespace operations::normalization

constexpr auto rms_norm = ttnn::register_operation<ttnn::operations::normalization::ExecuteRMSNorm>("ttnn::rms_norm");

}  // namespace ttnn
