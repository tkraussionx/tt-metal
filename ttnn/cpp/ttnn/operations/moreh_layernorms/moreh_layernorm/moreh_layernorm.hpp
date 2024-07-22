
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/moreh_layernorm_device_operation.hpp"
#include <optional>

namespace ttnn::operations::moreh_layernorms {

// This is the main operation that will be called by the user
struct MorehLayernormOperation {
    // This how the user can call the operation
    static std::tuple<std::optional<Tensor>,std::optional<Tensor>,std::optional<Tensor>> execute_on_worker_thread(
        uint8_t queue_id,
        const Tensor &input,
        uint32_t normalized_dims,
        float eps,
        const std::optional<const Tensor> gamma = std::nullopt,
        const std::optional<const Tensor> beta = std::nullopt,
        const std::optional<const Tensor> output = std::nullopt,
        const std::optional<const Tensor> mean = std::nullopt,
        const std::optional<const Tensor> rstd = std::nullopt,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt
        ) {
        return ttnn::device_operation::run<MorehLayernormDeviceOperation>(
            queue_id,
            MorehLayernormDeviceOperation::operation_attributes_t{.normalized_dims = normalized_dims, .eps=eps, .memory_config=memory_config.value_or(input.memory_config()), .compute_kernel_config=compute_kernel_config},
            MorehLayernormDeviceOperation::tensor_args_t{input, gamma, beta, output, mean, rstd});
    }

    // This how the user can call the operation
    static std::tuple<std::optional<Tensor>,std::optional<Tensor>,std::optional<Tensor>> execute_on_worker_thread(
        const Tensor &input,
        uint32_t normalized_dims,
        float eps,
        const std::optional<const Tensor> gamma = std::nullopt,
        const std::optional<const Tensor> beta = std::nullopt,
        const std::optional<const Tensor> output = std::nullopt,
        const std::optional<const Tensor> mean = std::nullopt,
        const std::optional<const Tensor> rstd = std::nullopt,
        const std::optional<MemoryConfig> &memory_config = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt
    ) { return execute_on_worker_thread(0, input, normalized_dims, eps, gamma, beta, output, mean, rstd, memory_config, compute_kernel_config); }

    // execute_on_main_thread can be overloaded to take any number of arguments

    // execute_on_main_thread doesn't imply anything about async or sync execution and the user needs to be aware of
    // that

    // If the user wants to make the operation async, automatically, then `execute_on_main_thread` should renamed to
    // `execute_on_worker_thread`
};

}  // namespace ttnn::operations::moreh_layernorms

namespace ttnn {

// Register the operation. The name, in this case, "ttnn::example" should match the namespace of the operation
// And the name will be directly mapped to python, where it will become "ttnn.example"
constexpr auto moreh_layernorm = ttnn::register_operation<operations::moreh_layernorms::MorehLayernormOperation>("ttnn::moreh_layernorm");

}  // namespace ttnn
