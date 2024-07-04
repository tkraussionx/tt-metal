// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "tt_dnn/op_library/compute_kernel_config.hpp"
#include "tt_dnn/op_library/run_operation.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "ttnn/operations/core.hpp"

using namespace tt::constants;

namespace ttnn::operations::normalization {

enum class LayerNormType {
    LAYERNORM, RMSNORM
};

struct LayerNormDefaultProgramConfig{
    static constexpr auto attribute_names = std::forward_as_tuple();
    static constexpr auto attribute_values() { return std::forward_as_tuple(); }
};
struct LayerNormShardedMultiCoreProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t subblock_w;
    std::size_t block_h;
    std::size_t block_w;
    bool inplace;

    static constexpr auto attribute_names =
        std::forward_as_tuple("compute_with_storage_grid_size", "subblock_w", "block_h", "block_w", "inplace");

    const auto attribute_values() const {
        return std::forward_as_tuple(compute_with_storage_grid_size, subblock_w, block_h, block_w, inplace);
    }
};

using LayerNormProgramConfig = std::variant<
    LayerNormDefaultProgramConfig,
    LayerNormShardedMultiCoreProgramConfig
>;

operation::ProgramWithCallbacks layernorm_multi_core(
    const Tensor &a,
    const std::optional<const Tensor> b,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta,
    Tensor& output,
    LayerNormType norm_type,
    float eps,
    DeviceComputeKernelConfig compute_kernel_config
);

operation::ProgramWithCallbacks layernorm_multi_core_sharded(
    const Tensor &a,
    const std::optional<const Tensor> b,
    const std::optional<const Tensor> gamma,
    const std::optional<const Tensor> beta,
    Tensor& output,
    LayerNormType norm_type,
    float eps,
    CoreCoord compute_grid_size,
    uint32_t subblock_wt,
    uint32_t block_ht,
    uint32_t block_wt,
    DeviceComputeKernelConfig compute_kernel_config
);

struct LayerNorm {
    LayerNormType norm_type;
    float eps;
    MemoryConfig output_mem_config;
    LayerNormProgramConfig program_config;
    const DeviceComputeKernelConfig compute_kernel_config;

    void validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const;
    std::vector<tt::tt_metal::Shape> compute_output_shapes(const std::vector<Tensor> &input_tensors) const;
    std::vector<Tensor> create_output_tensors(const std::vector<Tensor> &input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor> &output_tensors
    ) const;

    static constexpr auto attribute_names =
        std::forward_as_tuple("norm_type", "eps", "output_mem_config", "program_config", "compute_kernel_config");
    const auto attribute_values() const {
        return std::forward_as_tuple(norm_type, eps, output_mem_config, program_config, compute_kernel_config);
    }
};

template <LayerNormType layernorm_type>
struct make_layernorm {
    Tensor operator()(
        const Tensor& a,
        float eps,
        std::optional<const Tensor> gamma = std::nullopt,
        std::optional<const Tensor> beta = std::nullopt,
        const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG,
        const LayerNormProgramConfig& program_config = LayerNormDefaultProgramConfig{},
        std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) const {
        std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({a}))};
        operation::launch_op(
            [eps, mem_config, program_config, compute_kernel_config] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
                const auto& a = input_tensors.at(0);
                const auto& gamma = optional_input_tensors.at(0);
                const auto& beta = optional_input_tensors.at(1);
                auto arch = a.storage_type() == StorageType::DEVICE ? a.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
                auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);
                return operation::run(
                        LayerNorm{
                            .norm_type = layernorm_type,
                            .eps = eps,
                            .output_mem_config = mem_config,
                            .program_config = program_config,
                            .compute_kernel_config = kernel_config_val},
                        {a},
                        {std::nullopt, gamma, beta});
            }, {a}, output_tensors, {gamma, beta});
        return output_tensors.at(0);
    }
};

template <LayerNormType layernorm_type>
struct make_add_layernorm {
    Tensor operator()(
        const Tensor &a, const Tensor& b, float eps, std::optional<const Tensor> gamma = std::nullopt, std::optional<const Tensor> beta = std::nullopt, const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, const LayerNormProgramConfig& program_config = LayerNormDefaultProgramConfig{}, std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt) const {
        std::vector<Tensor> output_tensors = {Tensor(operation::get_workers_for_op_output({a, b}))};
        operation::launch_op(
            [eps, mem_config, program_config, compute_kernel_config] (const std::vector<Tensor>& input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors, const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
                const auto& a = input_tensors.at(0);
                const auto& b = input_tensors.at(1);
                const auto& gamma = optional_input_tensors.at(0);
                const auto& beta = optional_input_tensors.at(1);
                auto arch = a.storage_type() == StorageType::DEVICE ? a.device()->arch() : AutoFormat::GetDefaultDevice()->arch();
                auto kernel_config_val = init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);
                return operation::run(
                   LayerNorm{
                       .norm_type = layernorm_type,
                       .eps = eps,
                       .output_mem_config = mem_config,
                       .program_config = program_config,
                       .compute_kernel_config = kernel_config_val},
                   {a},
                   {b, gamma, beta});
            }, {a, b}, output_tensors, {gamma, beta});
        return output_tensors.at(0);
    }
};

constexpr auto layernorm = make_layernorm<LayerNormType::LAYERNORM>{};
constexpr auto rmsnorm = make_layernorm<LayerNormType::RMSNORM>{};

// computes layernorm(a+b)*gamma+beta
constexpr auto add_layernorm = make_add_layernorm<LayerNormType::LAYERNORM>{};
constexpr auto add_rmsnorm = make_add_layernorm<LayerNormType::RMSNORM>{};

}  // namespace ttnn::operations::normalization
