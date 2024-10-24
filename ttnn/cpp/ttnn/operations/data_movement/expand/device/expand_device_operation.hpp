// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <variant>
#include <vector>

#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::expand {
struct ExpandOperation {
    struct operation_attributes_t {
        const std::vector<uint32_t> output_shape = {0};

        const MemoryConfig output_mem_config;
        const DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor& input;
        const std::optional<Tensor>& output;
    };

    using shape_return_value_t = SimpleShape;
    using tensor_return_value_t = Tensor;

    struct Expand2DFactory {
        struct shared_variables_t {
            KernelHandle reader_kernel_id;
            KernelHandle writer_kernel_id;
            uint32_t num_cores;
            uint32_t num_cores_y;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensor);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& output_tensor);
    };

    using program_factory_t = std::variant<Expand2DFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input,
        const std::vector<uint32_t>& output_shape,

        const std::optional<Tensor>& output,
        const std::optional<MemoryConfig>& output_mem_config,
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
};
}  // namespace ttnn::operations::expand

namespace ttnn::prim {
constexpr auto expand = ttnn::register_operation<"ttnn::prim::expand", ttnn::operations::expand::ExpandOperation>();
}
