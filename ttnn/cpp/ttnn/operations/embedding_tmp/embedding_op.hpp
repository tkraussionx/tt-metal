// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#pragma once

#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/embedding.hpp"

namespace ttnn {

namespace operations {

namespace embedding {
enum class EmbeddingsType { GENERIC, PADDED, BINARY };
enum class EmbeddingsIndexType { UINT32, BFP16 };

struct Embedding {
    struct operation_attributes_t {
        const MemoryConfig output_mem_config;
        const bool tilized;
        const EmbeddingsType embeddings_type;
        const std::optional<uint32_t> pad_token;
        const DataType output_dtype;
    };

    struct tensor_args_t {
        const Tensor& input_tensor_arg;
        const Tensor& weight_arg;
        const std::optional<Tensor>& optional_output_tensor;
    };

    using shape_return_value_t = ttnn::Shape;
    using tensor_return_value_t = Tensor;

    struct SingleCore {
        struct shared_variables_t {
            KernelHandle reader_kernel_id;
            KernelHandle writer_kernel_id;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(const operation_attributes_t& operation_attributes,
                                       const tensor_args_t& tensor_args,
                                       tensor_return_value_t& output_tensor);
        static void override_runtime_arguments(cached_program_t& cached_program,
                                               const operation_attributes_t& operation_attributes,
                                               const tensor_args_t& tensor_args,
                                               tensor_return_value_t& output_tensor);
    };

    struct MultiCore {
        struct shared_variables_t {
            KernelHandle writer_kernel_id;
            uint8_t stride_h;
            uint8_t stride_w;
            uint32_t cb_src0;
            uint32_t cb_dst0;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(const operation_attributes_t& operation_attributes,
                                       const tensor_args_t& tensor_args,
                                       tensor_return_value_t& output_tensor);
        static void override_runtime_arguments(cached_program_t& cached_program,
                                               const operation_attributes_t& operation_attributes,
                                               const tensor_args_t& tensor_args,
                                               tensor_return_value_t& output_tensor);
    };

    using program_factory_t = std::variant<SingleCore, MultiCore>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};
}  // namespace embedding
}  // namespace operations

constexpr auto embedding = ttnn::register_operation_with_auto_launch_op<"ttnn::embedding", ttnn::operations::embedding::EmbeddingOperation>();

}  // namespace ttnn
