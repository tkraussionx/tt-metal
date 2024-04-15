// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/sdpa/sdpa_op.hpp"
#include "tt_metal/common/assert.hpp"
#include "common/base_types.hpp"
#include "tensor/types.hpp"
#include "tt_eager/tt_dnn/op_library/math.hpp"
#include "tt_eager/tt_dnn/op_library/work_split.hpp"
#include "tt_dnn/op_library/run_operation.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/math.hpp"
#include "tt_metal/detail/util.hpp"

#include <optional>
#include <type_traits>

using uint32_t = std::uint32_t;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace tt {
namespace operations {
namespace primary {

void ScaledDotProductAttention::validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(input_tensors.size() == 3 and optional_input_tensors.size() <= 1, "Must have 1 or 2 input tensors");

    for (auto& input_tensor : input_tensors) {
        TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
        TT_FATAL(input_tensor.buffer() != nullptr , "Operands to softmax need to be allocated in buffers on device!");
        TT_FATAL((input_tensor.get_layout() == Layout::TILE), "Inputs to softmax must be tilized");
        TT_FATAL(input_tensor.get_dtype() == DataType::FLOAT32 || input_tensor.get_dtype() == DataType::BFLOAT16 || input_tensor.get_dtype() == DataType::BFLOAT8_B);
        TT_FATAL(input_tensor.is_sharded() == false);
    }

    if (optional_input_tensors.size() == 1) {
        if (optional_input_tensors.at(0).has_value()) {
            auto& mask = optional_input_tensors.at(0).value();
            TT_FATAL(mask.storage_type() == StorageType::DEVICE, "Operands to softmax need to be on device!");
            TT_FATAL(input_tensor.device() == mask.device());
            TT_FATAL(mask.get_layout() == Layout::TILE);
            TT_FATAL(mask.get_legacy_shape() == input_tensors.at(0).get_legacy_shape());
        }
    }


    std::visit(
        [&](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (
                std::is_same_v<ProgramConfigType, tt::operations::primary::transformers::SDPAProgramConfig>
            ) {
                const auto q_shape = input_tensors.at(0).get_legacy_shape();
                uint32_t M = input_tensor.volume() / shape[-1];
                uint32_t K = shape[-1];

                TT_FATAL(M % TILE_HEIGHT == 0, "M must be divisible by tile height.");
                TT_FATAL(K % TILE_WIDTH == 0, "K must be divisible by tile width.");
                TT_FATAL(program_config.block_w % program_config.subblock_w == 0, "block_w must be divisible by subblock_w.");
                TT_FATAL(program_config.block_w * TILE_WIDTH == shape[3], "shard width must equal to input tensor shape[3]!");
                TT_FATAL(this->inplace);
                if (!this->is_scale_causal_mask_hw_dims_softmax) {
                    // grid
                    auto num_cores_c = program_config.compute_with_storage_grid_size.x;
                    auto num_cores_r = program_config.compute_with_storage_grid_size.y;
                    // check dims
                    TT_FATAL(M * K / ((program_config.block_w * program_config.block_h) * TILE_HW) == num_cores_r * num_cores_c, "number of shards must equal to number of cores. M = {}, K = {}, block_w = {}, block_h = {}, num_cores = {}", M, K, program_config.block_w, program_config.block_h, num_cores_r * num_cores_c);
                } else {
                    TT_FATAL(this->is_causal_mask);
                    TT_FATAL(mask.get_layout() == Layout::TILE);
                    TT_FATAL(mask.is_sharded() == false);
                    TT_FATAL(input_tensor.get_layout() == Layout::TILE);
                    TT_FATAL(input_tensor.is_sharded());
                    TT_FATAL(input_tensor.shard_spec()->orientation == ShardOrientation::ROW_MAJOR);
                    TT_FATAL(this->scale.has_value());
                }
            }
        },
        this->program_config
    );
}

std::vector<Shape> ScaledDotProductAttention::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    return {input_tensors.at(0).get_legacy_shape()};
}

std::vector<Tensor> ScaledDotProductAttention::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensors.at(0).get_dtype(), Layout::TILE, this->output_mem_config);
}

operation::ProgramWithCallbacks ScaledDotProductAttention::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor> &output_tensors
) const {
    auto& input_tensor_q = input_tensors.at(0);
    auto& input_tensor_k = input_tensors.at(1);
    auto& input_tensor_v = input_tensors.at(2);
    auto& output_tensor = output_tensors.at(0);
    const auto& attn_mask = optional_input_tensors.at(0);

    return sdpa_multi_core(input_tensor_q, input_tensor_k, input_tensor_v, output_tensor, attn_mask, this->scale, this->is_causal, this->compute_kernel_config);
}

// What is this?
tt::stl::reflection::Attributes ScaledDotProductAttention::attributes() const {
    return {
        {"scale", this->scale},
        {"output_mem_config", this->output_mem_config},
    };
}


// const operation::Hash ScaledDotProductAttention::compute_program_hash(
//     const std::vector<Tensor> &input_tensors,
//     const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
//     return operation::hash_operation<Softmax>(
//         input_tensors.at(0).memory_config(),
//         input_tensors.at(0).get_dtype(),
//         optional_input_tensors.at(0).has_value() ? std::optional{optional_input_tensors.at(0).value().memory_config()}
//                                                  : std::nullopt,
//         optional_input_tensors.at(0).has_value() ? std::optional{optional_input_tensors.at(0).value().get_dtype()}
//                                                  : std::nullopt,
//         this->output_mem_config);
// }


namespace transformers {
// Function which is bound to the Python API
Tensor scaled_dot_product_attention(Tensor& input_tensor_q, Tensor& input_tensor_k, Tensor& input_tensor_v, std::optional<const Tensor> attn_mask, const bool is_causal, std::optional<float> scale, const MemoryConfig& output_mem_config, const SDPAProgramConfig& program_config, std::optional<const DeviceComputeKernelConfig> compute_kernel_config) {
    // TODO: Determine if fp32 acc and L1 acc is necessary
    auto kernel_config_val = init_device_compute_kernel_config(input_tensor_q.device()->arch(), compute_kernel_config, MathFidelity::HiFi4, false, false, false);
    return operation::run(ScaledDotProductAttention{.scale=scale, .output_mem_config=output_mem_config, .program_config=program_config, .is_causal=is_causal, .compute_kernel_config=kernel_config_val}, {input_tensor_q, input_tensor_k, input_tensor_v}, {attn_mask});
}


}  // namespace transformers
}  // namespace primary
}  // namespace operations

}  // namespace tt
