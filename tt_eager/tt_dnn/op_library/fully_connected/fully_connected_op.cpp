// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <functional>
#include <type_traits>

#include "tt_metal/host_api.hpp"
#include "tt_dnn/op_library/bmm/bmm_op.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"

namespace tt {
namespace tt_metal {

Tensor fully_connected_(const Tensor& act, const Tensor& weights, std::optional<std::reference_wrapper<const Tensor>> bias, const MemoryConfig& output_mem_config) {
    Tensor mm_output = tt::operations::primary::matmul(act, weights, /*bias=*/std::nullopt, /*program_config=*/std::nullopt, output_mem_config);
    if (bias) {
        Tensor tiled_bias = ttnn::to_layout(bias.value(), ttnn::TILE_LAYOUT, std::nullopt, std::nullopt, (Device*)nullptr);
        return ttnn::add(mm_output, tiled_bias, std::nullopt, output_mem_config);
    }
    return mm_output;
}

Tensor fully_connected(const Tensor &act, const Tensor& weights, std::optional<std::reference_wrapper<const Tensor>> bias, const MemoryConfig& output_mem_config) {
    TT_ASSERT(act.storage_type() == StorageType::DEVICE && weights.storage_type() == StorageType::DEVICE, "Activation and weight tensors need to be on device");
    // Assuming padding is already included. Not adding padding here.
    // NOTE: Bias is never padded.
    Device * device = act.device();
    return fully_connected_(act, weights, bias, output_mem_config);
}

}  // namespace tt_metal
}  // namespace tt
