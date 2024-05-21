#pragma once

#include <optional>

#include "tt_dnn/op_library/compute_kernel_config.hpp"
#include "tt_eager/tensor/tensor.hpp"
#include "ttnn/operations/core.hpp"

namespace tt {
namespace operations {
namespace primary {

using namespace tt_metal;

namespace transformers {

    Tensor llama_mlp_decode_forward(Tensor& input_tensor, const Tensor& w1, const Tensor& w2, const Tensor& w3);

    std::tuple<Tensor, Tensor, Tensor> llama_attn_qkv_decode_forward(const Tensor& input_tensor, const Tensor& rot_mat, const Tensor& wqkv, const MemoryConfig sharded_mem_config);

}  // namespace transformers
}  // namespace primary
}  // namespace operations
}  // namespace tt
