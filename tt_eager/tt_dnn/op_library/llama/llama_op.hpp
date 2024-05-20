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

    Tensor llama_mlp_decode_forward(Tensor& input_tensor, Tensor& w1, Tensor& w2, Tensor& w3);

}  // namespace transformers
}  // namespace primary
}  // namespace operations
}  // namespace tt
