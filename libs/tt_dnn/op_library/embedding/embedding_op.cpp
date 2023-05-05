#include "libs/tt_dnn/op_library/embedding/embedding_op.hpp"
#include "tt_metal/host_api.hpp"
#include "constants.hpp"

#include <iostream>


namespace tt {
namespace tt_metal {

Tensor embedding_single_core(const Tensor& weights, const Tensor& in_act) {
    Program *program = new Program();
    tt_xy_pair core = {0, 0};

    const auto& weight_shape = weights.shape(), act_shape = in_act.shape();

    // the weights and activations need to be on the same device in allocated buffers
    TT_ASSERT(not weights.on_host() && not in_act.on_host(), "Weights and activations need to be on a device!");
    TT_ASSERT(weights.device() == in_act.device(), "Weights and activations need to be on the same device!");
    TT_ASSERT(weights.buffer() != std::nullptr && in_act.buffer() != nullptr, "Weights and activations need to be allocated in buffers on the device!");

    // make sure datatypes for weights and activations are as expected
    TT_ASSERT(weights.dtype() == tt::tt_metal::Datatype::BFLOAT16 || weights.dtype == tt::tt_metal::Datatype::BFLOAT8_B, "Invalid weights data type");
}

Tensor embedding(const Tensor& weights, const Tensor& in_act) {
    return embedding_single_core(weights, in_act);
}

} // namespace tt_metal
} // namespace tt
