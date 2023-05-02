#include "tt_dnn/op_library/transpose/transpose_op.hpp"

#include "tt_metal/host_api.hpp"
#include "constants.hpp"
#include "tt_dnn/op_library/auto_pad.hpp"

using u32 = std::uint32_t;
using namespace tt::constants;

namespace transpose_op_utils {

using namespace tt::tt_metal;

TransposeOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a, TransposeOpDim::Enum transpose_dim){
    auto ashape = a.shape();
    uint32_t num_tiles = a.volume() / TILE_HW;
    if (transpose_dim == TransposeOpDim::WH && num_tiles > 1) {
        return TransposeOpParallelizationStrategy::MULTI_CORE_WH;
    } else if (transpose_dim == TransposeOpDim::HC && num_tiles > 1) { // Always true for legal shape until requirement on tile size IO is no longer required
        return TransposeOpParallelizationStrategy::MULTI_CORE_HC;
    } else {
        return TransposeOpParallelizationStrategy::SINGLE_CORE;
    }
}

} // namespace transpose_op_utils

namespace tt {

namespace tt_metal {

Tensor transpose_(const Tensor &a, TransposeOpDim::Enum transpose_dim) {

    Device * device;

    // Get the device
    if (a.on_host()) {
        device = AutoPad::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else {
        device = a.device();
    }

    // Bring tensor to host if it isn't already, pad and convert layout, send to device
    auto input1 = AutoPad::format_input_tensor(a, device, transpose_dim == TransposeOpDim::HC);

    Tensor output = Tensor({1, 1, 1, 1}, Initialize::ZEROS, DataType::BFLOAT16, Layout::ROW_MAJOR); // No Default Tensor Constructor, create dummy

    switch (transpose_op_utils::get_parallelization_strategy(input1, transpose_dim)){
        case TransposeOpParallelizationStrategy::MULTI_CORE_WH:
            output = transpose_wh_multi_core(input1);
            break;
        case TransposeOpParallelizationStrategy::MULTI_CORE_HC:
            output = transpose_hc_multi_core(input1);
            break;
        case TransposeOpParallelizationStrategy::SINGLE_CORE:
        default:
            output = transpose_single_core(input1, transpose_dim);
    }


    // Convert tensor back to original
    auto shape = a.shape();
    switch (transpose_dim){
        case TransposeOpDim::CN:
            shape[0] = a.shape()[1];
            shape[1] = a.shape()[0];
            break;
        case TransposeOpDim::HC:
            shape[1] = a.shape()[2];
            shape[2] = a.shape()[1];
            break;
        case TransposeOpDim::WH:
            shape[2] = a.shape()[3];
            shape[3] = a.shape()[2];
            break;
    }
    output = AutoPad::format_output_tensor(a, output, shape, device);

    return output;
}

}  // namespace tt_metal

}  // namespace tt
