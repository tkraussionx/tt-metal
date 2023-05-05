#include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
#include "tt_metal/host_api.hpp"
#include "constants.hpp"
#include "tt_dnn/op_library/auto_pad.hpp"

using namespace tt::constants;

namespace eltwise_binary_op_utils {
using namespace tt::tt_metal;

void add_defines(ComputeKernel * eltwise_binary_kernel, BinaryOpType::Enum op_type){
    string op_name, op_code;
    switch (op_type) {
        case BinaryOpType::ADD: op_name = "add_tiles"; op_code = "0"; break;
        case BinaryOpType::SUB: op_name = "sub_tiles"; op_code = "1"; break;
        case BinaryOpType::MUL: op_name = "mul_tiles"; op_code = "2"; break;
        default: TT_ASSERT(false && "Undefined op type");
    }
    eltwise_binary_kernel->add_define("ELTWISE_OP", op_name.c_str());
    eltwise_binary_kernel->add_define("ELTWISE_OP_CODE", op_code.c_str());
}

BinaryOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a, const Tensor &b){
    uint32_t num_tiles = a.volume() / TILE_HW;
    if(num_tiles > 1){
        return BinaryOpParallelizationStrategy::MULTI_CORE;
    }
    else{
        return BinaryOpParallelizationStrategy::SINGLE_CORE;
    }
}

}  // eltwise_binary_op_utils

namespace tt {

namespace tt_metal {

Tensor eltwise_binary(const Tensor &a, const Tensor &b, BinaryOpType::Enum op_type) {

    Device * device;

    // Get the device
    if (a.on_host() && b.on_host()) {
        device = AutoPad::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else if (!a.on_host()){
        device = a.device();
    } else {
        device = b.device();
    }

    TT_ASSERT(a.shape() == b.shape() && "Operand to eltwise binary need to be the same size!");

    // Bring tensor to host if it isn't already, pad and convert layout, send to device
    auto input1 = AutoPad::format_input_tensor(a, device);
    auto input2 = AutoPad::format_input_tensor(b, device);

    Tensor output = Tensor({1, 1, 1, 1}, Initialize::ZEROS, DataType::BFLOAT16, Layout::ROW_MAJOR); // No Default Tensor Constructor, create dummy

    switch (eltwise_binary_op_utils::get_parallelization_strategy(input1, input2)){
        case BinaryOpParallelizationStrategy::MULTI_CORE:
            output = eltwise_binary_multi_core(input1, input2, op_type);
            break;
        case BinaryOpParallelizationStrategy::SINGLE_CORE:
        default:
            output = eltwise_binary_single_core(input1, input2, op_type);
    }

    // Convert tensor back to original
    output = AutoPad::format_output_tensor(a, output, a.shape(), device);

    return output;

}

}  // namespace tt_metal

}  // namespace tt
