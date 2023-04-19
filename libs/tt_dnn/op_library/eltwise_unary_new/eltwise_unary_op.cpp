#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_metal/host_api.hpp"
#include "constants.hpp"

using namespace tt::constants;

namespace eltwise_unary_op_utils {
using namespace tt::tt_metal;

string get_op_name(UnaryOpType::Enum op_type) {
    string op_name;
    switch (op_type) {
        case UnaryOpType::EXP: op_name = "exp_tile_init(); exp_tile(0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::RECIP: op_name = "recip_tile_init(); recip_tile(0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::GELU: op_name = "gelu_tile_init(); gelu_tile(0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::RELU: op_name = "pack_relu_tile_to_stream(0, CB::c_out0);"; break;
        case UnaryOpType::SQRT: op_name = "sqrt_tile_init(); sqrt_tile(0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::SIGMOID: op_name = "sigmoid_tile_init(); sigmoid_tile(0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::LOG: op_name = "log_tile_init(); log_tile(0); pack_tile(0, CB::c_out0);"; break;
        case UnaryOpType::TANH: op_name = "tanh_tile_init(); tanh_tile(0); pack_tile(0, CB::c_out0);"; break;

        default: TT_ASSERT(false && "Undefined op type");
    }
    return op_name;
}

void add_defines(ComputeKernel * eltwise_unary_kernel, UnaryOpType::Enum op_type){
    string op_name = get_op_name(op_type);
    eltwise_unary_kernel->add_define("SFPU_OP_AND_PACK", op_name);
    bool is_relu = (op_type == UnaryOpType::RELU);
    eltwise_unary_kernel->add_define("INIT_RELU", is_relu ? "pack_relu_config(1);" : "");
    eltwise_unary_kernel->add_define("DEINIT_RELU", is_relu ? "pack_relu_config(0);" : "");
    return;
}

UnaryOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a){
    uint32_t num_tiles = a.volume() / TILE_HW;
    if(num_tiles > 1){
        return UnaryOpParallelizationStrategy::MULTI_CORE;
    }
    else{
        return UnaryOpParallelizationStrategy::SINGLE_CORE;
    }
}

}  // namespace eltwise_unary_op_utils

namespace tt {

namespace tt_metal {

class EltwiseUnaryArgs {
    public:
        UnaryOpParallelizationStrategy parallelization;
        tt_xy_pair core_grid_shape;
        tt_xy_pair core_location; // Do we need this?
};

Tensor eltwise_unary(const Tensor &a, UnaryOpType::Enum op_type, EltwiseUnaryArgs op_args) {
    switch (op_args.parallelization){
        case UnaryOpParallelizationStrategy::SINGLE_CORE:
        default:
            return eltwise_unary_single_core(a, op_type, op_args);
    }
}

//Tensor eltwise_unary(const Tensor &a, UnaryOpType::Enum op_type, UnaryOpParallelizationStrategy::Enum strat) {
//    auto default_op_args = {"strat": strat, ...};
//    return eltwise_unary(a, op_type, default_op_args);
//}

//Tensor eltwise_unary(const Tensor &a, UnaryOpType::Enum op_type) {
//    auto default_op_args = {}; // get_parallelization_strategy would be here
//    return eltwise_unary(a, op_type, default_op_args);
//}

}  // namespace tt_metal

}  // namespace tt
