#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_metal/host_api.hpp"
#include "constants.hpp"
#include "tt_dnn/op_library/auto_pad.hpp"

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

static Profiler op_profiler = Profiler();
static uint32_t call_count = 0;
static const string op_name = "eltwise_unary";
static const string perf_folder = "/tmp/tt_perf/ops/";
static string prepend_string = "";

static const unordered_map<UnaryOpType::Enum ,string> optype_to_name = {
    {UnaryOpType::Enum::EXP,"EXP"},
    {UnaryOpType::Enum::RECIP,"SURECIPB"},
    {UnaryOpType::Enum::GELU,"GELU"},
    {UnaryOpType::Enum::RELU,"RELU"},
    {UnaryOpType::Enum::SQRT,"SQRT"},
    {UnaryOpType::Enum::SIGMOID,"SIGMOID"},
    {UnaryOpType::Enum::LOG,"LOG"},
    {UnaryOpType::Enum::TANH,"TANH"}
};

Tensor eltwise_unary_(const Tensor &a, UnaryOpType::Enum op_type) {

    switch (eltwise_unary_op_utils::get_parallelization_strategy(a)){
        case UnaryOpParallelizationStrategy::MULTI_CORE:
            prepend_string += "_MULTI_CORE-" + optype_to_name.at(op_type);
            return eltwise_unary_multi_core(a, op_type, call_count);
            break;
        case UnaryOpParallelizationStrategy::SINGLE_CORE:
        default:
            prepend_string += "_SINGLE_CORE-" + optype_to_name.at(op_type);
            return eltwise_unary_single_core(a, op_type, call_count);
    }

}


Tensor _eltwise_unary(const Tensor &a, UnaryOpType::Enum op_type) {

    Device * device;

    // Get the device
    if (a.on_host()) {
        device = AutoPad::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else {
        device = a.device();
    }

    auto a_pad_shape = AutoPad::pad_to_tile_shape(a.shape());
    auto out_shape = a.shape();
    if (AutoPad::check_input_tensor_format(a, a_pad_shape)) {
        prepend_string += "NO_PAD_A";
        return eltwise_unary_(a, op_type);
    } else {
        prepend_string += "PAD_A";
        auto output = eltwise_unary_(AutoPad::format_input_tensor(a, device, a_pad_shape, 0), op_type);
        AutoPad::format_output_tensor(a, output, out_shape, device);
        return output;
    }

}

Tensor eltwise_unary(const Tensor &a, UnaryOpType::Enum op_type) {
    op_profiler.markStart(op_name);
    op_profiler.setOutputDir(perf_folder + op_name);
    call_count ++;

    Tensor ret = _eltwise_unary(a, op_type) ;

    op_profiler.markStop(op_name);
    op_profiler.dumpHostResults(to_string(call_count) + "-" + prepend_string);
    prepend_string = "";

    return ret;
}


}  // namespace tt_metal

}  // namespace tt
