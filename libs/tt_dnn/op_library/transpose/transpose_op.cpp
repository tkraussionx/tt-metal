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

static Profiler op_profiler = Profiler();
static uint32_t call_count = 0;
static const string op_name = "transpose";
static const string perf_folder = "/tmp/tt_perf/ops/";
static string prepend_string = "";

Tensor transpose__(const Tensor &a, TransposeOpDim::Enum transpose_dim) {
    switch (transpose_op_utils::get_parallelization_strategy(a, transpose_dim)){
        case TransposeOpParallelizationStrategy::MULTI_CORE_WH:
            prepend_string += "_MULTI_CORE_WH";
            return transpose_wh_multi_core(a, call_count);
            break;
        case TransposeOpParallelizationStrategy::MULTI_CORE_HC:
            prepend_string += "_MULTI_CORE_HC";
            return transpose_hc_multi_core(a, call_count);
            break;
        case TransposeOpParallelizationStrategy::SINGLE_CORE:
        default:
            prepend_string += "_SINGLE_CORE";
            return transpose_single_core(a, transpose_dim);
    }
}


Tensor transpose_(const Tensor &a, TransposeOpDim::Enum transpose_dim) {

    Device * device;

    // Get the device
    if (a.on_host()) {
        device = AutoPad::GetDefaultDevice();
        TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to op are on device");
    } else {
        device = a.device();
    }

    // Convert tensor back to original
    auto a_pad_shape = AutoPad::pad_to_tile_shape(a.shape(), transpose_dim == TransposeOpDim::HC);
    auto out_shape = a.shape();
    switch (transpose_dim){
        case TransposeOpDim::CN:
            out_shape[0] = a.shape()[1];
            out_shape[1] = a.shape()[0];
            break;
        case TransposeOpDim::HC:
            out_shape[1] = a.shape()[2];
            out_shape[2] = a.shape()[1];
            break;
        case TransposeOpDim::WH:
            out_shape[2] = a.shape()[3];
            out_shape[3] = a.shape()[2];
            break;
    }

    if (AutoPad::check_input_tensor_format(a, a_pad_shape)) {
        prepend_string += "NO_PAD_A";
        return transpose__(a, transpose_dim);
    } else {
        prepend_string += "PAD_A";
        auto output = transpose__(AutoPad::format_input_tensor(a, device, a_pad_shape, 0), transpose_dim);
        AutoPad::format_output_tensor(a, output, out_shape, device);
        return output;

    }
}
Tensor _transpose(const Tensor &a, TransposeOpDim::Enum transpose_dim) {
    op_profiler.markStart(op_name);
    op_profiler.setOutputDir(perf_folder + op_name);
    call_count ++;

    Tensor ret = transpose_(a, transpose_dim);

    switch (transpose_dim){
        case TransposeOpDim::CN:
            prepend_string += "_CN";
            break;
        case TransposeOpDim::HC:
            prepend_string += "_HC";
            break;
        case TransposeOpDim::WH:
            prepend_string += "_WH";
            break;
    }

    op_profiler.markStop(op_name);
    op_profiler.dumpHostResults(to_string(call_count) + "-" + prepend_string);
    prepend_string = "";

    return ret;
}

}  // namespace tt_metal

}  // namespace tt
