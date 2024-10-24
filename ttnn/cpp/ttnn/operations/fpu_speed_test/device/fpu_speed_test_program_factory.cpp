#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/operations/fpu_speed_test/device/fpu_speed_test_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::fpu_speed_test {

using namespace tt;
using namespace tt::tt_metal;

FPUSpeedTestDeviceOperation::SingleCore::cached_program_t FPUSpeedTestDeviceOperation::SingleCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto num_tiles = operation_attributes.num_tiles;
    const auto fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;
    const auto& dummy = tensor_args.dummy;

    Program program{};
    const auto device = dummy.device();
    const CoreCoord core{0, 0};

    // create CBs
    constexpr CB cb_input = CB::c_in0;
    constexpr CB cb_other = CB::c_in1;
    constexpr CB cb_output = CB::c_out0;

    constexpr tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    constexpr uint32_t cb_num_tiles = 2;

    tt::operations::primary::CreateCircularBuffer(
        program,
        core,
        cb_data_format,
        {
            {cb_input, cb_num_tiles},
            {cb_other, cb_num_tiles},
            {cb_output, cb_num_tiles},
        });

    // create reader and writer kernels
    const auto reader_kernel_file = "ttnn/cpp/ttnn/operations/fpu_speed_test/device/kernels/reader_fpu_speed_test.cpp";
    const auto reader_kernel_id = tt::operations::primary::CreateReadKernel(program, reader_kernel_file, core);

    const auto writer_kernel_file = "ttnn/cpp/ttnn/operations/fpu_speed_test/device/kernels/writer_fpu_speed_test.cpp";
    const auto writer_kernel_id = tt::operations::primary::CreateWriteKernel(program, writer_kernel_file, core);

    // create compute kernel
    std::map<string, string> compute_defines;
    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    const auto compute_kernel_file = "ttnn/cpp/ttnn/operations/fpu_speed_test/device/kernels/fpu_speed_test.cpp";
    const auto compute_kernel_id = tt::operations::primary::CreateComputeKernel(
        program, compute_kernel_file, {core, num_tiles, {}}, compute_defines, MathFidelity::HiFi4, fp32_dest_acc_en);

    // set runtime args
    SetRuntimeArgs(
        program,
        compute_kernel_id,
        core,
        {
            num_tiles,
        });

    return {std::move(program), {}};
}

}  // namespace ttnn::operations::fpu_speed_test
