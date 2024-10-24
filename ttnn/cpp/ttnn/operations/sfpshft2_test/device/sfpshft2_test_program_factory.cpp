#include "tt_metal/common/constants.hpp"
#include "tt_metal/common/work_split.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/sfpshft2_test/device/sfpshft2_test_device_operation.hpp"

namespace ttnn::operations::sfpshft2_test {

using namespace tt;
using namespace tt::tt_metal;

SFPSHFT2TestDeviceOperation::SingleCore::cached_program_t SFPSHFT2TestDeviceOperation::SingleCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;

    const auto shape = input.get_padded_shape();
    const uint32_t num_tiles = shape.volume() / tt::constants::TILE_HW;

    Program program{};
    const auto device = input.device();
    const CoreCoord core{0, 0};

    // create CBs
    constexpr CB cb_input = CB::c_in0;
    constexpr CB cb_output = CB::c_out0;

    constexpr tt::DataFormat cb_data_format = tt::DataFormat::Float16_b;
    constexpr uint32_t cb_num_tiles = 2;

    tt::operations::primary::CreateCircularBuffer(
        program,
        core,
        cb_data_format,
        {
            {cb_input, cb_num_tiles},
            {cb_output, cb_num_tiles},
        });

    // create reader and writer kernels
    auto input_buffer = input.buffer();
    const uint32_t input_is_dram = input_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args{
        input_is_dram,
        input_buffer->address(),
        num_tiles,
    };
    const auto reader_kernel_file = "ttnn/cpp/ttnn/operations/sfpshft2_test/device/kernels/reader_sfpshft2_test.cpp";
    const auto reader_kernel_id =
        tt::operations::primary::CreateReadKernel(program, reader_kernel_file, core, reader_compile_time_args);

    auto output_buffer = output.buffer();
    const uint32_t output_is_dram = output_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args{
        output_is_dram,
        output_buffer->address(),
        num_tiles,
    };
    const auto writer_kernel_file = "ttnn/cpp/ttnn/operations/sfpshft2_test/device/kernels/writer_sfpshft2_test.cpp";
    const auto writer_kernel_id =
        tt::operations::primary::CreateWriteKernel(program, writer_kernel_file, core, writer_compile_time_args);

    // create compute kernel
    std::vector<uint32_t> compute_compile_time_args{
        num_tiles,
    };
    const auto compute_kernel_file = "ttnn/cpp/ttnn/operations/sfpshft2_test/device/kernels/sfpshft2_test.cpp";
    const auto compute_kernel_id = tt::operations::primary::CreateComputeKernel(
        program, compute_kernel_file, {core, num_tiles, compute_compile_time_args});

    return {std::move(program), {}};
}

}  // namespace ttnn::operations::sfpshft2_test
