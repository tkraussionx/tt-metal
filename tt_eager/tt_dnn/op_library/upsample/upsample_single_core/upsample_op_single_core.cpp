#include <math.h>

#include "tt_dnn/op_library/upsample/upsample_op.hpp"
#include "tt_dnn/op_library/math.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/common/math.hpp"

#include "tt_metal/tt_stl/reflection.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks upsample_single_core(const Tensor &input, Tensor& output, int scale_factor_h, int scale_factor_w) {
    Program program{};
    CoreRange core = {.start={0, 0}, .end={0, 0}};

    tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t input_unit_size = input.shape()[-1] * input.element_size();
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_unit_size = output.shape()[-1] * output.element_size();

    uint32_t output_num_units = output.volume() / output.shape()[-1]; // N*H*W for outout
    uint32_t input_num_units = input.volume() / input.shape()[-1];  // N*H*W for input

    auto output_shape = output.shape();
    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = output.device();

    //circulat buffer for input
    uint32_t src0_cb_index = CB::c_in0;
    uint32_t num_input_units = 2;
    uint32_t aligned_input_unit_size = round_up_to_mul32(input_unit_size);
    tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_units * aligned_input_unit_size, {{src0_cb_index, input_cb_data_format}})
		.set_page_size(src0_cb_index, aligned_input_unit_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    //circulat buffer same for input and output. No compute kernels.
    uint32_t output_cb_index = src0_cb_index; // same as input cb

    auto src_buffer = input.buffer();
    auto dst_buffer = output.buffer();
    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;

    /*
    The data layout is mapped in DRAM as follows:
        number of channel = stick size
        NHW is number of sticks
    */

    std::vector<uint32_t> reader_compile_time_args, writer_compile_time_args;
    bool src_stick_size_is_power_of_two = is_power_of_two_at_least_32(input_unit_size);
    uint32_t src_log2_stick_size = src_stick_size_is_power_of_two ? (std::uint32_t)log2(input_unit_size) : 0;
    reader_compile_time_args = {
        (std::uint32_t) src0_cb_index,
        (std::uint32_t) src_is_dram,
        (std::uint32_t) src_stick_size_is_power_of_two,
        (std::uint32_t) src_log2_stick_size
    };

    bool dst_stick_size_is_power_of_two = is_power_of_two_at_least_32(output_unit_size);
    uint32_t dst_log2_stick_size = dst_stick_size_is_power_of_two ? (std::uint32_t)log2(output_unit_size) : 0;
    writer_compile_time_args = {
        (std::uint32_t) output_cb_index,
        (std::uint32_t) dst_is_dram,
        (std::uint32_t) dst_stick_size_is_power_of_two,
        (std::uint32_t) dst_log2_stick_size
    };

    std::map<string, string> kernel_defines;
    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/upsample/kernels/dataflow/reader_upsample_unary_stick_layout_interleaved_start_id.cpp",
        core,
        tt_metal::ReaderDataMovementConfig{.compile_args = reader_compile_time_args, .defines = kernel_defines});

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/upsample/kernels/dataflow/writer_upsample_unary_stick_layout_interleaved_start_id.cpp",
        core,
        tt_metal::WriterDataMovementConfig{.compile_args = writer_compile_time_args, .defines = kernel_defines});


    SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        {
            src_buffer->address(),
            input_unit_size,
            input_num_units
        }
    );

    SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        {
            dst_buffer->address(),
            input_unit_size,
            input_num_units,
            (uint32_t)scale_factor_h,
            (uint32_t)scale_factor_w,
            (uint32_t)output_shape[1],
            (uint32_t)output_shape[2]
        }
    );

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);

        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace tt_metal
}  // namespace tt
