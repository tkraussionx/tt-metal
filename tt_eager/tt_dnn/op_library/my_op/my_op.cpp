// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/my_op/my_op.hpp"

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/operation.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"

using namespace tt::constants;

namespace tt {
namespace tt_metal {
void MyOp::validate(const std::vector<Tensor> &input_tensors) const {
    const auto &inp_tensor0 = input_tensors.at(0);
    const auto &inp_tensor1 = input_tensors.at(1);
    TT_FATAL(inp_tensor0.shape() == inp_tensor1.shape(), "Shapes must be the same!");
    TT_FATAL(
        inp_tensor0.storage_type() == StorageType::DEVICE and inp_tensor1.storage_type() == StorageType::DEVICE,
        "Operands to eltwise binary need to be on device!");
    TT_FATAL(inp_tensor0.device() == inp_tensor1.device(), "Operands to eltwise binary need to be on the same device!");
    TT_FATAL(
        inp_tensor0.buffer() != nullptr and inp_tensor1.buffer() != nullptr,
        "Operands to eltwise binary need to be allocated in buffers on device!");
    TT_FATAL(
        (inp_tensor0.layout() == Layout::TILE && inp_tensor1.layout() == Layout::TILE),
        "Inputs to eltwise binary must be tilized");
}

std::vector<Shape> MyOp::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    return {input_tensors.at(0).shape()};
}

std::vector<Tensor> MyOp::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto &input_tensor_a = input_tensors.at(0);
    return operation::generic_create_output_tensors(
        *this, input_tensors, input_tensor_a.dtype(), Layout::TILE, input_tensor_a.memory_config());
}

operation::ProgramWithCallbacks MyOp::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) {
    Program program{};
    CoreRange core = {.start = {0, 0}, .end = {0, 0}};
    const auto& output_tensor = output_tensors.at(0);

    const Tensor& a = input_tensors[0];
    const Tensor& b = input_tensors[1];

    tt::DataFormat src0_cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t src0_single_tile_size = tt_metal::detail::TileSize(src0_cb_data_format);
    tt::DataFormat src1_cb_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());
    uint32_t src1_single_tile_size = tt_metal::detail::TileSize(src1_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t dst_single_tile_size = tt_metal::detail::TileSize(dst_cb_data_format);

    tt_metal::Buffer *src0_buffer = a.buffer();
    tt_metal::Buffer *src1_buffer = b.buffer();

    uint32_t num_tiles = a.volume() / TILE_HW;

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    tt_metal::Buffer *dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt_metal::CircularBufferConfig cb_src0_config =
        tt_metal::CircularBufferConfig(num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
            .set_page_size(src0_cb_index, src0_single_tile_size);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    uint32_t src1_cb_index = 1;
    tt_metal::CircularBufferConfig cb_src1_config =
        tt_metal::CircularBufferConfig(num_input_tiles * src1_single_tile_size, {{src1_cb_index, src1_cb_data_format}})
            .set_page_size(src1_cb_index, src1_single_tile_size);
    auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    uint32_t output_cb_index = 16;  // output operands start at index 16
    uint32_t num_output_tiles = 2;
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, dst_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(std::uint32_t)src0_is_dram, (std::uint32_t)src1_is_dram};

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram};

    KernelHandle binary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/eltwise_binary/kernels/dataflow/reader_binary_interleaved_start_id.cpp",
        core,
        tt_metal::ReaderDataMovementConfig{.compile_args = reader_compile_time_args});

    KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core,
        tt_metal::WriterDataMovementConfig{.compile_args = writer_compile_time_args});

    vector<uint32_t> compute_kernel_args = {};

    std::map<std::string, std::string> my_op_defines = {};

    auto eltwise_binary_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/compute/my_binary.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = my_op_defines});

    tt_metal::SetRuntimeArgs(
        program, binary_reader_kernel_id, core, {src0_buffer->address(), src1_buffer->address(), num_tiles, 0});

    tt_metal::SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {num_tiles, 1});

    tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_buffer->address(), num_tiles, 0});

    auto override_runtime_arguments_callback = [
            binary_reader_kernel_id,
            unary_writer_kernel_id,
            eltwise_binary_kernel_id
        ]
    (
        const void* operation,
        const Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {

        auto src_buffer_a = input_tensors.at(0).buffer();
        auto src_buffer_b = input_tensors.at(1).buffer();

        auto dst_buffer = output_tensors.size() == 1 ? output_tensors.at(0).buffer() : src_buffer_a;

        CoreCoord core = {0, 0};

        uint32_t num_tiles = input_tensors.at(0).volume() / TILE_HW;

        {
            auto &runtime_args = GetRuntimeArgs(program, binary_reader_kernel_id, core);
            runtime_args[0] = src_buffer_a->address();
            runtime_args[1] = src_buffer_b->address();
            runtime_args[2] = num_tiles;
        }

        {
            auto &runtime_args = GetRuntimeArgs(program, eltwise_binary_kernel_id, core);
            runtime_args[0] = num_tiles;
        }

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
            runtime_args[1] = num_tiles;
        }
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}
}  // namespace tt_metal
}  // namespace tt
