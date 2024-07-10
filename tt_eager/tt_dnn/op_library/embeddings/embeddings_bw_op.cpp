// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/cb_utils.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_eager/tt_dnn/op_library/embeddings/embeddings_op.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace tt::tt_metal {

operation::ProgramWithCallbacks embeddings_(const Tensor &weights, const Tensor &index, Tensor &output) {
    tt_metal::Buffer *weights_buffer = weights.buffer();
    tt_metal::Buffer *index_buffer = index.buffer();
    tt_metal::Buffer *out_buffer = output.buffer();

    Device *device = weights.device();

    Program program{};

    bool weights_is_dram = weights_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    tt::DataFormat weights_cb_data_format = tt_metal::datatype_to_dataformat_converter(weights.get_dtype());
    uint32_t weights_single_tile_size = tt_metal::detail::TileSize(weights_cb_data_format);

    bool index_is_dram = index_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    tt::DataFormat index_cb_data_format = tt_metal::datatype_to_dataformat_converter(index.get_dtype());
    uint32_t index_single_tile_size = tt_metal::detail::TileSize(index_cb_data_format);

    bool out_is_dram = out_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_cb_data_format);

    CoreCoord core{0, 0};

    auto [weights_cb, weights_handle] =
        create_cb(CB::c_in0, program, core, weights_single_tile_size, 1, weights_cb_data_format);
    auto [index_cb, index_handle] =
        create_cb(CB::c_in1, program, core, index_single_tile_size, 1, index_cb_data_format);
    auto [output_cb, output_handle] =
        create_cb(CB::c_out0, program, core, output_single_tile_size, 1, output_cb_data_format);

    // Create Kernels
    std::vector<uint32_t> reader_compile_time_args = {weights_cb, index_cb, weights_is_dram, index_is_dram};

    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/embeddings/kernels/dataflow/reader_embeddings_bw.cpp",
        core,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    SetRuntimeArgs(program, reader_kernel_id, core, {weights_buffer->address(), index_buffer->address()});

    auto reshuffle_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/embeddings/kernels/compute/reshuffle.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = {weights_cb, index_cb, output_cb}});

    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "tt_eager/tt_dnn/op_library/embeddings/kernels/dataflow/writer_embeddings_bw.cpp",
        core,
        tt_metal::WriterDataMovementConfig({output_cb, out_is_dram}));

    SetRuntimeArgs(program, writer_kernel_id, core, {out_buffer->address()});

    auto override_runtime_args_callback = [](const Program &program,
                                             const std::vector<Buffer *> &input_buffers,
                                             const std::vector<Buffer *> &output_buffers) {};

    return {std::move(program), override_runtime_args_callback};
}

std::vector<Shape> EmbeddingsBw::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto &weight_tensor = input_tensors.at(0);
    return {weight_tensor.get_legacy_shape()};
}

std::vector<Tensor> EmbeddingsBw::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto &weight_tensor = input_tensors.at(0);
    auto dtype = weight_tensor.dtype();

    return operation::generic_create_output_tensors(*this, input_tensors, dtype, Layout::TILE, nullopt);
}

operation::ProgramWithCallbacks EmbeddingsBw::create_program(
    const std::vector<Tensor> &input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto &weights = input_tensors.at(0);
    const auto &index = input_tensors.at(1);
    auto &output = output_tensors.at(0);

    return embeddings_(weights, index, output);
}

}  // namespace tt::tt_metal
