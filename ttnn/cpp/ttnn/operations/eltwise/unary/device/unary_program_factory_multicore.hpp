// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>

#include "ttnn/operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "unary_op.hpp"

#include <iostream>

namespace ttnn::operations::unary::detail {

using namespace tt::constants;

operation::ProgramWithCallbacks unary_multi_core(const Tensor &a, Tensor &output, const std::vector<UnaryWithParam> op_chain, bool fp32_dest_acc_en, bool preserve_fp32_precision) {

    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);
    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    uint32_t single_tile_size_output = tt::tt_metal::detail::TileSize(cb_data_format_output);

    uint32_t num_tiles = a.volume() / tt::constants::TILE_HW;

    tt::tt_metal::Device *device = a.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    std::cout << "compute_with_storage_grid_size" << compute_with_storage_grid_size.str() << std::endl;
    std::cout << "num_tiles: " << num_tiles << std::endl;
    std::cout << "num_cores: " << num_cores << std::endl;
    std::cout << "all_cores: " << all_cores.str() << std::endl;
    std::cout << "core_group_1: " << core_group_1.str() << std::endl;

    for (const auto& core_range : core_group_1.ranges()) {
        std::cout << "core_range: " << core_range.str() << std::endl;
    }

    std::cout << "core_group_2: " << core_group_2.str() << std::endl;
    std::cout << "num_tiles_per_core_group_1: " << num_tiles_per_core_group_1 << std::endl;
    std::cout << "num_tiles_per_core_group_2: " << num_tiles_per_core_group_2 << std::endl;

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_cb_index = 16;  // output operands start at index 16
    uint32_t num_output_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * single_tile_size_output, {{output_cb_index, cb_data_format_output}})
            .set_page_size(output_cb_index, single_tile_size_output);
    auto cb_output = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    auto src_buffer = a.buffer();

    auto dst_buffer = output.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src_is_dram};
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    vector<uint32_t> compute_kernel_args_group_1 = {
        num_tiles_per_core_group_1,  // per_core_block_cnt
        1                            // per_core_block_size
    };

    bool math_approx_mode = std::all_of(
        op_chain.begin(), op_chain.end(), [](const auto &u) { return utils::get_op_approx_mode(u.op_type); });
    std::map<string, string> unary_defines = utils::get_block_defines(op_chain);
    auto eltwise_unary_kernel_group_1_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/kernels/compute/eltwise_sfpu.cpp",
        core_group_1,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .preserve_fp32_precision = preserve_fp32_precision,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args_group_1,
            .defines = unary_defines});

    std::cout << "math_fidelity: " << MathFidelity::HiFi4 << std::endl;
    std::cout << "fp32_dest_acc_en: " << fp32_dest_acc_en << std::endl;
    std::cout << "preserve_fp32_precision: " << preserve_fp32_precision << std::endl;
    std::cout << "math_approx_mode: " << math_approx_mode << std::endl;
    {
        std::cout << "compile_args: "; //<< compute_kernel_args_group_1 << std::endl;
        bool isFirst = true;
        for (auto const& value : compute_kernel_args_group_1) {
            std::cout << (isFirst ? "" : ", ") << value;
            isFirst = false;
        }
        std::cout << std::endl;
    }

    {
        bool isFirst = true;
        for (const auto& [key, value] :  unary_defines) {
            std::cout << (isFirst ? "" : ", ") << "[" << key << "] = " << value;
            isFirst = false;
        }
        std::cout << std::endl;
    }

    if (!core_group_2.ranges().empty()) {
        std::cout << "core_group_2 is not empty" << std::endl;

        vector<uint32_t> compute_kernel_args_group_2 = {
            num_tiles_per_core_group_2,  // per_core_block_cnt
            1                            // per_core_block_size
        };

        auto eltwise_unary_kernel_group_2_id = tt::tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/compute/eltwise_sfpu.cpp",
            core_group_2,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .preserve_fp32_precision = preserve_fp32_precision,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_kernel_args_group_2,
                .defines = unary_defines});
    }

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        tt::tt_metal::SetRuntimeArgs(
            program, unary_reader_kernel_id, core, {src_buffer->address(), num_tiles_per_core, num_tiles_written});

        tt::tt_metal::SetRuntimeArgs(
            program, unary_writer_kernel_id, core, {dst_buffer->address(), num_tiles_per_core, num_tiles_written});
        num_tiles_written += num_tiles_per_core;
    }

    auto override_runtime_args_callback = [unary_reader_kernel_id = std::move(unary_reader_kernel_id),
                                           unary_writer_kernel_id = std::move(unary_writer_kernel_id),
                                           num_cores,
                                           num_cores_y](
                                              const tt::tt_metal::Program &program,
                                              const std::vector<tt::tt_metal::Buffer *> &input_buffers,
                                              const std::vector<tt::tt_metal::Buffer *> &output_buffers) {
        auto src_buffer = input_buffers.at(0);

        auto dst_buffer = output_buffers.at(0);

        for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
            CoreCoord core = {i / num_cores_y, i % num_cores_y};

            {
                auto &runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
                runtime_args[0] = src_buffer->address();
            }

            {
                auto &runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
                runtime_args[0] = dst_buffer->address();
            }
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace ttnn::operations::unary
