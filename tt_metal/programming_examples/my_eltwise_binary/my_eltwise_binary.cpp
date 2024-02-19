// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <memory>
#include <random>

#include "base_types.hpp"
#include "common/bfloat16.hpp"
#include "core_coord.h"
#include "impl/buffers/buffer.hpp"
#include "impl/buffers/circular_buffer_types.hpp"
#include "impl/dispatch/command_queue.hpp"
#include "impl/kernels/data_types.hpp"
#include "impl/kernels/kernel_types.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_backend_api_types.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt;
using namespace tt::tt_metal;

/*
 * 1. Host creates two vectors of data.
 * 2. Device eltwise adds them together.
 * 3. Intermediate result read back to host.
 * 4. Create another vector and send vectors to input DRAMs again.
 * 5. Device eltwise muls them together.
 * 6. Read result back and compare to golden.
 * */

/*
 * We need to copy the types of the compute kernel arguments to use them host-
 * side.
 */

int main(int argc, char **argv) {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    bool pass = true;

    try {
        {
            // create device
            constexpr int device_id = 0;
            Device *device = CreateDevice(device_id);

            // Setup command queue
            CommandQueue &cq = tt::tt_metal::detail::GetCommandQueue(device);
            Program program = CreateProgram();

            constexpr CoreCoord core = {0, 0};
            constexpr uint32_t single_tile_size_in_bytes = 2 * 32 * 32;
            constexpr uint32_t num_tiles = 32;
            constexpr uint32_t dram_buffer_size_in_bytes = num_tiles * single_tile_size_in_bytes;

            tt_metal::InterleavedBufferConfig dram_config{
                .device = device,
                .size = dram_buffer_size_in_bytes,
                .page_size = dram_buffer_size_in_bytes,
                .buffer_type = tt_metal::BufferType::DRAM};

            std::shared_ptr<tt::tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config);
            std::shared_ptr<tt::tt_metal::Buffer> src1_dram_buffer = CreateBuffer(dram_config);
            std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config);

            /// Create circular buffers

            // Create circular buffer
            constexpr uint32_t src0_cb_index = CB::c_in0;
            constexpr uint32_t num_input_tiles = 2;
            CircularBufferConfig cb_src0_config =
                CircularBufferConfig(
                    num_input_tiles * single_tile_size_in_bytes, {{src0_cb_index, tt::DataFormat::Float16_b}})
                    .set_page_size(src0_cb_index, single_tile_size_in_bytes);
            CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

            constexpr uint32_t src1_cb_index = CB::c_in1;
            CircularBufferConfig cb_src1_config =
                CircularBufferConfig(
                    num_input_tiles * single_tile_size_in_bytes, {{src1_cb_index, tt::DataFormat::Float16_b}})
                    .set_page_size(src1_cb_index, single_tile_size_in_bytes);
            CBHandle cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

            constexpr uint32_t output_cb_index = CB::c_out0;
            constexpr uint32_t num_output_tiles = 2;
            CircularBufferConfig cb_output_config =
                CircularBufferConfig(
                    num_output_tiles * single_tile_size_in_bytes, {{output_cb_index, tt::DataFormat::Float16_b}})
                    .set_page_size(output_cb_index, single_tile_size_in_bytes);
            CBHandle cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

            // Create datamovement kernels
            KernelHandle reader_kernel = CreateKernel(
                program,
                "tt_metal/kernels/dataflow/reader_binary_diff_lengths.cpp",
                core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

            KernelHandle writer_kernel = CreateKernel(
                program,
                "tt_metal/kernels/dataflow/writer_unary.cpp",
                core,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

            vector<uint32_t> compute_kernel_args{};
            map<string, string> defines{};

            KernelHandle compute_handle = CreateKernel(
                program,
                "tt_metal/kernels/compute/my_binary.cpp",
                core,
                ComputeConfig{
                    .math_fidelity = MathFidelity::HiFi4,
                    .fp32_dest_acc_en = false,
                    .math_approx_mode = false,
                    .compile_args = compute_kernel_args,
                    .defines = defines});

            // create source data and write to dram
            std::vector<uint32_t> src0_vec = create_constant_vector_of_bfloat16(dram_buffer_size_in_bytes, -5.0f);
            EnqueueWriteBuffer(cq, src0_dram_buffer, src0_vec, true);

            std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size_in_bytes, -3.0f);
            EnqueueWriteBuffer(cq, src1_dram_buffer, src1_vec, true);

            // Set runtime args for reader kernel
            SetRuntimeArgs(
                program,
                reader_kernel,
                core,
                {src0_dram_buffer->address(),
                 static_cast<uint32_t>(src0_dram_buffer->noc_coordinates().x),
                 static_cast<uint32_t>(src0_dram_buffer->noc_coordinates().y),
                 num_tiles,
                 src1_dram_buffer->address(),
                 static_cast<uint32_t>(src1_dram_buffer->noc_coordinates().x),
                 static_cast<uint32_t>(src1_dram_buffer->noc_coordinates().y),
                 num_tiles,
                 0});

            log_info(tt::LogVerif, "1");

            SetRuntimeArgs(program, compute_handle, core, {num_tiles, 1});

            log_info(tt::LogVerif, "2");

            SetRuntimeArgs(
                program,
                writer_kernel,
                core,
                {dst_dram_buffer->address(),
                 static_cast<uint32_t>(dst_dram_buffer->noc_coordinates().x),
                 static_cast<uint32_t>(dst_dram_buffer->noc_coordinates().y),
                 num_tiles});

            log_info(tt::LogVerif, "3");

            EnqueueProgram(cq, program, true);

            log_info(tt::LogVerif, "4");

            Finish(cq);

            log_info(tt::LogVerif, "5");

            // Read result into host vec
            std::vector<uint32_t> result_vec;
            EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);

            log_info(tt::LogVerif, "6");

            std::vector<uint32_t> golden_result_vec =
                create_constant_vector_of_bfloat16(dram_buffer_size_in_bytes, -8.0f);

            constexpr float abs_tolerance = 0.01f;
            constexpr float rel_tolerance = 0.001f;
            std::function<bool(const float, const float)> comparison_function = [](const float a, const float b) {
                return is_close(a, b, rel_tolerance, abs_tolerance);
            };

            pass &= packed_uint32_t_vector_comparison(golden_result_vec, result_vec, comparison_function);

            log_info(tt::LogVerif, "7");

            pass &= CloseDevice(device);
        }
    } catch (const std::exception &e) {
        tt::log_error(tt::LogTest, "Test failed with exception!");
        tt::log_error(tt::LogTest, "{}", e.what());

        throw;
    }

    if (pass) {
        tt::log_info(tt::LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass);
    return 0;
}
