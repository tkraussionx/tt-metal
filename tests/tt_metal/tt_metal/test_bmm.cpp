// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "common/bfloat16.hpp"
#include "test_gold_impls.hpp"

using namespace tt;
using namespace tt::tt_metal;

// TODO: move it!
std::string to_string(BufferType btype) {
    switch(btype) {
        case BufferType::DRAM:
            return "DRAM";
        case BufferType::L1:
            return "L1";
        case BufferType::L1_SMALL:
            return "L1_SMALL";
        default:
            return "other";
    }
}
std::string to_string(TensorMemoryLayout layout) {
    switch(layout) {
        case TensorMemoryLayout::INTERLEAVED:
            return "INTERLEAVED";
        case TensorMemoryLayout::SINGLE_BANK:
            return "SINGLE_BANK";
        case TensorMemoryLayout::HEIGHT_SHARDED:
            return "HEIGHT_SHARDED";
        case TensorMemoryLayout::WIDTH_SHARDED:
            return "WIDTH_SHARDED";
        case TensorMemoryLayout::BLOCK_SHARDED:
            return "BLOCK_SHARDED";
        default:
            return "other";
    }
}

void dump_json(std::string opath, const nlohmann::json& ojson) {
    std::ofstream out(opath);
    if (out.fail()) {
        throw std::runtime_error("output file open failure");
    }
    std::string summaries = ojson.dump(2);
    out << summaries << std::endl;
    out.close();
}

namespace tt::stl::json {
// TODO: move to buffer.hpp?
template <>
struct to_json_t<std::shared_ptr<tt::tt_metal::Buffer>> {
    nlohmann::json operator()(std::shared_ptr<tt::tt_metal::Buffer> buffer) { // TODO: what happed for having except?
        nlohmann::json ojson;
        ojson["device id"] = buffer->device()->id(); // TODO: include device breaks compilation (in buffer.hpp)
        ojson["total bytes"] = buffer->size();
        ojson["address"] = buffer->address();
        ojson["page size (B)"] = buffer->page_size();
        ojson["buffer type"] = to_string(buffer->buffer_type());
        ojson["tensor memory layout"] = to_string(buffer->buffer_layout());
        if (is_sharded(buffer->buffer_layout())) { // TODO: need to test it
            ojson["shard spec"] = to_json(buffer->shard_spec().tensor_shard_spec);
        } else {
            ojson["shard spec"] = "na";
        }
        return ojson;
    }
};
}

//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    bool pass = true;

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    log_info(LogTest, "====================================================================");
    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Device Setup
        ////////////////////////////////////////////////////////////////////////////
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);

        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
        tt_metal::Program program = tt_metal::CreateProgram();

        CoreCoord core = {0, 0};
        uint32_t single_tile_size = 2 * 1024;
        uint32_t Mt = 4, Kt = 2, Nt = 3, B = 2;
        uint32_t num_tilesA = Mt*Kt*B;
        uint32_t num_tilesB = Mt*Kt*B;
        uint32_t num_tilesC = Mt*Nt*B;
        uint32_t bytesA = single_tile_size * num_tilesA;
        uint32_t bytesB = single_tile_size * num_tilesB;
        uint32_t bytesC = single_tile_size * num_tilesC;

        tt_metal::InterleavedBufferConfig src0_config{
                                        .device=device,
                                        .size = bytesA,
                                        .page_size = single_tile_size,
                                        .buffer_type = tt_metal::BufferType::DRAM
                                        };

        auto src0_dram_buffer = CreateBuffer(src0_config);
        uint32_t dram_buffer_src0_addr = src0_dram_buffer->address();

        tt_metal::InterleavedBufferConfig src1_config{
                                        .device=device,
                                        .size = bytesB,
                                        .page_size = single_tile_size,
                                        .buffer_type = tt_metal::BufferType::DRAM
                                        };

        auto src1_dram_buffer = CreateBuffer(src1_config);
        uint32_t dram_buffer_src1_addr = src1_dram_buffer->address();

        tt_metal::InterleavedBufferConfig dst_config{
                                        .device=device,
                                        .size = bytesC,
                                        .page_size = single_tile_size,
                                        .buffer_type = tt_metal::BufferType::DRAM
                                        };
        auto dst_dram_buffer = CreateBuffer(dst_config);
        uint32_t dram_buffer_dst_addr = dst_dram_buffer->address();

        uint32_t src0_cb_index = 0;
        uint32_t num_input_tiles = 2;
        tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
        auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t src1_cb_index = 1;
        tt_metal::CircularBufferConfig cb_src1_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, single_tile_size);
        auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t num_output_tiles = 2;
        tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{ouput_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(ouput_cb_index, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        bool src0_is_dram = true;
        bool src1_is_dram = true;
        std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

        bool dst_is_dram = true;
        std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};
        auto reader = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/reader_bmm_8bank.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

        auto writer = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/writer_bmm_8bank.cpp",
            core,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

        vector<uint32_t> compute_kernel_args = {
            B, // batch
            Mt, // Mt
            Kt, // Kt
            Nt // Nt
        };

        auto eltwise_binary_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/compute/bmm.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args}
        );



        std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(bytesA, 1.0f, 0x1234);
        std::vector<uint32_t> src1_vec = create_random_vector_of_bfloat16(bytesB, 1.0f, 0x1234, -0.45f);
        tt_metal::detail::WriteToBuffer(src0_dram_buffer, src0_vec);
        tt_metal::detail::WriteToBuffer(src1_dram_buffer, src1_vec);

        uint32_t do_bcast = 0;
        tt_metal::SetRuntimeArgs(
            program, reader, core,
            {dram_buffer_src0_addr, dram_buffer_src1_addr, Mt, Kt, Nt, Mt*Kt, Kt*Nt, B, do_bcast}
        );
        tt_metal::SetRuntimeArgs(
            program, writer, core,
            {dram_buffer_dst_addr, 0, Mt, Kt, Nt, Mt*Kt, Kt*Nt, B}
        );

        tt_metal::detail::LaunchProgram(device, program);

        program.dump_circular_buffer_info("/tmp");
        dump_json("/tmp/src0_buffer_info.json", tt::stl::json::to_json(src0_dram_buffer));
        dump_json("/tmp/src1_buffer_info.json", tt::stl::json::to_json(src1_dram_buffer));
        dump_json("/tmp/dst_buffer_info.json", tt::stl::json::to_json(dst_dram_buffer));

        std::vector<uint32_t> result_vec;
        tt_metal::detail::ReadFromBuffer(dst_dram_buffer, result_vec);

        {
            // Read the result back from device DRAM and ref comparisone
            int argfail = -1;
            auto comparison_function = [](float a, float b) {
                const float rtol = 0.05f; // TODO(AP): need a spec for reference
                const float atol = 0.05f;
                float maxabs = fmaxf(fabsf(a), fabsf(b));
                float absdiff = fabsf(a - b);
                auto result = (absdiff <= atol) || absdiff < rtol * maxabs;
                return result;
            };

            // recover a linear view of input vector for consumption by gold_ function
            vector<uint32_t> shapeA = {1, B, Mt*32, Kt*32};
            vector<uint32_t> shapeB = {1, B, Kt*32, Nt*32};
            vector<uint32_t> shapeC = {1, B, Mt*32, Nt*32};
            auto u16_src0_vec = u16_from_u32_vector(src0_vec);
            auto u16_src1_vec = u16_from_u32_vector(src1_vec);
            vector<uint16_t> src0_linear = convert_layout<uint16_t>(u16_src0_vec, shapeA, TensorLayout::TILED_NFACES, TensorLayout::LIN_ROW_MAJOR);
            vector<uint16_t> src1_linear = convert_layout<uint16_t>(u16_src1_vec, shapeB, TensorLayout::TILED_NFACES, TensorLayout::LIN_ROW_MAJOR);
            vector<uint16_t> ref_bmm = gold_bmm(shapeA, src0_linear, shapeB, src1_linear);

            // Tilize gold from row major and convert to pairs (uint32_t)
            auto gold_4f_u32 = u32_from_u16_vector( convert_layout<uint16_t>(
                ref_bmm, shapeC, TensorLayout::LIN_ROW_MAJOR, TensorLayout::TILED_NFACES));

            pass &= packed_uint32_t_vector_comparison(result_vec, gold_4f_u32, comparison_function, &argfail);
            if (!pass)
                log_error(LogTest, "Failure position={}", argfail);

        }
        //pass &= (src0_vec == result_vec);
        pass &= tt_metal::CloseDevice(device);

    } catch (const std::exception &e) {
        pass = false;
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass, "Error");

    return 0;
}
