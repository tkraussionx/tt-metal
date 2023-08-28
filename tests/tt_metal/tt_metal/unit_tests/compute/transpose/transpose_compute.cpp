#include <algorithm>
#include <functional>
#include <random>

#include "doctest.h"
#include "single_device_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"  // FIXME: Should remove dependency on this
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "tt_metal/test_utils/tilization.hpp"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::compute::transpose {
struct ReaderTransposeWithinTileWriterConfig {
    size_t num_tiles = 0;
    size_t tile_byte_size = 0;
    size_t output_dram_byte_address = 0;
    size_t input_dram_byte_address = 0;
    size_t l1_input_byte_address = 0;
    tt::DataFormat l1_input_data_format = tt::DataFormat::Invalid;
    size_t l1_output_byte_address = 0;
    tt::DataFormat l1_output_data_format = tt::DataFormat::Invalid;
    CoreCoord core = {};
};
/// @brief Does Dram --> Reader --> CB --> Datacopy --> CB --> Writer --> Dram on a single core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool reader_transpose_within_tile_writer(
    tt_metal::Device* device, const ReaderTransposeWithinTileWriterConfig& test_config) {
    // Once this test is uplifted to use fast dispatch, this can be removed.
    tt::tt_metal::detail::GLOBAL_CQ.reset();
    char env[] = "TT_METAL_SLOW_DISPATCH_MODE=1";
    putenv(env);

    bool pass = true;
    const uint32_t r = 32*test_config.num_tiles;
    const uint32_t c = 32;
    const uint32_t input0_cb_index = 0;
    const uint32_t output_cb_index = 16;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;
    tt_metal::Program program = tt_metal::Program();
    auto input_dram_buffer =
        tt_metal::Buffer(device, byte_size, test_config.input_dram_byte_address, byte_size, tt_metal::BufferType::DRAM);
    auto input_dram_noc_xy = input_dram_buffer.noc_coordinates();
    auto output_dram_buffer = tt_metal::Buffer(
        device, byte_size, test_config.output_dram_byte_address, byte_size, tt_metal::BufferType::DRAM);
    auto output_dram_noc_xy = output_dram_buffer.noc_coordinates();

    auto l1_input_cb = tt_metal::CreateCircularBuffer(
        program,
        input0_cb_index,
        test_config.core,
        test_config.num_tiles,
        byte_size,
        test_config.l1_input_data_format,
        test_config.l1_input_byte_address);
    auto l1_output_cb = tt_metal::CreateCircularBuffer(
        program,
        output_cb_index,
        test_config.core,
        test_config.num_tiles,
        byte_size,
        test_config.l1_output_data_format,
        test_config.l1_output_byte_address);

    auto reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp",
        test_config.core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = {input0_cb_index}});

    auto writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp",
        test_config.core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {output_cb_index}});

    vector<uint32_t> compute_kernel_args = {
        uint(test_config.num_tiles)  // per_core_tile_cnt
    };
    auto datacopy_kernel = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/unit_tests/transpose/within_tile_transpose.cpp",
        test_config.core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<bfloat16> inputs = tt::test_utils::generate_strided_vector<bfloat16>(
        0.0f, 1.0f, 8, 7, (byte_size / bfloat16::SIZEOF));
    std::vector<bfloat16> expected_outputs ((byte_size / bfloat16::SIZEOF), 0.0f);
    for (int i = 0; i < inputs.size(); i++) {
        int row = i / 32;
        if ((row % 8) == 7) {
            expected_outputs.at(i) = bfloat16(1.0f);
        }
    }
    auto inputs_tilized = tt::test_utils::tilize<bfloat16, 16, 16>(inputs, r, c);
    auto inputs_tilized_packed = tt::test_utils::pack_vector<uint32_t, bfloat16> (inputs_tilized);
    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////
    pass &= tt_metal::CompileProgram(device, program);
    tt_metal::WriteToBuffer(input_dram_buffer, inputs_tilized_packed);

    pass &= tt_metal::ConfigureDeviceWithProgram(device, program);
    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        test_config.core,
        {
            (uint32_t)test_config.input_dram_byte_address,
            (uint32_t)input_dram_noc_xy.x,
            (uint32_t)input_dram_noc_xy.y,
            (uint32_t)test_config.num_tiles,
        });
    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        test_config.core,
        {
            (uint32_t)test_config.output_dram_byte_address,
            (uint32_t)output_dram_noc_xy.x,
            (uint32_t)output_dram_noc_xy.y,
            (uint32_t)test_config.num_tiles,
        });
    tt_metal::WriteRuntimeArgsToDevice(device, program);
    pass &= tt_metal::LaunchKernels(device, program);

    std::vector<uint32_t> dest_buffer_data_packed_tilized;
    tt_metal::ReadFromBuffer(output_dram_buffer, dest_buffer_data_packed_tilized);
    auto dest_buffer_data_tilized = tt::test_utils::unpack_vector<bfloat16, uint32_t> (dest_buffer_data_packed_tilized);
    auto dest_buffer_data = tt::test_utils::untilize<bfloat16, 16, 16>(dest_buffer_data_tilized, r, c);
    pass &= expected_outputs == dest_buffer_data;
    log_info ("expected_outputs");
    tt::test_utils::print_vector_fixed_numel_per_row<bfloat16>(
        expected_outputs,
        32);
    log_info ("Outputs tilized");
    tt::test_utils::print_vector_fixed_numel_per_row<bfloat16>(
        dest_buffer_data_tilized,
        32);
    log_info ("Outputs");
    tt::test_utils::print_vector_fixed_numel_per_row<bfloat16>(
        dest_buffer_data,
        32);
    return pass;
}
}  // namespace unit_tests::compute::transpose

TEST_SUITE("SingleCoreTranspose") {
    TEST_CASE_FIXTURE(unit_tests::SingleDeviceFixture, "ReaderTransposeWithinTileWriter") {
        unit_tests::compute::transpose::ReaderTransposeWithinTileWriterConfig test_config = {
            .num_tiles = 1,
            .tile_byte_size = 2 * 32 * 32,
            .output_dram_byte_address = 0,
            .input_dram_byte_address = 16 * 32 * 32,
            .l1_input_byte_address = UNRESERVED_BASE,
            .l1_input_data_format = tt::DataFormat::Float16_b,
            .l1_output_byte_address = UNRESERVED_BASE + 16 * 32 * 32,
            .l1_output_data_format = tt::DataFormat::Float16_b,
            .core = {.x = 0, .y = 0}};
        SUBCASE("SingleTile") {
            test_config.num_tiles = 1;
            REQUIRE(unit_tests::compute::transpose::reader_transpose_within_tile_writer(device_, test_config));
        }
        SUBCASE("MultiTile") {
            test_config.num_tiles = 4;
            REQUIRE(unit_tests::compute::transpose::reader_transpose_within_tile_writer(device_, test_config));
            test_config.num_tiles = 8;
            REQUIRE(unit_tests::compute::transpose::reader_transpose_within_tile_writer(device_, test_config));
        }
    }
}
