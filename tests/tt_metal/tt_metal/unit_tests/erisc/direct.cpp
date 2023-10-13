// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <functional>
#include <random>

#include "device_fixture.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"  // FIXME: Should remove dependency on this
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

constexpr std::int32_t FIRMWARE_BASE = 0x9040;
constexpr std::int32_t RUN_APP_FLAG = 0x10E8;
constexpr std::int32_t DATA_BUFFER_SPACE_BASE = 0x28000;
constexpr std::int32_t ETH_L1_ARGS_BASE = 0x3E000;

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::erisc::direct {
/// @brief Does Dram --> Reader --> L1 on a single core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool reader_only(
    tt_metal::Device* device,
    const size_t& byte_size,
    const size_t& l1_byte_address,
    const CoreCoord& reader_core) {

    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();

    auto input_dram_buffer = CreateBuffer(device, byte_size, byte_size, tt_metal::BufferType::DRAM);
    uint32_t dram_byte_address = input_dram_buffer.address();
    auto dram_noc_xy = input_dram_buffer.noc_coordinates();
    // TODO (abhullar): Use L1 buffer after bug with L1 banking and writing to < 1 MB is fixed.
    //                  Try this after KM uplifts TLB setup
    // auto l1_buffer =
    //     CreateBuffer(device, byte_size, l1_byte_address, byte_size, tt_metal::BufferType::L1);

    auto reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/unit_tests/dram/direct_reader_dram_to_l1.cpp",
        reader_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
    tt_metal::WriteToBuffer(input_dram_buffer, inputs);

    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        reader_core,
        {
            (uint32_t)dram_byte_address,
            (uint32_t)dram_noc_xy.x,
            (uint32_t)dram_noc_xy.y,
            (uint32_t)l1_byte_address,
            (uint32_t)byte_size,
        });

    tt_metal::LaunchProgram(device, program);

    std::vector<uint32_t> dest_core_data;
    // tt_metal::ReadFromBuffer(l1_buffer, dest_core_data);
    tt_metal::detail::ReadFromDeviceL1(device, reader_core, l1_byte_address, byte_size, dest_core_data);
    pass &= (dest_core_data == inputs);
    if (not pass) {
        std::cout << "Mismatch at Core: " << reader_core.str() << std::endl;
    }
    return pass;
}

/// @brief Does L1 --> Writer --> Dram on a single core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool writer_only(
    tt_metal::Device* device,
    const size_t& byte_size,
    const size_t& l1_byte_address,
    const CoreCoord& writer_core) {

    bool pass = true;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::Program();

    auto output_dram_buffer = CreateBuffer(device, byte_size, byte_size, tt_metal::BufferType::DRAM);
    uint32_t dram_byte_address = output_dram_buffer.address();
    auto dram_noc_xy = output_dram_buffer.noc_coordinates();
    auto l1_bank_ids = device->bank_ids_from_logical_core(writer_core);
    // TODO (abhullar): Use L1 buffer after bug with L1 banking and writing to < 1 MB is fixed.
    //                  Try this after KM uplifts TLB setup
    // auto l1_buffer =
    //     CreateBuffer(device, byte_size, l1_byte_address, byte_size, tt_metal::BufferType::L1);

    auto writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/unit_tests/dram/direct_writer_l1_to_dram.cpp",
        writer_core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
    tt_metal::detail::WriteToDeviceL1(device, writer_core, l1_byte_address, inputs);
    // tt_metal::WriteToBuffer(l1_buffer, inputs);


    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        writer_core,
        {
            (uint32_t)dram_byte_address,
            (uint32_t)dram_noc_xy.x,
            (uint32_t)dram_noc_xy.y,
            (uint32_t)l1_byte_address,
            (uint32_t)byte_size,
        });

    tt_metal::LaunchProgram(device, program);

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    pass &= (dest_buffer_data == inputs);
    if (not pass) {
        std::cout << "Mismatch at Core: " << writer_core.str() << std::endl;
    }
    return pass;
}

struct ReaderWriterConfig {
    size_t num_tiles = 0;
    size_t tile_byte_size = 0;
    tt::DataFormat l1_data_format = tt::DataFormat::Invalid;
    CoreCoord core = {};
};
/// @brief Does Dram --> Reader --> CB --> Writer --> Dram on a single core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool reader_writer(tt_metal::Device* device, const ReaderWriterConfig& test_config) {

    bool pass = true;

    const uint32_t cb_index = 0;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;
    tt_metal::Program program = tt_metal::Program();
    auto input_dram_buffer = CreateBuffer(device, byte_size, byte_size, tt_metal::BufferType::DRAM);
    uint32_t input_dram_byte_address = input_dram_buffer.address();
    auto input_dram_noc_xy = input_dram_buffer.noc_coordinates();
    auto output_dram_buffer = CreateBuffer(device, byte_size, byte_size, tt_metal::BufferType::DRAM);
    uint32_t output_dram_byte_address = output_dram_buffer.address();
    auto output_dram_noc_xy = output_dram_buffer.noc_coordinates();

    tt_metal::CircularBufferConfig l1_cb_config = tt_metal::CircularBufferConfig(byte_size, {{cb_index, test_config.l1_data_format}})
        .set_page_size(cb_index, test_config.tile_byte_size);
    auto l1_cb = tt_metal::CreateCircularBuffer(program, test_config.core, l1_cb_config);

    auto reader_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/unit_tests/dram/direct_reader_unary.cpp",
        test_config.core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = {cb_index}});

    auto writer_kernel = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/unit_tests/dram/direct_writer_unary.cpp",
        test_config.core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {cb_index}});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> inputs = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -1.0f, 1.0f, byte_size / bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());
    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::WriteToBuffer(input_dram_buffer, inputs);


    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        test_config.core,
        {
            (uint32_t)input_dram_byte_address,
            (uint32_t)input_dram_noc_xy.x,
            (uint32_t)input_dram_noc_xy.y,
            (uint32_t)test_config.num_tiles,
        });
    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        test_config.core,
        {
            (uint32_t)output_dram_byte_address,
            (uint32_t)output_dram_noc_xy.x,
            (uint32_t)output_dram_noc_xy.y,
            (uint32_t)test_config.num_tiles,
        });

    tt_metal::LaunchProgram(device, program);

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    pass &= inputs == dest_buffer_data;
    return pass;
}
struct ReaderDatacopyWriterConfig {
    size_t num_tiles = 0;
    size_t tile_byte_size = 0;
    tt::DataFormat l1_input_data_format = tt::DataFormat::Invalid;
    tt::DataFormat l1_output_data_format = tt::DataFormat::Invalid;
    CoreCoord core = {};
};
/// @brief Does Dram --> Reader --> CB --> Datacopy --> CB --> Writer --> Dram on a single core
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool reader_datacopy_writer(tt_metal::Device* device, const ReaderDatacopyWriterConfig& test_config) {

    bool pass = true;

    const uint32_t input0_cb_index = 0;
    const uint32_t output_cb_index = 16;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    const size_t byte_size = test_config.num_tiles * test_config.tile_byte_size;
    tt_metal::Program program = tt_metal::Program();
    auto input_dram_buffer = CreateBuffer(device, byte_size, byte_size, tt_metal::BufferType::DRAM);
    uint32_t input_dram_byte_address = input_dram_buffer.address();
    auto input_dram_noc_xy = input_dram_buffer.noc_coordinates();
    auto output_dram_buffer = CreateBuffer(device, byte_size, byte_size, tt_metal::BufferType::DRAM);
    uint32_t output_dram_byte_address = output_dram_buffer.address();
    auto output_dram_noc_xy = output_dram_buffer.noc_coordinates();

    tt_metal::CircularBufferConfig l1_input_cb_config = tt_metal::CircularBufferConfig(byte_size, {{input0_cb_index, test_config.l1_input_data_format}})
        .set_page_size(input0_cb_index, test_config.tile_byte_size);
    auto l1_input_cb = tt_metal::CreateCircularBuffer(program, test_config.core, l1_input_cb_config);

    tt_metal::CircularBufferConfig l1_output_cb_config = tt_metal::CircularBufferConfig(byte_size, {{output_cb_index, test_config.l1_output_data_format}})
        .set_page_size(output_cb_index, test_config.tile_byte_size);
    auto l1_output_cb = tt_metal::CreateCircularBuffer(program, test_config.core, l1_output_cb_config);

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
        "tt_metal/kernels/compute/eltwise_copy.cpp",
        test_config.core,
        tt_metal::ComputeConfig{.compile_args = compute_kernel_args});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> inputs = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -1.0f, 1.0f, byte_size / bfloat16::SIZEOF, std::chrono::system_clock::now().time_since_epoch().count());
    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::WriteToBuffer(input_dram_buffer, inputs);


    tt_metal::SetRuntimeArgs(
        program,
        reader_kernel,
        test_config.core,
        {
            (uint32_t)input_dram_byte_address,
            (uint32_t)input_dram_noc_xy.x,
            (uint32_t)input_dram_noc_xy.y,
            (uint32_t)test_config.num_tiles,
        });
    tt_metal::SetRuntimeArgs(
        program,
        writer_kernel,
        test_config.core,
        {
            (uint32_t)output_dram_byte_address,
            (uint32_t)output_dram_noc_xy.x,
            (uint32_t)output_dram_noc_xy.y,
            (uint32_t)test_config.num_tiles,
        });

    tt_metal::LaunchProgram(device, program);

    std::vector<uint32_t> dest_buffer_data;
    tt_metal::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    pass &= inputs == dest_buffer_data;
    return pass;
}
    /// @brief Does eth L1 chip 0 --> eth L1 chip 1
/// @param device
/// @param test_config - Configuration of the test -- see struct
/// @return
bool send_over_eth(
    tt_metal::Device* sender_device,
    tt_metal::Device* receiver_device,
    const CoreCoord& sender_core,
    const CoreCoord& receiver_core,
    const size_t& byte_size) {
  auto inputs = generate_uniform_random_vector<uint32_t>(0, 100, byte_size / sizeof(uint32_t));
  for ( const auto& v: inputs) {
    std::cout << "inputs " << v << std::endl;
  }
  uint32_t arg_0 = byte_size >> 4; // number of 16 byte sends
  std::vector<uint32_t> args = {arg_0};
  llrt::write_hex_vec_to_core(sender_device->id(), sender_core, args, ETH_L1_ARGS_BASE);

  llrt::write_hex_vec_to_core(sender_device->id(), sender_core, inputs, DATA_BUFFER_SPACE_BASE);

  ll_api::memory binary_mem_send = llrt::get_risc_binary("erisc_app.hex", sender_device->id(), true);
  ll_api::memory binary_mem_receive = llrt::get_risc_binary("erisc_app.hex", receiver_device->id(), true);

  std::vector<CoreCoord> eth_cores = {{.x = 9, .y = 0}, {.x = 1, .y = 0}, {.x = 8, .y = 0}, {.x = 2, .y = 0}, {.x = 9, .y = 6}, {.x = 1, .y = 6}, {.x = 8, .y = 6}, {.x = 2, .y = 6}, {.x = 7, .y = 0}, {.x = 3, .y = 0}, {.x = 6, .y = 0}, {.x = 4, .y = 0}, {.x = 7, .y = 6}, {.x = 3, .y = 6}, {.x = 6, .y = 6}, {.x = 4, .y = 6}};

  /*for (const auto& eth_core: eth_cores) {
    std::cout << " eth_core " << eth_core.x << " " << eth_core.y << std::endl;
    std::vector<uint32_t> reset_fw_base;
    for (int i=0; i<200; i++) {
      reset_fw_base.push_back(0xffffffff);
    }
    llrt::write_hex_vec_to_core(sender_device->id(), eth_core, reset_fw_base, FIRMWARE_BASE);
    llrt::write_hex_vec_to_core(receiver_device->id(), eth_core, reset_fw_base, FIRMWARE_BASE);
  }*/
  for (const auto& eth_core: eth_cores) {
    std::cout << " eth_core " << eth_core.x << " " << eth_core.y << std::endl;
    llrt::write_hex_vec_to_core(sender_device->id(), eth_core, binary_mem_send.data(), FIRMWARE_BASE);
    llrt::write_hex_vec_to_core(receiver_device->id(), eth_core, binary_mem_receive.data(), FIRMWARE_BASE);
  }

    std::vector<uint32_t> run_test_app_flag = {0x1};
    llrt::write_hex_vec_to_core(sender_device->id(), sender_core, run_test_app_flag, RUN_APP_FLAG);
//    llrt::write_hex_vec_to_core(receiver_device->id(), receiver_core, run_test_app_flag, 0x10E8);

  bool pass = true;
  auto readback_vec =  llrt::read_hex_vec_from_core(receiver_device->id(), receiver_core, DATA_BUFFER_SPACE_BASE , byte_size);
  for ( const auto& v: readback_vec) {
    std::cout << "readback " << v << std::endl;
  }
    run_test_app_flag = {0x0};
    llrt::write_hex_vec_to_core(sender_device->id(), sender_core, run_test_app_flag, RUN_APP_FLAG);

  return pass;

}
}  // namespace unit_tests::erisc::direct

TEST_F(DeviceFixture, SingleEthCoreDirectSend) {
  ASSERT_TRUE(this->num_devices_ == 2);
  const auto& device_0 = devices_.at(0);
  const auto& device_1 = devices_.at(1);
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_0, device_1, {.x = 9, .y = 6}, {.x = 9, .y = 0}, 48));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_0, device_1, {.x = 1, .y = 6}, {.x = 1, .y = 0}, 48));
  ASSERT_TRUE(
    unit_tests::erisc::direct::send_over_eth(device_0, device_1, {.x = 1, .y = 6}, {.x = 1, .y = 0}, 64));
 // ASSERT_TRUE(
   // unit_tests::erisc::direct::send_over_eth(device_0, device_1, {.x = 9, .y = 6}, {.x = 9, .y = 0}, 64));
 // ASSERT_TRUE(
 //   unit_tests::erisc::direct::send_over_eth(device_0, device_1, {.x = 9, .y = 6}, {.x = 9, .y = 0}, 256));
}
TEST_F(DeviceFixture, SingleEthCoreDirectSend2) {
  ASSERT_TRUE(this->num_devices_ == 2);
  const auto& device_0 = devices_.at(0);
  const auto& device_1 = devices_.at(1);
  //ASSERT_TRUE(
  //  unit_tests::erisc::direct::send_over_eth(device_0, device_1, {.x = 9, .y = 6}, {.x = 9, .y = 0}, 48));
//  ASSERT_TRUE(
 //   unit_tests::erisc::direct::send_over_eth(device_0, device_1, {.x = 9, .y = 6}, {.x = 9, .y = 0}, 64));
}
