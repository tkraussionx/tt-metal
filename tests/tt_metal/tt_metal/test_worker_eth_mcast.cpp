// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <random>

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "common/bfloat16.hpp"
#include "tt_metal/test_utils/deprecated/tensor.hpp"
// #include "tt_gdb/tt_gdb.hpp"


//////////////////////////////////////////////////////////////////////////////////////////
// TODO: explain what test does
//////////////////////////////////////////////////////////////////////////////////////////
using namespace tt;

int main(int argc, char **argv) {
    bool pass = true;

    auto slow_dispatch_mode = getenv("TT_METAL_SLOW_DISPATCH_MODE");
    TT_FATAL(slow_dispatch_mode, "This test only supports TT_METAL_SLOW_DISPATCH_MODE");

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    int device_id = 0;
    tt_metal::Device *device =
        tt_metal::CreateDevice(device_id);
    TT_FATAL(device->arch() == tt::ARCH::BLACKHOLE, "This test only supports Blackhole!");


    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::Program program = tt_metal::CreateProgram();

    auto eth_cores = device->get_inactive_ethernet_cores();
    uint32_t min_x=1000,min_y=1000,max_x=0,max_y=0;
    for (auto core : eth_cores) {
        auto physical = device->ethernet_core_from_logical_core(core);
        log_info(tt::LogTest, "ETH core {} {}", physical.x, physical.y);
        if (physical.x < min_x) min_x = physical.x;
        if (physical.x > max_x) max_x = physical.x;
        if (physical.y < min_y) min_y = physical.y;
        if (physical.y > max_y) max_y = physical.y;
    }

    max_x = 7;
    CoreCoord core = {0, 0};
    uint32_t single_tile_size = 2 * 1024;
    uint32_t num_tiles = 1;
    uint32_t dram_buffer_size = single_tile_size * num_tiles; // num_tiles of FP16_B, hard-coded in the reader/writer kernels

    uint32_t local_buffer_addr = 200 * 1024;

    // same address as local_buffer
    // Note: src will NOT write into its dst buffer address
    // since we are not setting NOC_CMD_BRCST_SRC_INCLUDE
    uint32_t dest_buffer_addr = 200 * 1024;

    tt_metal::InterleavedBufferConfig dram_config{
                            .device=device,
                            .size = dram_buffer_size,
                            .page_size = dram_buffer_size,
                            .buffer_type = tt_metal::BufferType::DRAM
                            };
    auto dram_buffer = CreateBuffer(dram_config);
    uint32_t dram_buffer_addr = dram_buffer->address();

    auto dram_noc_xy = dram_buffer->noc_coordinates();

    //CoreCoord core_start = *eth_cores.begin();
    //CoreCoord grid_size = device->logical_grid_size();
    //CoreCoord core_end = {core_start.x, core_start.y + eth_cores.size()};
    //auto core_start_physical = device->ethernet_core_from_logical_core(core_start);
    //auto core_end_physical = device->ethernet_core_from_logical_core(core_end);
    std::vector<uint32_t> mcast_reader_args = {
        (std::uint32_t)dram_buffer_addr,
        (std::uint32_t)dram_noc_xy.x,
        (std::uint32_t)dram_noc_xy.y,
        (std::uint32_t)dram_buffer_size,
        (std::uint32_t)local_buffer_addr,
        (std::uint32_t)dest_buffer_addr,
        (std::uint32_t)min_x,
        (std::uint32_t)min_y,
        (std::uint32_t)max_x,
        (std::uint32_t)max_y,
        (std::uint32_t)(eth_cores.size()-1)}; // Note: exclude src from acks, since we are not setting NOC_CMD_BRCST_SRC_INCLUDE

    log_debug(LogTest, "Start = {}, {}", min_x, min_y);
    log_debug(LogTest, "End = {}, {}", max_x, max_y);
    auto mcast_reader_kernel = tt_metal::CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_to_l1_multicast.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

    SHAPE shape = {1, 1, 32, 32};
    tt::deprecated::Tensor<bfloat16> tensor = tt::deprecated::initialize_tensor<bfloat16>(shape, tt::deprecated::Initialize::RANDOM, 100, std::chrono::system_clock::now().time_since_epoch().count());
    auto activations = pack_bfloat16_vec_into_uint32_vec(tensor.get_values());
    tt_metal::detail::WriteToBuffer(dram_buffer, activations);

    tt_metal::SetRuntimeArgs(program, mcast_reader_kernel, core, mcast_reader_args);

    log_debug(LogTest, "Launching kernels");
    tt_metal::detail::LaunchProgram(device, program);
    log_debug(LogTest, "Kernels done");
    log_info(tt::LogTest, "Kernels done {} {}", min_x, max_x);
    for(int i = min_y ; i <= max_y; i++) {
        for(int j = min_x ; j <= max_x; j++) {
            CoreCoord dest_core = {(std::size_t)j, (std::size_t)i};
            std::vector<uint32_t> dest_core_data;
            tt_metal::detail::ReadFromDeviceL1(device, dest_core, dest_buffer_addr, dram_buffer_size, dest_core_data);
            auto dest_core_data_unpacked = unpack_uint32_vec_into_bfloat16_vec(dest_core_data);
            pass &= (dest_core_data_unpacked == tensor.get_values());
            if(not (dest_core_data_unpacked == tensor.get_values())) {
                log_info(LogTest, "Mismatch on core {}, {}", dest_core.x, dest_core.y);
                print_vec_of_bfloat16(dest_core_data_unpacked, 1, "Result");
            }
        }
    }
    //return pass;

    if (pass) {
        log_info(LogTest, "Test Passed");
    } else {
        TT_THROW("Test Failed");
    }

    TT_FATAL(pass, "Error");

    return 0;
}
