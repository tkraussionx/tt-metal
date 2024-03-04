// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/llrt/rtoptions.hpp"
#include "tt_metal/impl/dispatch/kernels/cq_cmds.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"

using namespace tt;


int main(int argc, char **argv) {

    // init(argc, argv);

    bool pass = true;
    try {
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);

        CommandQueue& cq = device->command_queue();

        tt_metal::Program program = tt_metal::CreateProgram();

        CoreCoord traffic_gen_tx_core = {0, 0};
        CoreCoord traffic_gen_rx_core = {1, 0};

        CoreCoord mux_core = {0, 1};
        CoreCoord demux_core = {0, 2};

        CoreCoord phys_traffic_gen_tx_core = device->worker_core_from_logical_core(traffic_gen_tx_core);
        CoreCoord phys_traffic_gen_rx_core = device->worker_core_from_logical_core(traffic_gen_rx_core);

        CoreCoord phys_mux_core = device->worker_core_from_logical_core(mux_core);
        CoreCoord phys_demux_core = device->worker_core_from_logical_core(demux_core);

        std::vector<uint32_t> traffic_gen_tx_compile_args =
            {
                0x1, // input_queue_id
                0x2, // output_queue_id
                (0x80000 >> 4), // queue_start_addr_words
                (0x10000 >> 4), // queue_size_words
                (0x90000 >> 4), // remote_rx_queue_start_addr_words
                (0x20000 >> 4), // remote_rx_queue_size_words
                (uint32_t)phys_traffic_gen_rx_core.x, // remote_rx_x
                (uint32_t)phys_traffic_gen_rx_core.y, // remote_rx_y
                0x3, // remote_rx_queue_id
                (uint32_t)DispatchRemoteNetworkType::NOC0, // tx_network_type
                0x100000, // debug_buf_addr
                0x40000 // debug_buf_size
            };

        std::vector<uint32_t> traffic_gen_rx_compile_args =
            {
                0x3, // input_queue_id
                0x4, // output_queue_id
                (0x90000 >> 4), // queue_start_addr_words
                (0x20000 >> 4), // queue_size_words
                (uint32_t)phys_traffic_gen_tx_core.x, // remote_rx_x
                (uint32_t)phys_traffic_gen_tx_core.y, // remote_rx_y
                0x3, // remote_tx_queue_id
                (uint32_t)DispatchRemoteNetworkType::NOC0, // tx_network_type
                0x100000, // debug_buf_addr
                0x40000 // debug_buf_size
            };

        std::vector<uint32_t> mux_compile_args =
            {0x3};

        std::vector<uint32_t> demux_compile_args =
            {0x4};

        auto tg_tx = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/mux_demux/kernels/traffic_gen_tx.cpp",
            {traffic_gen_tx_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = traffic_gen_tx_compile_args,
                .defines = {}
            }
        );

        auto tg_rx = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/mux_demux/kernels/traffic_gen_rx.cpp",
            {traffic_gen_rx_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = traffic_gen_rx_compile_args,
                .defines = {}
            }
        );

        vector<uint32_t> args;
        // args.push_back(prefetcher_iterations_g);
        tt_metal::SetRuntimeArgs(program, tg_tx, traffic_gen_tx_core, args);
        tt_metal::SetRuntimeArgs(program, tg_rx, traffic_gen_rx_core, args);

        auto mux = tt_metal::CreateKernel(
            program,
            "tt_metal/impl/dispatch/kernels/packet_mux.cpp",
            {mux_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = mux_compile_args,
                .defines = {}
            }
        );

        auto demux = tt_metal::CreateKernel(
            program,
            "tt_metal/impl/dispatch/kernels/packet_demux.cpp",
            {demux_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = demux_compile_args,
                .defines = {}
            }
        );

        auto start = std::chrono::system_clock::now();
        EnqueueProgram(cq, program, false);
        Finish(cq);
        auto end = std::chrono::system_clock::now();

        std::chrono::duration<double> elapsed_seconds = (end-start);
        log_info(LogTest, "Ran in {}us", elapsed_seconds.count() * 1000 * 1000);

        pass &= tt_metal::CloseDevice(device);
    } catch (const std::exception& e) {
        pass = false;
        log_fatal(e.what());
    }

    tt::llrt::OptionsG.set_kernels_nullified(false);

    if (pass) {
        log_info(LogTest, "Test Passed");
        return 0;
    } else {
        log_fatal(LogTest, "Test Failed\n");
        return 1;
    }
}
