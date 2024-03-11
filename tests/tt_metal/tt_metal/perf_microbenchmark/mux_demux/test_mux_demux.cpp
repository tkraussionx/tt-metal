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

        constexpr uint32_t prng_seed = 0x100;
        constexpr uint32_t total_data_kb = 256;
        constexpr uint32_t max_packet_size_words = 0x100;

        constexpr uint32_t prefetcher_iterations_g = 1;

        std::vector<uint32_t> traffic_gen_tx_compile_args =
            {
                0xaa, // 0: src_endpoint_id
                0x1, // 1: num_dest_endpoints
                (0x80000 >> 4), // 2: queue_start_addr_words
                (0x10000 >> 4), // 3: queue_size_words
                (0x90000 >> 4), // 4: remote_rx_queue_start_addr_words
                (0x20000 >> 4), // 5: remote_rx_queue_size_words
                (uint32_t)phys_traffic_gen_rx_core.x, // 6: remote_rx_x
                (uint32_t)phys_traffic_gen_rx_core.y, // 7: remote_rx_y
                0x0, // 8: remote_rx_queue_id
                (uint32_t)DispatchRemoteNetworkType::NOC0, // 9: tx_network_type
                0x100000, // 10: debug_buf_addr
                0x40000, // 11: debug_buf_size
                prng_seed, // 12: prng_seed
                total_data_kb, // 13: total_data_kb
                max_packet_size_words, // 14: max_packet_size_words
                0xaa, // 15: src_endpoint_start_id
                0xbb // 16: dest_endpoint_start_id
            };

        std::vector<uint32_t> traffic_gen_rx_compile_args =
            {
                0xbb, // 0: dest_endpoint_id
                1, // 1: num_src_endpoints
                1, // 2: num_dest_endpoints
                (0x90000 >> 4), // 3: queue_start_addr_words
                (0x20000 >> 4), // 4: queue_size_words
                (uint32_t)phys_traffic_gen_tx_core.x, // 5: remote_rx_x
                (uint32_t)phys_traffic_gen_tx_core.y, // 6: remote_rx_y
                1, // 7: remote_tx_queue_id
                (uint32_t)DispatchRemoteNetworkType::NOC0, // 8: rx_rptr_update_network_type
                0x100000, // 9: debug_buf_addr
                0x40000, // 10: debug_buf_size
                prng_seed, // 11: prng_seed
                total_data_kb, // 12: total_data_kb
                max_packet_size_words, // 13: max_packet_size_words
                0, // 14: disable data check
                0xaa, // 15: src_endpoint_start_id
                0xbb // 16: dest_endpoint_start_id
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

        log_info(LogTest, "Starting test...");

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
