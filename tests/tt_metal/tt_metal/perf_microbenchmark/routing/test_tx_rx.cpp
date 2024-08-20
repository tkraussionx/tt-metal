// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/llrt/rtoptions.hpp"
#include "tt_metal/impl/dispatch/cq_commands.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"
#include "kernels/traffic_gen_test.hpp"

using namespace tt;


int main(int argc, char **argv) {

    constexpr uint32_t default_tx_x = 0;
    constexpr uint32_t default_tx_y = 0;
    constexpr uint32_t default_rx_x = 1;
    constexpr uint32_t default_rx_y = 0;

    constexpr uint32_t default_prng_seed = 0x100;
    constexpr uint32_t default_total_data_kb = 16*1024;
    constexpr uint32_t default_max_packet_size_words = 0x100;

    constexpr uint32_t default_test_result_buf_addr = L1_UNRESERVED_BASE;
    constexpr uint32_t default_test_result_buf_size = 1024;
    constexpr uint32_t default_tx_queue_start_addr = L1_UNRESERVED_BASE + default_test_result_buf_size;
    constexpr uint32_t default_tx_queue_size_bytes = 0x10000;
    constexpr uint32_t default_rx_queue_start_addr = L1_UNRESERVED_BASE + 0x2000;
    constexpr uint32_t default_rx_queue_size_bytes = 0x20000;

    constexpr uint32_t default_timeout_mcycles = 1000;
    constexpr uint32_t default_rx_disable_data_check = 0;

    std::vector<std::string> input_args(argv, argv + argc);
    if (test_args::has_command_option(input_args, "-h") ||
        test_args::has_command_option(input_args, "--help")) {
        log_info(LogTest, "Usage:");
        log_info(LogTest, "  --prng_seed: PRNG seed, default = 0x{:x}", default_prng_seed);
        log_info(LogTest, "  --total_data_kb: Total data in KB, default = {}", default_total_data_kb);
        log_info(LogTest, "  --max_packet_size_words: Max packet size in words, default = 0x{:x}", default_max_packet_size_words);
        log_info(LogTest, "  --tx_x: X coordinate of the TX core, default = {}", default_tx_x);
        log_info(LogTest, "  --tx_y: Y coordinate of the TX core, default = {}", default_tx_y);
        log_info(LogTest, "  --rx_x: X coordinate of the RX core, default = {}", default_rx_x);
        log_info(LogTest, "  --rx_y: Y coordinate of the RX core, default = {}", default_rx_y);
        log_info(LogTest, "  --tx_queue_start_addr: TX queue start address, default = 0x{:x}", default_tx_queue_start_addr);
        log_info(LogTest, "  --tx_queue_size_bytes: TX queue size in bytes, default = 0x{:x}", default_tx_queue_size_bytes);
        log_info(LogTest, "  --rx_queue_start_addr: RX queue start address, default = 0x{:x}", default_rx_queue_start_addr);
        log_info(LogTest, "  --rx_queue_size_bytes: RX queue size in bytes, default = 0x{:x}", default_rx_queue_size_bytes);
        log_info(LogTest, "  --test_result_buf_addr: Test results buffer address, default = 0x{:x}", default_test_result_buf_addr);
        log_info(LogTest, "  --test_result_buf_size: Test results buffer size, default = {} bytes", default_test_result_buf_size);
        log_info(LogTest, "  --timeout_mcycles: Timeout in MCycles, default = {}", default_timeout_mcycles);
        log_info(LogTest, "  --rx_disable_data_check: Disable data check on RX, default = {}", default_rx_disable_data_check);
        return 0;
    }

    uint32_t tx_x = test_args::get_command_option_uint32(input_args, "--tx_x", default_tx_x);
    uint32_t tx_y = test_args::get_command_option_uint32(input_args, "--tx_y", default_tx_y);
    uint32_t rx_x = test_args::get_command_option_uint32(input_args, "--rx_x", default_rx_x);
    uint32_t rx_y = test_args::get_command_option_uint32(input_args, "--rx_y", default_rx_y);
    uint32_t prng_seed = test_args::get_command_option_uint32(input_args, "--prng_seed", default_prng_seed);
    uint32_t total_data_kb = test_args::get_command_option_uint32(input_args, "--total_data_kb", default_total_data_kb);
    uint32_t max_packet_size_words = test_args::get_command_option_uint32(input_args, "--max_packet_size_words", default_max_packet_size_words);
    uint32_t tx_queue_start_addr = test_args::get_command_option_uint32(input_args, "--tx_queue_start_addr", default_tx_queue_start_addr);
    uint32_t tx_queue_size_bytes = test_args::get_command_option_uint32(input_args, "--tx_queue_size_bytes", default_tx_queue_size_bytes);
    uint32_t rx_queue_start_addr = test_args::get_command_option_uint32(input_args, "--rx_queue_start_addr", default_rx_queue_start_addr);
    uint32_t rx_queue_size_bytes = test_args::get_command_option_uint32(input_args, "--rx_queue_size_bytes", default_rx_queue_size_bytes);
    uint32_t test_result_buf_addr = test_args::get_command_option_uint32(input_args, "--test_result_buf_addr", default_test_result_buf_addr);
    uint32_t test_result_buf_size = test_args::get_command_option_uint32(input_args, "--test_result_buf_size", default_test_result_buf_size);
    uint32_t timeout_mcycles = test_args::get_command_option_uint32(input_args, "--timeout_mcycles", default_timeout_mcycles);
    uint32_t rx_disable_data_check = test_args::get_command_option_uint32(input_args, "--rx_disable_data_check", default_rx_disable_data_check);

    assert(is_power_of_2(tx_queue_size_bytes) && (tx_queue_size_bytes >= 1024));
    assert(is_power_of_2(rx_queue_size_bytes) && (rx_queue_size_bytes >= 1024));

    bool pass = true;
    try {
        int device_id = 0;
        tt_metal::Device *device = tt_metal::CreateDevice(device_id);

        tt_metal::Program *program = tt_metal::CreateProgram();

        CoreCoord traffic_gen_tx_core = {tx_x, tx_y};
        CoreCoord traffic_gen_rx_core = {rx_x, rx_y};

        CoreCoord phys_traffic_gen_tx_core = device->worker_core_from_logical_core(traffic_gen_tx_core);
        CoreCoord phys_traffic_gen_rx_core = device->worker_core_from_logical_core(traffic_gen_rx_core);

        std::vector<uint32_t> traffic_gen_tx_compile_args =
            {
                0xaa, // 0: src_endpoint_id
                1, // 1: num_dest_endpoints
                (tx_queue_start_addr >> 4), // 2: queue_start_addr_words
                (tx_queue_size_bytes >> 4), // 3: queue_size_words
                (rx_queue_start_addr >> 4), // 4: remote_rx_queue_start_addr_words
                (rx_queue_size_bytes >> 4), // 5: remote_rx_queue_size_words
                (uint32_t)phys_traffic_gen_rx_core.x, // 6: remote_rx_x
                (uint32_t)phys_traffic_gen_rx_core.y, // 7: remote_rx_y
                0x0, // 8: remote_rx_queue_id
                (uint32_t)DispatchRemoteNetworkType::NOC0, // 9: tx_network_type
                test_result_buf_addr, // 10: test_result_buf_addr
                test_result_buf_size, // 11: test_result_buf_size
                prng_seed, // 12: prng_seed
                total_data_kb, // 13: total_data_kb
                max_packet_size_words, // 14: max_packet_size_words
                0xaa, // 15: src_endpoint_start_id
                0xbb, // 16: dest_endpoint_start_id
                timeout_mcycles * 1000 * 1000, // 17: timeout_cycles
            };

        std::vector<uint32_t> traffic_gen_rx_compile_args =
            {
                0xbb, // 0: dest_endpoint_id
                1, // 1: num_src_endpoints
                1, // 2: num_dest_endpoints
                (rx_queue_start_addr >> 4), // 3: queue_start_addr_words
                (rx_queue_size_bytes >> 4), // 4: queue_size_words
                (uint32_t)phys_traffic_gen_tx_core.x, // 5: remote_rx_x
                (uint32_t)phys_traffic_gen_tx_core.y, // 6: remote_rx_y
                1, // 7: remote_tx_queue_id
                (uint32_t)DispatchRemoteNetworkType::NOC0, // 8: rx_rptr_update_network_type
                test_result_buf_addr, // 9: test_result_buf_addr
                test_result_buf_size, // 10: test_result_buf_size
                prng_seed, // 11: prng_seed
                total_data_kb, // 12: total_data_kb
                max_packet_size_words, // 13: max_packet_size_words
                rx_disable_data_check, // 14: disable data check
                0xaa, // 15: src_endpoint_start_id
                0xbb, // 16: dest_endpoint_start_id
                timeout_mcycles * 1000 * 1000, // 17: timeout_cycles
            };

        auto tg_tx = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/traffic_gen_tx.cpp",
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
            "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/traffic_gen_rx.cpp",
            {traffic_gen_rx_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = traffic_gen_rx_compile_args,
                .defines = {}
            }
        );

        log_info(LogTest, "Starting test...");

        auto start = std::chrono::system_clock::now();

        tt_metal::detail::LaunchProgram(device, program);

        auto end = std::chrono::system_clock::now();

        std::chrono::duration<double> elapsed_seconds = (end-start);
        log_info(LogTest, "Ran in {:.2f}us", elapsed_seconds.count() * 1000 * 1000);

        vector<uint32_t> tx_results =
            tt::llrt::read_hex_vec_from_core(
                device->id(), phys_traffic_gen_tx_core, test_result_buf_addr, test_result_buf_size);

        vector<uint32_t> rx_results =
            tt::llrt::read_hex_vec_from_core(
                device->id(), phys_traffic_gen_rx_core, test_result_buf_addr, test_result_buf_size);

        log_info(LogTest, "TX status = {}",
                packet_queue_test_status_to_string(tx_results[PQ_TEST_STATUS_INDEX]));
        log_info(LogTest, "RX status = {}",
                packet_queue_test_status_to_string(rx_results[PQ_TEST_STATUS_INDEX]));

        pass &= (tx_results[PQ_TEST_STATUS_INDEX] == PACKET_QUEUE_TEST_PASS);
        pass &= (rx_results[PQ_TEST_STATUS_INDEX] == PACKET_QUEUE_TEST_PASS);

        pass &= tt_metal::CloseDevice(device);

        if (pass) {
            uint64_t total_data_bytes = ((uint64_t)total_data_kb) * 1024;
            uint64_t total_data_words = total_data_bytes / PACKET_WORD_SIZE_BYTES;
            uint64_t tx_words_sent = get_64b_result(tx_results, PQ_TEST_WORD_CNT_INDEX);
            uint64_t rx_words_checked = get_64b_result(rx_results, PQ_TEST_WORD_CNT_INDEX);
            if (tx_words_sent == rx_words_checked) {
                log_info(LogTest, "TX words sent = {}, RX words checked = {} -> OK", tx_words_sent, rx_words_checked);
                if (tx_words_sent < total_data_words) {
                    log_error(LogTest, "TX words sent = {}, total_data_words = {}", tx_words_sent, total_data_words);
                }
            } else {
                log_error(LogTest, "TX words sent = {}, RX words checked = {}", tx_words_sent, rx_words_checked);
                pass = false;
            }
            double elapsed_us = elapsed_seconds.count() * 1000.0 * 1000.0;
            double wall_clock_bw = ((double)total_data_bytes) / elapsed_us;
            log_info(LogTest, "Wall clock time = {:.2f}us, bytes = {} -> wall clock BW = {:.2f} MB/s",
                elapsed_us, total_data_bytes, wall_clock_bw);
            uint64_t tx_elapsed_cycles = get_64b_result(tx_results, PQ_TEST_CYCLES_INDEX);
            uint64_t rx_elapsed_cycles = get_64b_result(rx_results, PQ_TEST_CYCLES_INDEX);
            double tx_bw = ((double)total_data_bytes) / tx_elapsed_cycles;
            double rx_bw = ((double)total_data_bytes) / rx_elapsed_cycles;
            log_info(LogTest, "TX elapsed cycles = {} -> TX BW = {:.2f} B/cycle", tx_elapsed_cycles, tx_bw);
            log_info(LogTest, "RX elapsed cycles = {} -> RX BW = {:.2f} B/cycle", rx_elapsed_cycles, rx_bw);
        }

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
