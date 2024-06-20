// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/llrt/rtoptions.hpp"
#include "tt_metal/impl/dispatch/cq_commands.hpp"
#include "tt_metal/hostdevcommon/common_runtime_address_map.h"
#include "tt_metal/impl/dispatch/kernels/packet_queue_ctrl.hpp"
#include "kernels/traffic_gen_test.hpp"
#include "tt_metal/tt_metal/unit_tests_fast_dispatch/streams/relay_stream_builders.hpp"

using namespace tt;

// TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_LOGGER_LEVEL=Debug TT_METAL_LOGGER_TYPES=Test ./build/test/tt_metal/perf_microbenchmark/routing/test_uni_tunnel_through_stream_single_chip


std::vector<uint32_t> get_relay_rt_args(
    Device* device,
    uint32_t relay_stream_id,
    uint32_t relay_stream_overlay_blob_addr,
    uint32_t relay_done_semaphore,
    CoreCoord const& sender_core,
    CoreCoord const& receiver_core,
    uint32_t sender_noc_id,
    uint32_t receiver_noc_id,
    // stream_config_t const& sender_stream_config,
    stream_config_t const& relay_stream_config,
    stream_config_t const& receiver_stream_config,
    uint32_t remote_src_start_phase_addr,
    uint32_t dest_remote_src_start_phase_addr,
    bool is_first_relay_in_chain) {
    return std::vector<uint32_t>{
        static_cast<uint32_t>(relay_stream_overlay_blob_addr),
        static_cast<uint32_t>(relay_stream_id),
        static_cast<uint32_t>(relay_stream_config.buffer_addr),
        static_cast<uint32_t>(relay_stream_config.buffer_size),
        static_cast<uint32_t>(relay_stream_config.tile_header_buffer_addr),
        static_cast<uint32_t>(relay_stream_config.tile_header_num_msgs),

        // noc0 address
        static_cast<uint32_t>(device->worker_core_from_logical_core(sender_core).x),
        static_cast<uint32_t>(device->worker_core_from_logical_core(sender_core).y),
        static_cast<uint32_t>(relay_stream_id),
        static_cast<uint32_t>(sender_noc_id),

        static_cast<uint32_t>(device->worker_core_from_logical_core(receiver_core).x),
        static_cast<uint32_t>(device->worker_core_from_logical_core(receiver_core).y),
        static_cast<uint32_t>(relay_stream_id),
        static_cast<uint32_t>(receiver_noc_id),
        static_cast<uint32_t>(receiver_stream_config.buffer_addr),
        static_cast<uint32_t>(receiver_stream_config.buffer_size),
        static_cast<uint32_t>(receiver_stream_config.tile_header_buffer_addr),

        static_cast<uint32_t>(relay_done_semaphore),
        static_cast<uint32_t>(is_first_relay_in_chain ? 1 : 0),

        remote_src_start_phase_addr,
        dest_remote_src_start_phase_addr};
}


int main(int argc, char **argv) {

    constexpr uint32_t default_tx_x = 0;
    constexpr uint32_t default_tx_y = 0;
    constexpr uint32_t default_rx_x = 0;
    constexpr uint32_t default_rx_y = 3;

    constexpr uint32_t default_mux_x = 0;
    constexpr uint32_t default_mux_y = 1;
    constexpr uint32_t default_demux_x = 3;
    constexpr uint32_t default_demux_y = 1;

    constexpr uint32_t default_tunneler_x = 0;
    constexpr uint32_t default_tunneler_y = 0;

    constexpr uint32_t default_prng_seed = 0x100;
    constexpr uint32_t default_data_kb_per_tx = 16*1024;
    constexpr uint32_t default_max_packet_size_words = 0x100;

    constexpr uint32_t default_tx_queue_start_addr = 0x80000;
    constexpr uint32_t default_tx_queue_size_bytes = 0x10000;
    constexpr uint32_t default_rx_queue_start_addr = 0xa0000;
    constexpr uint32_t default_rx_queue_size_bytes = 0x20000;
    constexpr uint32_t default_mux_queue_start_addr = 0x80000;
    constexpr uint32_t default_mux_queue_size_bytes = 0x10000;
    constexpr uint32_t default_demux_queue_start_addr = 0x90000;
    constexpr uint32_t default_demux_queue_size_bytes = 0x20000;

    constexpr uint32_t default_tunneler_queue_start_addr = default_tx_queue_start_addr;//0x19000;
    constexpr uint32_t default_tunneler_queue_size_bytes = 0x10000;

    constexpr uint32_t default_test_results_addr = 0x100000;
    constexpr uint32_t default_test_results_size = 0x40000;

    constexpr uint32_t default_tunneler_test_results_addr = 0x29000;
    constexpr uint32_t default_tunneler_test_results_size = 0x8000;

    constexpr uint32_t default_timeout_mcycles = 1000;
    constexpr uint32_t default_rx_disable_data_check = 0;

    constexpr uint32_t src_endpoint_start_id = 0xaa;
    constexpr uint32_t dest_endpoint_start_id = 0xbb;

    constexpr uint32_t num_src_endpoints = 4;
    constexpr uint32_t num_dest_endpoints = 4;

    constexpr uint32_t default_test_device_id = 0;

    constexpr uint32_t default_tile_header_buffer_num_messages = 128;
    constexpr uint32_t default_tile_header_buffer_size = default_tile_header_buffer_num_messages * 16;
    constexpr uint32_t semaphore_size_bytes = 16; // Set to 16 to keep all buffers 16B aligned
    constexpr uint32_t default_overlay_blob_size = 1024;

    uint32_t default_mux_sender_stream_tile_header_buffer_addr = default_mux_queue_start_addr + (default_mux_queue_size_bytes * (num_src_endpoints + 1));
    uint32_t default_tunneler_l_relay_stream_buffer_addr = default_tunneler_queue_start_addr;
    uint32_t default_tunneler_l_relay_stream_tile_header_buffer_addr = default_tunneler_l_relay_stream_buffer_addr + default_tunneler_queue_size_bytes;
    uint32_t default_tunneler_l_relay_stream_overlay_blob_addr = default_tunneler_l_relay_stream_tile_header_buffer_addr + default_tile_header_buffer_size;
    uint32_t default_tunneler_r_relay_stream_buffer_addr = default_tunneler_queue_start_addr;
    uint32_t default_tunneler_r_relay_stream_tile_header_buffer_addr = default_tunneler_l_relay_stream_buffer_addr + default_tunneler_queue_size_bytes;
    uint32_t default_tunneler_r_relay_stream_overlay_blob_addr = default_tunneler_r_relay_stream_tile_header_buffer_addr + default_tile_header_buffer_size;
    uint32_t default_tunneler_l_terminate_semaphore_addr = default_tunneler_l_relay_stream_overlay_blob_addr + default_overlay_blob_size;
    uint32_t default_tunneler_r_terminate_semaphore_addr = default_tunneler_r_relay_stream_overlay_blob_addr + default_overlay_blob_size;
    uint32_t default_tunneler_l_first_phase_of_remote_sender_addr = default_tunneler_l_terminate_semaphore_addr + semaphore_size_bytes;
    uint32_t default_remote_receiver_flushed_mux_semaphore_addr = default_mux_sender_stream_tile_header_buffer_addr + default_tile_header_buffer_size;
    uint32_t default_demux_receiver_stream_buffer_addr = default_demux_queue_start_addr;
    uint32_t default_demux_receiver_stream_buffer_size_bytes = default_demux_queue_size_bytes;
    uint32_t default_demux_receiver_stream_tile_header_buffer_addr = default_demux_receiver_stream_buffer_addr + (default_demux_receiver_stream_buffer_size_bytes * (num_dest_endpoints + 1));

    std::vector<std::string> input_args(argv, argv + argc);
    if (test_args::has_command_option(input_args, "-h") ||
        test_args::has_command_option(input_args, "--help")) {
        log_info(LogTest, "Usage:");
        log_info(LogTest, "  --prng_seed: PRNG seed, default = 0x{:x}", default_prng_seed);
        log_info(LogTest, "  --data_kb_per_tx: Total data in KB per TX endpoint, default = {}", default_data_kb_per_tx);
        log_info(LogTest, "  --max_packet_size_words: Max packet size in words, default = 0x{:x}", default_max_packet_size_words);
        log_info(LogTest, "  --tx_x: X coordinate of the starting TX core, default = {}", default_tx_x);
        log_info(LogTest, "  --tx_y: Y coordinate of the starting TX core, default = {}", default_tx_y);
        log_info(LogTest, "  --rx_x: X coordinate of the starting RX core, default = {}", default_rx_x);
        log_info(LogTest, "  --rx_y: Y coordinate of the starting RX core, default = {}", default_rx_y);
        log_info(LogTest, "  --mux_x: X coordinate of the starting mux core, default = {}", default_mux_x);
        log_info(LogTest, "  --mux_y: Y coordinate of the starting mux core, default = {}", default_mux_y);
        log_info(LogTest, "  --demux_x: X coordinate of the starting demux core, default = {}", default_demux_x);
        log_info(LogTest, "  --demux_y: Y coordinate of the starting demux core, default = {}", default_demux_y);
        log_info(LogTest, "  --tx_queue_start_addr: TX queue start address, default = 0x{:x}", default_tx_queue_start_addr);
        log_info(LogTest, "  --tx_queue_size_bytes: TX queue size in bytes, default = 0x{:x}", default_tx_queue_size_bytes);
        log_info(LogTest, "  --rx_queue_start_addr: RX queue start address, default = 0x{:x}", default_rx_queue_start_addr);
        log_info(LogTest, "  --rx_queue_size_bytes: RX queue size in bytes, default = 0x{:x}", default_rx_queue_size_bytes);
        log_info(LogTest, "  --mux_queue_start_addr: MUX queue start address, default = 0x{:x}", default_mux_queue_start_addr);
        log_info(LogTest, "  --mux_queue_size_bytes: MUX queue size in bytes, default = 0x{:x}", default_mux_queue_size_bytes);
        log_info(LogTest, "  --demux_queue_start_addr: DEMUX queue start address, default = 0x{:x}", default_demux_queue_start_addr);
        log_info(LogTest, "  --demux_queue_size_bytes: DEMUX queue size in bytes, default = 0x{:x}", default_demux_queue_size_bytes);
        log_info(LogTest, "  --test_results_addr: test results buf address, default = 0x{:x}", default_test_results_addr);
        log_info(LogTest, "  --test_results_size: test results buf size, default = 0x{:x}", default_test_results_size);
        log_info(LogTest, "  --timeout_mcycles: Timeout in MCycles, default = {}", default_timeout_mcycles);
        log_info(LogTest, "  --rx_disable_data_check: Disable data check on RX, default = {}", default_rx_disable_data_check);
        log_info(LogTest, "  --device_id: Device on which the test will be run, default = {}", default_test_device_id);
        log_info(LogTest, "  --n_msg_per_phase: num messages to send before having to drain relay buffers. Larger = better performance, more L1 use, Lower = worse performance, less L1 user, default = {}. Minimum allowed = 128", default_tile_header_buffer_num_messages);

        return 0;
    }

    uint32_t tx_x = test_args::get_command_option_uint32(input_args, "--tx_x", default_tx_x);
    uint32_t tx_y = test_args::get_command_option_uint32(input_args, "--tx_y", default_tx_y);
    uint32_t rx_x = test_args::get_command_option_uint32(input_args, "--rx_x", default_rx_x);
    uint32_t rx_y = test_args::get_command_option_uint32(input_args, "--rx_y", default_rx_y);
    uint32_t mux_x = test_args::get_command_option_uint32(input_args, "--mux_x", default_mux_x);
    uint32_t mux_y = test_args::get_command_option_uint32(input_args, "--mux_y", default_mux_y);
    uint32_t demux_x = test_args::get_command_option_uint32(input_args, "--demux_x", default_demux_x);
    uint32_t demux_y = test_args::get_command_option_uint32(input_args, "--demux_y", default_demux_y);
    uint32_t tunneler_x = test_args::get_command_option_uint32(input_args, "--tunneler_x", default_tunneler_x);
    uint32_t tunneler_y = test_args::get_command_option_uint32(input_args, "--tunneler_y", default_tunneler_y);
    uint32_t prng_seed = test_args::get_command_option_uint32(input_args, "--prng_seed", default_prng_seed);
    uint32_t data_kb_per_tx = test_args::get_command_option_uint32(input_args, "--data_kb_per_tx", default_data_kb_per_tx);
    uint32_t max_packet_size_words = test_args::get_command_option_uint32(input_args, "--max_packet_size_words", default_max_packet_size_words);
    uint32_t tx_queue_start_addr = test_args::get_command_option_uint32(input_args, "--tx_queue_start_addr", default_tx_queue_start_addr);
    uint32_t tx_queue_size_bytes = test_args::get_command_option_uint32(input_args, "--tx_queue_size_bytes", default_tx_queue_size_bytes);
    uint32_t rx_queue_start_addr = test_args::get_command_option_uint32(input_args, "--rx_queue_start_addr", default_rx_queue_start_addr);
    uint32_t rx_queue_size_bytes = test_args::get_command_option_uint32(input_args, "--rx_queue_size_bytes", default_rx_queue_size_bytes);
    uint32_t mux_queue_start_addr = test_args::get_command_option_uint32(input_args, "--mux_queue_start_addr", default_mux_queue_start_addr);
    uint32_t mux_queue_size_bytes = test_args::get_command_option_uint32(input_args, "--mux_queue_size_bytes", default_mux_queue_size_bytes);
    uint32_t demux_queue_start_addr = test_args::get_command_option_uint32(input_args, "--demux_queue_start_addr", default_demux_queue_start_addr);
    uint32_t demux_queue_size_bytes = test_args::get_command_option_uint32(input_args, "--demux_queue_size_bytes", default_demux_queue_size_bytes);
    uint32_t tunneler_queue_start_addr = test_args::get_command_option_uint32(input_args, "--tunneler_queue_start_addr", default_tunneler_queue_start_addr);
    uint32_t tunneler_queue_size_bytes = test_args::get_command_option_uint32(input_args, "--tunneler_queue_size_bytes", default_tunneler_queue_size_bytes);
    uint32_t test_results_addr = test_args::get_command_option_uint32(input_args, "--test_results_addr", default_test_results_addr);
    uint32_t test_results_size = test_args::get_command_option_uint32(input_args, "--test_results_size", default_test_results_size);
    uint32_t tunneler_test_results_addr = test_args::get_command_option_uint32(input_args, "--tunneler_test_results_addr", default_tunneler_test_results_addr);
    uint32_t tunneler_test_results_size = test_args::get_command_option_uint32(input_args, "--tunneler_test_results_size", default_tunneler_test_results_size);
    uint32_t timeout_mcycles = test_args::get_command_option_uint32(input_args, "--timeout_mcycles", default_timeout_mcycles);
    uint32_t rx_disable_data_check = test_args::get_command_option_uint32(input_args, "--rx_disable_data_check", default_rx_disable_data_check);
    uint32_t test_device_id = test_args::get_command_option_uint32(input_args, "--device_id", default_test_device_id);
    uint32_t tile_header_buffer_num_messages = test_args::get_command_option_uint32(input_args, "--n_msg_per_phase", default_tile_header_buffer_num_messages);

    uint32_t mux_sender_stream_tile_header_buffer_addr = test_args::get_command_option_uint32(input_args, "--mux_msg_info_buf_addr", default_mux_sender_stream_tile_header_buffer_addr);
    uint32_t tunneler_l_relay_stream_buffer_addr = tunneler_queue_start_addr;
    uint32_t tunneler_l_relay_stream_tile_header_buffer_addr = test_args::get_command_option_uint32(input_args, "--l_tunnel_msg_info_buf_addr", default_tunneler_l_relay_stream_tile_header_buffer_addr);;
    uint32_t tunneler_r_relay_stream_buffer_addr = tunneler_queue_start_addr;
    uint32_t tunneler_r_relay_stream_tile_header_buffer_addr = test_args::get_command_option_uint32(input_args, "--r_tunnel_msg_info_buf_addr", default_tunneler_r_relay_stream_tile_header_buffer_addr);;
    uint32_t tunneler_l_first_phase_of_remote_sender_addr = test_args::get_command_option_uint32(input_args, "--l_tunnel_first_phase_addr", default_tunneler_l_first_phase_of_remote_sender_addr);
    uint32_t remote_receiver_flushed_mux_semaphore_addr = test_args::get_command_option_uint32(input_args, "--mux_rr_flushed_addr", default_remote_receiver_flushed_mux_semaphore_addr);
    uint32_t tunneler_l_terminate_semaphore_addr = test_args::get_command_option_uint32(input_args, "--l_tunnel_terminate_addr", default_tunneler_l_terminate_semaphore_addr);
    uint32_t tunneler_r_terminate_semaphore_addr = test_args::get_command_option_uint32(input_args, "--r_tunnel_terminate_addr", default_tunneler_r_terminate_semaphore_addr);
    uint32_t demux_receiver_stream_buffer_addr = test_args::get_command_option_uint32(input_args, "--demux_buf_addr", default_demux_receiver_stream_buffer_addr);;
    uint32_t demux_receiver_stream_buffer_size_bytes = test_args::get_command_option_uint32(input_args, "--demux_buf_size", default_demux_receiver_stream_buffer_size_bytes);
    uint32_t demux_receiver_stream_tile_header_buffer_addr = test_args::get_command_option_uint32(input_args, "--demux_msg_info_buf_addr", default_demux_receiver_stream_tile_header_buffer_addr);
    uint32_t tunneler_l_relay_stream_overlay_blob_addr = test_args::get_command_option_uint32(input_args, "--tunneler_l_blob_addr", default_tunneler_l_relay_stream_overlay_blob_addr);
    uint32_t tunneler_r_relay_stream_overlay_blob_addr = test_args::get_command_option_uint32(input_args, "--tunneler_r_blob_addr", default_tunneler_r_relay_stream_overlay_blob_addr);

    uint32_t tunneler_to_tunneler_data_noc_id = 0;
    uint32_t relay_stream_buffer_size_bytes = tunneler_queue_size_bytes;
    uint32_t mux_sender_stream_id = 32;
    uint32_t demux_receiver_stream_id = 32;
    uint32_t tunneler_l_stream_id = 32;
    uint32_t tunneler_r_stream_id = 32;
    uint32_t tunneler_l_in_data_noc_id = 0;
    uint32_t tunneler_r_out_data_noc_id = 0;
    TT_ASSERT(tile_header_buffer_num_messages >= 128, "Minimum messages per phase that is currently supported is 128");

    uint32_t stream_tile_header_buffer_size_bytes = tile_header_buffer_num_messages * streams::tile_header_size;

    bool pass = true;
    try {

        int num_devices = tt_metal::GetNumAvailableDevices();
        if (test_device_id >= num_devices) {
            log_info(LogTest,
                "Device {} is not valid. Highest valid device id = {}.",
                test_device_id, num_devices-1);
            throw std::runtime_error("Invalid Device Id.");
        }
        int device_id = test_device_id;

        tt_metal::Device *device = tt_metal::CreateDevice(device_id);
        auto const& device_active_eth_cores = device->get_active_ethernet_cores();

        if (device_active_eth_cores.size() < 2) {
            log_info(LogTest,
                "Device {} does not have enough active cores. Need 2 active ethernet cores for this test.",
                device_id);
            tt_metal::CloseDevice(device);
            throw std::runtime_error("Test cannot run on specified device.");
        }

        auto eth_core_iter = device_active_eth_cores.begin();
        //CoreCoord tunneler_logical_core = device->get_ethernet_sockets(5)[0];
        // CoreCoord tunneler_logical_core = *eth_core_iter;
        // CoreCoord tunneler_phys_core = device->ethernet_core_from_logical_core(tunneler_logical_core);
        CoreCoord tunneler_logical_core = CoreCoord(1,1);
        CoreCoord tunneler_phys_core = device->worker_core_from_logical_core(tunneler_logical_core);

        //CoreCoord r_tunneler_logical_core = device->get_ethernet_sockets(5)[1];
        eth_core_iter++;
        // CoreCoord r_tunneler_logical_core = *eth_core_iter;
        // CoreCoord r_tunneler_phys_core = device->ethernet_core_from_logical_core(r_tunneler_logical_core);
        CoreCoord r_tunneler_logical_core = CoreCoord(2,1); // Putting on same row as tunneler l core to simplify stream dumping
        CoreCoord r_tunneler_phys_core = device->worker_core_from_logical_core(r_tunneler_logical_core);



        std::cout<<"Left Tunneler = "<<tunneler_logical_core.str()<<std::endl;
        std::cout<<"Right Tunneler = "<<r_tunneler_logical_core.str()<<std::endl;

        tt_metal::Program program = tt_metal::CreateProgram();

        CoreCoord mux_core = {mux_x, mux_y};
        CoreCoord mux_phys_core = device->worker_core_from_logical_core(mux_core);

        CoreCoord demux_core = {demux_x, demux_y};
        CoreCoord demux_phys_core = device->worker_core_from_logical_core(demux_core);

        std::vector<CoreCoord> tx_phys_core;
        for (uint32_t i = 0; i < num_src_endpoints; i++) {
            CoreCoord core = {tx_x+i, tx_y};
            tx_phys_core.push_back(device->worker_core_from_logical_core(core));
            std::vector<uint32_t> compile_args =
                {
                    src_endpoint_start_id + i, // 0: src_endpoint_id
                    num_dest_endpoints, // 1: num_dest_endpoints
                    (tx_queue_start_addr >> 4), // 2: queue_start_addr_words
                    (tx_queue_size_bytes >> 4), // 3: queue_size_words
                    ((mux_queue_start_addr + i*mux_queue_size_bytes) >> 4), // 4: remote_rx_queue_start_addr_words
                    (mux_queue_size_bytes >> 4), // 5: remote_rx_queue_size_words
                    (uint32_t)mux_phys_core.x, // 6: remote_rx_x
                    (uint32_t)mux_phys_core.y, // 7: remote_rx_y
                    i, // 8: remote_rx_queue_id
                    (uint32_t)DispatchRemoteNetworkType::NOC0, // 9: tx_network_type
                    test_results_addr, // 10: test_results_addr
                    test_results_size, // 11: test_results_size
                    prng_seed, // 12: prng_seed
                    data_kb_per_tx, // 13: total_data_kb
                    max_packet_size_words, // 14: max_packet_size_words
                    src_endpoint_start_id, // 15: src_endpoint_start_id
                    dest_endpoint_start_id, // 16: dest_endpoint_start_id
                    timeout_mcycles * 1000 * 1000, // 17: timeout_cycles
                };

            log_info(LogTest, "run traffic_gen_tx at x={},y={}", core.x, core.y);
            log_debug(LogTest, "Args:");
            std::size_t ct_arg_idx = 0;
            log_debug(LogTest, "\tsrc_endpoint_id: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\tnum_dest_endpoints: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\tqueue_start_addr_words: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\tqueue_size_words: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\tremote_rx_queue_start_addr_words: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\tremote_rx_queue_size_words: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\tremote_rx_x: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\tremote_rx_y: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\tremote_rx_queue_id: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\ttx_network_type: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\ttest_results_addr: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\ttest_results_size: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\tprng_seed: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\ttotal_data_kb: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\tmax_packet_size_words: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\tsrc_endpoint_start_id: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\tdest_endpoint_start_id: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\ttimeout_cycles: {}", compile_args.at(ct_arg_idx++));

            auto kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/traffic_gen_tx.cpp",
                {core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default,
                    .compile_args = compile_args,
                    .defines = {}
                }
            );
        }

        // Mux
        static constexpr uint32_t input_packetize_log_page_size = 8;
        static constexpr uint32_t packet_size_bytes = 4096;
        static_assert((1 << input_packetize_log_page_size) * 16 == packet_size_bytes);
        std::vector<uint32_t> mux_compile_args =
            {
                0, // 0: reserved
                (mux_queue_start_addr >> 4), // 1: rx_queue_start_addr_words
                (mux_queue_size_bytes >> 4), // 2: rx_queue_size_words
                num_src_endpoints, // 3: mux_fan_in
                packet_switch_4B_pack((uint32_t)tx_phys_core[0].x,
                                      (uint32_t)tx_phys_core[0].y,
                                      1,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 4: src 0 info
                packet_switch_4B_pack((uint32_t)tx_phys_core[1].x,
                                      (uint32_t)tx_phys_core[1].y,
                                      1,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 5: src 1 info
                packet_switch_4B_pack((uint32_t)tx_phys_core[2].x,
                                      (uint32_t)tx_phys_core[2].y,
                                      1,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 6: src 2 info
                packet_switch_4B_pack((uint32_t)tx_phys_core[3].x,
                                      (uint32_t)tx_phys_core[3].y,
                                      1,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 7: src 3 info
                (tunneler_queue_start_addr >> 4), // 8: remote_tx_queue_start_addr_words
                (tunneler_queue_size_bytes >> 4), // 9: remote_tx_queue_size_words
                (uint32_t)tunneler_phys_core.x, // 10: remote_tx_x
                (uint32_t)tunneler_phys_core.y, // 11: remote_tx_y
                0, // 12: remote_tx_queue_id
                (uint32_t)DispatchRemoteNetworkType::STREAM, // 13: tx_network_type
                test_results_addr, // 14: test_results_addr
                test_results_size, // 15: test_results_size
                timeout_mcycles * 1000 * 1000, // 16: timeout_cycles
                0,// 17: output_depacketize
                0, // 18: output_depacketize info
                // input 0 packetization info
                packet_switch_4B_pack(1, // packetize
                                    input_packetize_log_page_size, // log_page_size
                                    0, // downstream sem - set to 0 because stream manages credits + handshaking
                                    0), // local sem - set to 0 because stream manages credits + handshaking
                0, 0, 0, // 20 - 22 input1,2,3 packetization info (All 0s for this test)
                0, 0, // 23, 24 input_packetize_src_endpoint, input_packetize_dest_endpoint
                1 // 25: use_stream_for_writer
            };

        log_info(LogTest, "run mux at x={},y={}", mux_core.x, mux_core.y);
        auto mux_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/impl/dispatch/kernels/packet_mux.cpp",
            {mux_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = mux_compile_args,
                .defines = {}
            }
        );

        uint32_t remote_receiver_flushed_mux_semaphore_addr /*sender_receiver_semaphore_sender*/ = CreateSemaphore(program, mux_core, 0, CoreType::WORKER);
        // uint32_t remote_sender_hang_toggle_addr = CreateSemaphore(program, sender_core, 0, CoreType::WORKER);
        uint32_t tunneler_l_terminate_semaphore_addr /*first_relay_done_semaphore*/ = CreateSemaphore(program, tunneler_logical_core, 0, CoreType::WORKER);
        uint32_t tunneler_r_terminate_semaphore_addr/*second_relay_done_semaphore*/ = CreateSemaphore(program, r_tunneler_logical_core, 0, CoreType::WORKER);
        uint32_t tunneler_l_first_phase_of_remote_sender_addr /*first_relay_remote_src_start_phase_addr*/ = CreateSemaphore(program, tunneler_logical_core, 0, CoreType::WORKER);
        uint32_t second_relay_remote_src_start_phase_addr =
            CreateSemaphore(program, r_tunneler_logical_core, 0, CoreType::WORKER);
        uint32_t receiver_remote_src_start_phase_addr = CreateSemaphore(program, demux_core, 0, CoreType::WORKER);

        std::vector<uint32_t> packet_mux_rt_args =
        {
            mux_sender_stream_id, // local_stream_id;
            mux_sender_stream_tile_header_buffer_addr, // local_stream_tile_header_buffer_addr;
            tile_header_buffer_num_messages, // messages_per_phase;
            tunneler_phys_core.x, // remote_dest_noc_x;
            tunneler_phys_core.y, // remote_dest_noc_y;
            tunneler_l_stream_id, // remote_dest_noc_stream_id;
            tunneler_l_in_data_noc_id, // remote_dest_noc_id;
            tunneler_l_relay_stream_buffer_addr, // remote_buffer_base_addr;
            relay_stream_buffer_size_bytes, // remote_buffer_size_4B_words;
            tunneler_l_relay_stream_tile_header_buffer_addr, // remote_tile_header_buffer_addr;
            tunneler_l_terminate_semaphore_addr, // relay_done_semaphore_addr;
            r_tunneler_phys_core.x, // other_relay_core_to_signal_x;
            r_tunneler_phys_core.y, // other_relay_core_to_signal_y;
            tunneler_r_terminate_semaphore_addr, // second_relay_done_semaphore
            remote_receiver_flushed_mux_semaphore_addr, // wait_receiver_semaphore;
            second_relay_remote_src_start_phase_addr //tunneler_l_first_phase_of_remote_sender_addr, // first_relay_remote_src_start_phase_addr;
        };
        tt_metal::SetRuntimeArgs(program, mux_kernel, mux_core, packet_mux_rt_args);
        {
            log_debug(tt::LogTest, "MUX kernel RT args:");
            std::size_t i = 0;
            log_debug(tt::LogTest, "\tlocal_stream_id: {}", packet_mux_rt_args[i++]);
            log_debug(tt::LogTest, "\tlocal_stream_tile_header_buffer_addr: {}", packet_mux_rt_args[i++]);
            log_debug(tt::LogTest, "\tmessages_per_phase: {}", packet_mux_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_dest_noc_x: {}", packet_mux_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_dest_noc_y: {}", packet_mux_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_dest_noc_stream_id: {}", packet_mux_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_dest_noc_id: {}", packet_mux_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_buffer_base_addr: {}", packet_mux_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_buffer_size_4B_words: {}", packet_mux_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_tile_header_buffer_addr: {}", packet_mux_rt_args[i++]);
            log_debug(tt::LogTest, "\trelay_done_semaphore_addr: {}", packet_mux_rt_args[i++]);
            log_debug(tt::LogTest, "\tother_relay_core_to_signal_x: {}", packet_mux_rt_args[i++]);
            log_debug(tt::LogTest, "\tother_relay_core_to_signal_y: {}", packet_mux_rt_args[i++]);
            log_debug(tt::LogTest, "\tother_relay_done_semaphore: {}", packet_mux_rt_args[i++]);
            log_debug(tt::LogTest, "\twait_receiver_semaphore: {}", packet_mux_rt_args[i++]);
            log_debug(tt::LogTest, "\tfirst_relay_remote_src_start_phase_addr: {}", packet_mux_rt_args[i++]);
            TT_ASSERT(i == packet_mux_rt_args.size(), "Mismatch in number of runtime args");
        }

        std::vector<uint32_t> packet_demux_rt_args = {
            demux_receiver_stream_id, // stream_id
            demux_receiver_stream_buffer_addr, // stream_buffer_addr
            demux_receiver_stream_buffer_size_bytes, // stream_buffer_size
            demux_receiver_stream_tile_header_buffer_addr, // stream_tile_header_buffer_addr
            tile_header_buffer_num_messages, // num_message_per_phase
            r_tunneler_phys_core.x, // remote_src_noc_x
            r_tunneler_phys_core.y, // remote_src_noc_y
            tunneler_r_stream_id, // remote_src_noc_stream_id
            tunneler_r_out_data_noc_id, // remote_src_data_noc_id
            tunneler_r_terminate_semaphore_addr, // relay_done_semaphore_addr
            tunneler_phys_core.x, // other_relay_core_to_signal_x
            tunneler_phys_core.y, // other_relay_core_to_signal_y
            tunneler_l_terminate_semaphore_addr, // other_relay_done_semaphore
            mux_core.x, // sender_noc_x
            mux_core.y, // sender_noc_y
            remote_receiver_flushed_mux_semaphore_addr, // sender_wait_finish_semaphore
            receiver_remote_src_start_phase_addr, //second_relay_remote_src_start_phase_addr //tunneler_r_relay_remote_src_start_phase_addr  // remote_src_start_phase_addr
        };

        {
            log_debug(tt::LogTest, "DEMUX kernel RT args:");
            std::size_t i = 0;
            log_debug(tt::LogTest, "\tlocal_stream_id: {}", packet_demux_rt_args[i++]);
            log_debug(tt::LogTest, "\tstream_buffer_addr: {}", packet_demux_rt_args[i++]);
            log_debug(tt::LogTest, "\tstream_buffer_size: {}", packet_demux_rt_args[i++]);
            log_debug(tt::LogTest, "\tstream_tile_header_buffer_addr: {}", packet_demux_rt_args[i++]);
            log_debug(tt::LogTest, "\tstream_tile_header_max_num_messages: {}", packet_demux_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_src_noc_x: {}", packet_demux_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_src_noc_y: {}", packet_demux_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_src_noc_stream_id: {}", packet_demux_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_src_data_noc_id: {}", packet_demux_rt_args[i++]);
            log_debug(tt::LogTest, "\trelay_done_semaphore_addr: {}", packet_demux_rt_args[i++]);
            log_debug(tt::LogTest, "\tother_relay_core_to_signal_x: {}", packet_demux_rt_args[i++]);
            log_debug(tt::LogTest, "\tother_relay_core_to_signal_y: {}", packet_demux_rt_args[i++]);
            log_debug(tt::LogTest, "\tother_relay_done_semaphore: {}", packet_demux_rt_args[i++]);
            log_debug(tt::LogTest, "\tsender_noc_x: {}", packet_demux_rt_args[i++]);
            log_debug(tt::LogTest, "\tsender_noc_y: {}", packet_demux_rt_args[i++]);
            log_debug(tt::LogTest, "\tsender_wait_finish_semaphore: {}", packet_demux_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_src_start_phase_addr: {}", packet_demux_rt_args[i++]);
        }


        // stream_config_t mux_remote_sender_stream_config {
        //     0,
        //     0,
        //     mux_sender_stream_tile_header_buffer_addr,
        //     tile_header_buffer_num_messages,
        //     stream_tile_header_buffer_size_bytes};
        stream_config_t tunneler_l_relay_stream_config {
            tunneler_l_relay_stream_buffer_addr,
            relay_stream_buffer_size_bytes,
            tunneler_l_relay_stream_tile_header_buffer_addr,
            tile_header_buffer_num_messages,
            stream_tile_header_buffer_size_bytes};
        stream_config_t tunneler_r_relay_stream_config  {
            tunneler_r_relay_stream_buffer_addr,
            relay_stream_buffer_size_bytes,
            tunneler_r_relay_stream_tile_header_buffer_addr,
            tile_header_buffer_num_messages,
            stream_tile_header_buffer_size_bytes};
        stream_config_t demux_remote_receiver_stream_config  {
            demux_receiver_stream_buffer_addr,
            demux_receiver_stream_buffer_size_bytes,
            demux_receiver_stream_tile_header_buffer_addr,
            tile_header_buffer_num_messages,
            stream_tile_header_buffer_size_bytes};

        std::vector<uint32_t> tunneler_l_compile_args =
            {
                dest_endpoint_start_id, // 0: endpoint_id_start_index
                1, // 1: tunnel_lanes. 1 => Unidirectional. 2 => Bidirectional.
                (tunneler_queue_start_addr >> 4), // 2: rx_queue_start_addr_words
                (tunneler_queue_size_bytes >> 4), // 3: rx_queue_size_words
                packet_switch_4B_pack(r_tunneler_phys_core.x,
                                      r_tunneler_phys_core.y,
                                      0,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 4: remote_receiver_0_info
                0, // 5: remote_receiver_1_info
                (tunneler_queue_start_addr >> 4), // 6: remote_receiver_queue_start_addr_words 0
                (tunneler_queue_size_bytes >> 4), // 7: remote_receiver_queue_size_words 0
                0, // 8: remote_receiver_queue_start_addr_words 1
                2, // 9: remote_receiver_queue_size_words 1.
                   // Unused. Setting to 2 to get around size check assertion that does not allow 0.
                packet_switch_4B_pack(mux_phys_core.x,
                                      mux_phys_core.y,
                                      num_dest_endpoints,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 10: remote_sender_0_info
                0, // 11: remote_sender_1_info
                tunneler_test_results_addr, // 12: test_results_addr
                tunneler_test_results_size, // 13: test_results_size
                timeout_mcycles * 1000 * 1000 * 4, // 14: timeout_cycles
            };

        auto tunneler_l_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/streams/stream_relay.cpp",
            tunneler_logical_core,
            tt_metal::DataMovementConfig{.noc = tt_metal::NOC::NOC_0, .compile_args = {}});
        auto tunneler_r_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tt_metal/test_kernels/dataflow/streams/stream_relay.cpp",
            r_tunneler_logical_core,
            tt_metal::DataMovementConfig{.noc = tt_metal::NOC::NOC_0, .compile_args = {}});

        auto tunneler_l_stream_relay_rt_args = get_relay_rt_args(
                device,
                tunneler_l_stream_id,
                tunneler_l_relay_stream_overlay_blob_addr,//first_relay_stream_overlay_blob_addr, // relay_stream_overlay_blob_addr
                tunneler_l_terminate_semaphore_addr, // relay_done_semaphore
                mux_core, // producer core // sender_core
                r_tunneler_logical_core, // dest core // receiver_core
                tunneler_l_in_data_noc_id, // sender_noc_id
                tunneler_to_tunneler_data_noc_id, // receiver_noc_id
                /*sender_stream_config,*/ tunneler_l_relay_stream_config, // relay_stream_config
                tunneler_r_relay_stream_config, // receiver_stream_config
                tunneler_l_first_phase_of_remote_sender_addr, //tunneler_l_relay_remote_src_start_phase_addr, // remote_src_start_phase_addr
                second_relay_remote_src_start_phase_addr, //tunneler_r_relay_remote_src_start_phase_addr, // dest_remote_src_start_phase_addr
                true); // is_first_relay_in_chain
        auto tunneler_r_stream_relay_rt_args = get_relay_rt_args(
                device,
                tunneler_r_stream_id,
                tunneler_r_relay_stream_overlay_blob_addr, // second_relay_stream_overlay_blob_addr,
                tunneler_r_terminate_semaphore_addr, // second_relay_done_semaphore,
                tunneler_logical_core, // producer core
                demux_core,
                tunneler_to_tunneler_data_noc_id,
                tunneler_r_out_data_noc_id,
                /*sender_stream_config,*/ tunneler_r_relay_stream_config,
                demux_remote_receiver_stream_config,
                second_relay_remote_src_start_phase_addr, // tunneler_r_relay_remote_src_start_phase_addr,
                receiver_remote_src_start_phase_addr, // demux_receiver_remote_src_start_phase_addr,
                false);

        {
            log_debug(tt::LogTest, "tunneler_l_stream_relay_rt_args: ");
            std::size_t i = 0;
            log_debug(tt::LogTest, "\trelay_stream_overlay_blob_addr: {}", tunneler_l_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tstream_id: {}", tunneler_l_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tstream_buffer_addr: {}", tunneler_l_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tstream_buffer_size: {}", tunneler_l_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tstream_tile_header_buffer_addr: {}", tunneler_l_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tstream_tile_header_max_num_messages: {}", tunneler_l_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_src_noc_x: {}", tunneler_l_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_src_noc_y: {}", tunneler_l_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_src_stream_id: {}", tunneler_l_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_src_noc_id: {}", tunneler_l_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_dest_noc_x: {}", tunneler_l_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_dest_noc_y: {}", tunneler_l_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_dest_noc_stream_id: {}", tunneler_l_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_dest_noc_id: {}", tunneler_l_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_dest_buf_addr: {}", tunneler_l_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_dest_buf_size_4B_words: {}", tunneler_l_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_dest_tile_header_buffer_addr: {}", tunneler_l_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\ttx_rx_done_semaphore_addr: {}", tunneler_l_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tis_first_relay_stream_in_chain: {}", tunneler_l_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_src_start_phase_addr: {}", tunneler_l_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tdest_remote_src_start_phase_addr: {}", tunneler_l_stream_relay_rt_args[i++]);
            TT_ASSERT(i == tunneler_l_stream_relay_rt_args.size(), "Mismatch in number of runtime args");
        }
        tt_metal::SetRuntimeArgs(program, tunneler_l_kernel, tunneler_logical_core, tunneler_l_stream_relay_rt_args);


        {
            log_debug(tt::LogTest, "tunneler_r_stream_relay_rt_args: ");
            std::size_t i = 0;
            log_debug(tt::LogTest, "\trelay_stream_overlay_blob_addr: {}", tunneler_r_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tstream_id: {}", tunneler_r_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tstream_buffer_addr: {}", tunneler_r_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tstream_buffer_size: {}", tunneler_r_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tstream_tile_header_buffer_addr: {}", tunneler_r_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tstream_tile_header_max_num_messages: {}", tunneler_r_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_src_noc_x: {}", tunneler_r_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_src_noc_y: {}", tunneler_r_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_src_stream_id: {}", tunneler_r_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_src_noc_id: {}", tunneler_r_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_dest_noc_x: {}", tunneler_r_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_dest_noc_y: {}", tunneler_r_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_dest_noc_stream_id: {}", tunneler_r_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_dest_noc_id: {}", tunneler_r_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_dest_buf_addr: {}", tunneler_r_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_dest_buf_size_4B_words: {}", tunneler_r_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_dest_tile_header_buffer_addr: {}", tunneler_r_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\ttx_rx_done_semaphore_addr: {}", tunneler_r_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tis_first_relay_stream_in_chain: {}", tunneler_r_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tremote_src_start_phase_addr: {}", tunneler_r_stream_relay_rt_args[i++]);
            log_debug(tt::LogTest, "\tdest_remote_src_start_phase_addr: {}", tunneler_r_stream_relay_rt_args[i++]);
            TT_ASSERT(i == tunneler_l_stream_relay_rt_args.size(), "Mismatch in number of runtime args");
        }
        tt_metal::SetRuntimeArgs(program, tunneler_r_kernel, r_tunneler_logical_core, tunneler_r_stream_relay_rt_args);

        std::vector<uint32_t> tunneler_r_compile_args =
            {
                dest_endpoint_start_id, // 0: endpoint_id_start_index
                1,  // 1: tunnel_lanes. 1 => Unidirectional. 2 => Bidirectional.
                (tunneler_queue_start_addr >> 4), // 2: rx_queue_start_addr_words
                (tunneler_queue_size_bytes >> 4), // 3: rx_queue_size_words
                packet_switch_4B_pack(demux_phys_core.x,
                                      demux_phys_core.y,
                                      num_dest_endpoints,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 4: remote_receiver_0_info
                0, // 5: remote_receiver_1_info
                (demux_queue_start_addr >> 4), // 6: remote_receiver_queue_start_addr_words 0
                (demux_queue_size_bytes >> 4), // 7: remote_receiver_queue_size_words 0
                0, // 8: remote_receiver_queue_start_addr_words 1
                2, // 9: remote_receiver_queue_size_words 1
                   // Unused. Setting to 2 to get around size check assertion that does not allow 0.
                packet_switch_4B_pack(tunneler_phys_core.x,
                                      tunneler_phys_core.y,
                                      2,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 10: remote_sender_0_info
                0, // 11: remote_sender_1_info
                tunneler_test_results_addr, // 12: test_results_addr
                tunneler_test_results_size, // 13: test_results_size
                timeout_mcycles * 1000 * 1000 * 4, // 14: timeout_cycles
            };


        std::vector<CoreCoord> rx_phys_core;
        for (uint32_t i = 0; i < num_dest_endpoints; i++) {
            CoreCoord core = {rx_x+i, rx_y};
            rx_phys_core.push_back(device->worker_core_from_logical_core(core));
            std::vector<uint32_t> compile_args =
                {
                    dest_endpoint_start_id + i, // 0: dest_endpoint_id
                    num_src_endpoints, // 1: num_src_endpoints
                    num_dest_endpoints, // 2: num_dest_endpoints
                    (rx_queue_start_addr >> 4), // 3: queue_start_addr_words
                    (rx_queue_size_bytes >> 4), // 4: queue_size_words
                    (uint32_t)demux_phys_core.x, // 5: remote_tx_x
                    (uint32_t)demux_phys_core.y, // 6: remote_tx_y
                    i, // 7: remote_tx_queue_id
                    (uint32_t)DispatchRemoteNetworkType::NOC0, // 8: rx_rptr_update_network_type
                    test_results_addr, // 9: test_results_addr
                    test_results_size, // 10: test_results_size
                    prng_seed, // 11: prng_seed
                    0, // 12: reserved
                    max_packet_size_words, // 13: max_packet_size_words
                    rx_disable_data_check, // 14: disable data check
                    src_endpoint_start_id, // 15: src_endpoint_start_id
                    dest_endpoint_start_id, // 16: dest_endpoint_start_id
                    timeout_mcycles * 1000 * 1000, // 17: timeout_cycles
                };

            log_info(LogTest, "run traffic_gen_rx at x={},y={}", core.x, core.y);

            log_debug(LogTest, "Args:");
            std::size_t ct_arg_idx = 0;
            log_debug(LogTest, "\tdest_endpoint_id: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\tnum_src_endpoints: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\tnum_dest_endpoints: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\tqueue_start_addr_words: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\tqueue_size_words: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\tremote_tx_x: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\tremote_tx_y: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\tremote_tx_queue_id: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\trx_rptr_update_network_type: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\ttest_results_addr: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\ttest_results_size: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\tprng_seed: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\treserved: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\tmax_packet_size_words: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\tcheck: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\tsrc_endpoint_start_id: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\tdest_endpoint_start_id: {}", compile_args.at(ct_arg_idx++));
            log_debug(LogTest, "\ttimeout_cycles: {}", compile_args.at(ct_arg_idx++));
            auto kernel = tt_metal::CreateKernel(
                program,
                "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/traffic_gen_rx.cpp",
                {core},
                tt_metal::DataMovementConfig{
                    .processor = tt_metal::DataMovementProcessor::RISCV_1,
                    .noc = tt_metal::NOC::RISCV_1_default,
                    .compile_args = compile_args,
                    .defines = {}
                }
            );
        }

        // Demux
        uint32_t dest_map_array[4] = {0, 1, 2, 3};
        uint64_t dest_endpoint_output_map = packet_switch_dest_pack(dest_map_array, 4);
        std::vector<uint32_t> demux_compile_args =
            {
                dest_endpoint_start_id, // 0: endpoint_id_start_index
                (demux_queue_start_addr >> 4), // 1: rx_queue_start_addr_words
                (demux_queue_size_bytes >> 4), // 2: rx_queue_size_words
                num_dest_endpoints, // 3: demux_fan_out
                packet_switch_4B_pack(rx_phys_core[0].x,
                                      rx_phys_core[0].y,
                                      0,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 4: remote_tx_0_info
                packet_switch_4B_pack(rx_phys_core[1].x,
                                      rx_phys_core[1].y,
                                      0,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 5: remote_tx_1_info
                packet_switch_4B_pack(rx_phys_core[2].x,
                                      rx_phys_core[2].y,
                                      0,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 6: remote_tx_2_info
                packet_switch_4B_pack(rx_phys_core[3].x,
                                      rx_phys_core[3].y,
                                      0,
                                      (uint32_t)DispatchRemoteNetworkType::NOC0), // 7: remote_tx_3_info
                (rx_queue_start_addr >> 4), // 8: remote_tx_queue_start_addr_words 0
                (rx_queue_size_bytes >> 4), // 9: remote_tx_queue_size_words 0
                (rx_queue_start_addr >> 4), // 10: remote_tx_queue_start_addr_words 1
                (rx_queue_size_bytes >> 4), // 11: remote_tx_queue_size_words 1
                (rx_queue_start_addr >> 4), // 12: remote_tx_queue_start_addr_words 2
                (rx_queue_size_bytes >> 4), // 13: remote_tx_queue_size_words 2
                (rx_queue_start_addr >> 4), // 14: remote_tx_queue_start_addr_words 3
                (rx_queue_size_bytes >> 4), // 15: remote_tx_queue_size_words 3
                //(uint32_t)mux_phys_core.x, // 16: remote_rx_x
                //(uint32_t)mux_phys_core.y, // 17: remote_rx_y
                (uint32_t)r_tunneler_phys_core.x, // 16: remote_rx_x
                (uint32_t)r_tunneler_phys_core.y, // 17: remote_rx_y
                2, // 18: remote_rx_queue_id
                (uint32_t)DispatchRemoteNetworkType::STREAM, // 19: remote_rx_network_type
                (uint32_t)(dest_endpoint_output_map >> 32), // 20: dest_endpoint_output_map_hi
                (uint32_t)(dest_endpoint_output_map & 0xFFFFFFFF), // 21: dest_endpoint_output_map_lo
                test_results_addr, // 22: test_results_addr
                test_results_size, // 23: test_results_size
                timeout_mcycles * 1000 * 1000, // 24: timeout_cycles
                0, 0, 0, 0, 0, // 25-29: packetize/depacketize settings
                1 // 30: use_stream_for_writer
            };

        log_info(LogTest, "run demux at x={},y={}", demux_core.x, demux_core.y);
        auto demux_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/impl/dispatch/kernels/packet_demux.cpp",
            {demux_core},
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1,
                .noc = tt_metal::NOC::RISCV_0_default,
                .compile_args = demux_compile_args,
                .defines = {}
            }
        );
        tt_metal::SetRuntimeArgs(program, demux_kernel, demux_core, packet_demux_rt_args);

        log_info(LogTest, "Starting test...");

        auto start = std::chrono::system_clock::now();
        tt_metal::detail::LaunchProgram(device, program);
        auto end = std::chrono::system_clock::now();

        std::chrono::duration<double> elapsed_seconds = (end-start);
        log_info(LogTest, "Ran in {:.2f}us", elapsed_seconds.count() * 1000 * 1000);

        vector<vector<uint32_t>> tx_results;
        vector<vector<uint32_t>> rx_results;

        for (uint32_t i = 0; i < num_src_endpoints; i++) {
            tx_results.push_back(
                tt::llrt::read_hex_vec_from_core(
                    device->id(), tx_phys_core[i], test_results_addr, test_results_size));
            log_info(LogTest, "TX{} status = {}", i, packet_queue_test_status_to_string(tx_results[i][PQ_TEST_STATUS_INDEX]));
            pass &= (tx_results[i][PQ_TEST_STATUS_INDEX] == PACKET_QUEUE_TEST_PASS);
        }

        for (uint32_t i = 0; i < num_dest_endpoints; i++) {
            rx_results.push_back(
                tt::llrt::read_hex_vec_from_core(
                    device->id(), rx_phys_core[i], test_results_addr, test_results_size));
            log_info(LogTest, "RX{} status = {}", i, packet_queue_test_status_to_string(rx_results[i][PQ_TEST_STATUS_INDEX]));
            pass &= (rx_results[i][PQ_TEST_STATUS_INDEX] == PACKET_QUEUE_TEST_PASS);
        }

        vector<uint32_t> mux_results =
            tt::llrt::read_hex_vec_from_core(
                device->id(), mux_phys_core, test_results_addr, test_results_size);
        log_info(LogTest, "MUX status = {}", packet_queue_test_status_to_string(mux_results[PQ_TEST_STATUS_INDEX]));
        pass &= (mux_results[PQ_TEST_STATUS_INDEX] == PACKET_QUEUE_TEST_PASS);

        vector<uint32_t> demux_results =
            tt::llrt::read_hex_vec_from_core(
                device->id(), demux_phys_core, test_results_addr, test_results_size);
        log_info(LogTest, "DEMUX status = {}", packet_queue_test_status_to_string(demux_results[PQ_TEST_STATUS_INDEX]));
        pass &= (demux_results[0] == PACKET_QUEUE_TEST_PASS);

        pass &= tt_metal::CloseDevice(device);

        if (pass) {
            double total_tx_bw = 0.0;
            uint64_t total_tx_words_sent = 0;
            uint64_t total_rx_words_checked = 0;
            for (uint32_t i = 0; i < num_src_endpoints; i++) {
                uint64_t tx_words_sent = get_64b_result(tx_results[i], PQ_TEST_WORD_CNT_INDEX);
                total_tx_words_sent += tx_words_sent;
                uint64_t tx_elapsed_cycles = get_64b_result(tx_results[i], PQ_TEST_CYCLES_INDEX);
                double tx_bw = ((double)tx_words_sent) * PACKET_WORD_SIZE_BYTES / tx_elapsed_cycles;
                log_info(LogTest,
                         "TX {} words sent = {}, elapsed cycles = {} -> BW = {:.2f} B/cycle",
                         i, tx_words_sent, tx_elapsed_cycles, tx_bw);
                total_tx_bw += tx_bw;
            }
            log_info(LogTest, "Total TX BW = {:.2f} B/cycle", total_tx_bw);
            double total_rx_bw = 0.0;
            for (uint32_t i = 0; i < num_dest_endpoints; i++) {
                uint64_t rx_words_checked = get_64b_result(rx_results[i], PQ_TEST_WORD_CNT_INDEX);
                total_rx_words_checked += rx_words_checked;
                uint64_t rx_elapsed_cycles = get_64b_result(rx_results[i], PQ_TEST_CYCLES_INDEX);
                double rx_bw = ((double)rx_words_checked) * PACKET_WORD_SIZE_BYTES / rx_elapsed_cycles;
                log_info(LogTest,
                         "RX {} words checked = {}, elapsed cycles = {} -> BW = {:.2f} B/cycle",
                         i, rx_words_checked, rx_elapsed_cycles, rx_bw);
                total_rx_bw += rx_bw;
            }
            log_info(LogTest, "Total RX BW = {:.2f} B/cycle", total_rx_bw);
            if (total_tx_words_sent != total_rx_words_checked) {
                log_error(LogTest, "Total TX words sent = {} != Total RX words checked = {}", total_tx_words_sent, total_rx_words_checked);
                pass = false;
            } else {
                log_info(LogTest, "Total TX words sent = {} == Total RX words checked = {} -> OK", total_tx_words_sent, total_rx_words_checked);
            }
            uint64_t mux_words_sent = get_64b_result(mux_results, PQ_TEST_WORD_CNT_INDEX);
            uint64_t mux_elapsed_cycles = get_64b_result(mux_results, PQ_TEST_CYCLES_INDEX);
            uint64_t mux_iter = get_64b_result(mux_results, PQ_TEST_ITER_INDEX);
            double mux_bw = ((double)mux_words_sent) * PACKET_WORD_SIZE_BYTES / mux_elapsed_cycles;
            double mux_cycles_per_iter = ((double)mux_elapsed_cycles) / mux_iter;
            log_info(LogTest,
                     "MUX words sent = {}, elapsed cycles = {} -> BW = {:.2f} B/cycle",
                     mux_words_sent, mux_elapsed_cycles, mux_bw);
            log_info(LogTest,
                        "MUX iters = {} -> cycles/iter = {:.1f}",
                        mux_iter, mux_cycles_per_iter);
            if (mux_words_sent != total_rx_words_checked) {
                log_error(LogTest, "MUX words sent = {} != Total RX words checked = {}", mux_words_sent, total_rx_words_checked);
                pass = false;
            } else {
                log_info(LogTest, "MUX words sent = {} == Total RX words checked = {} -> OK", mux_words_sent, total_rx_words_checked);
            }

            uint64_t demux_words_sent = get_64b_result(demux_results, PQ_TEST_WORD_CNT_INDEX);
            uint64_t demux_elapsed_cycles = get_64b_result(demux_results, PQ_TEST_CYCLES_INDEX);
            double demux_bw = ((double)demux_words_sent) * PACKET_WORD_SIZE_BYTES / demux_elapsed_cycles;
            uint64_t demux_iter = get_64b_result(demux_results, PQ_TEST_ITER_INDEX);
            double demux_cycles_per_iter = ((double)demux_elapsed_cycles) / demux_iter;
            log_info(LogTest,
                     "DEMUX words sent = {}, elapsed cycles = {} -> BW = {:.2f} B/cycle",
                     demux_words_sent, demux_elapsed_cycles, demux_bw);
            log_info(LogTest,
                     "DEMUX iters = {} -> cycles/iter = {:.1f}",
                     demux_iter, demux_cycles_per_iter);
            if (demux_words_sent != total_rx_words_checked) {
                log_error(LogTest, "DEMUX words sent = {} != Total RX words checked = {}", demux_words_sent, total_rx_words_checked);
                pass = false;
            } else {
                log_info(LogTest, "DEMUX words sent = {} == Total RX words checked = {} -> OK", demux_words_sent, total_rx_words_checked);
            }
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
