// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

/**
 * Any two RISC processors cannot use the same CMD_BUF
 * non_blocking APIs shouldn't be mixed with slow noc.h APIs
 * explicit flushes need to be used since the calls are non-blocking
 * */


void eth_setup_handshake(std::uint32_t handshake_register_address, bool is_sender) {
    if (is_sender) {
        eth_send_bytes(handshake_register_address,handshake_register_address, 16);
        eth_wait_for_receiver_done();

        // eth_wait_for_bytes(16);
        // eth_receiver_done();
    } else {
        eth_wait_for_bytes(16);
        eth_receiver_done();

        // eth_send_bytes(handshake_register_address,handshake_register_address, 16);
        // eth_wait_for_receiver_done();
    }
}

void kernel_main() {
    std::uint32_t local_eth_l1_src_addr = get_arg_val<uint32_t>(0);
    std::uint32_t remote_eth_l1_dst_addr = get_arg_val<uint32_t>(1);
    std::size_t num_bytes_ = get_arg_val<uint32_t>(2);
    std::uint32_t num_loops_ = get_arg_val<uint32_t>(3);
    std::uint32_t num_sends_per_loop_ = get_arg_val<uint32_t>(4);

    constexpr uint32_t num_bytes_per_send = get_compile_time_arg_val(0);
    constexpr uint32_t num_bytes_per_send_word_size = get_compile_time_arg_val(1);

    constexpr std::size_t num_bytes = get_compile_time_arg_val(2);
    constexpr std::uint32_t num_loops = get_compile_time_arg_val(3);
    constexpr std::uint32_t num_sends_per_loop = get_compile_time_arg_val(4);


    constexpr uint32_t MAX_NUM_CHANNELS=8;
    constexpr uint32_t CHANNEL_MASK = (MAX_NUM_CHANNELS - 1);
    // Handshake first before timestamping to make sure we aren't measuring any
    // dispatch/setup times for the kernels on both sides of the link.
    eth_setup_handshake(remote_eth_l1_dst_addr, false);

    kernel_profiler::mark_time(12);

    uint32_t j = 0;
    for (uint32_t i = 0; i < num_loops; i++) {
        // for (uint32_t j = 0; j < num_sends_per_loop; j++) {
        kernel_profiler::mark_time(14);
        eth_wait_for_bytes_on_channel(num_bytes, j);
        kernel_profiler::mark_time(15);
        eth_receiver_channel_done(j);
        kernel_profiler::mark_time(16);
        // }
        j = (j + 1) & CHANNEL_MASK;
    }
    kernel_profiler::mark_time(13);

    kernel_profiler::mark_time(100);
    kernel_profiler::mark_time(100);
    kernel_profiler::mark_time(100);
    kernel_profiler::mark_time(100);


    // This helps flush out the "end" timestamp
    // eth_setup_handshake(remote_eth_l1_dst_addr, false);
    // for (int i = 0; i < 30000; i++);
}
