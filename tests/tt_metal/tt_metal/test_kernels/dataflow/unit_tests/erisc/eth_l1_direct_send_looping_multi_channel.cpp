// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <array>

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
    std::uint32_t num_bytes_ = get_arg_val<uint32_t>(2);
    std::uint32_t num_loops_ = get_arg_val<uint32_t>(3);
    std::uint32_t num_sends_per_loop_ = get_arg_val<uint32_t>(4);

    constexpr uint32_t num_bytes_per_send = get_compile_time_arg_val(0);
    constexpr uint32_t num_bytes_per_send_word_size = get_compile_time_arg_val(1);

    constexpr std::uint32_t num_bytes = get_compile_time_arg_val(2);
    constexpr std::uint32_t num_loops = get_compile_time_arg_val(3);
    constexpr std::uint32_t num_sends_per_loop = get_compile_time_arg_val(4);

    constexpr uint32_t MAX_NUM_CHANNELS=8;
    constexpr uint32_t CHANNEL_MASK = (MAX_NUM_CHANNELS - 1);
    std::array<std::uint32_t, MAX_NUM_CHANNELS> channels_active = {0,0,0,0,0,0,0,0};

    eth_setup_handshake(remote_eth_l1_dst_addr, true);

    kernel_profiler::mark_time(100 + num_sends_per_loop);
    kernel_profiler::mark_time(10);
    uint32_t j = 0;
    for (uint32_t i = 0; i < num_loops; i++) {
        // for (uint32_t j = 0; j < num_sends_per_loop; j++) {
        kernel_profiler::mark_time(20);
        if (channels_active[j] != 0) {
            kernel_profiler::mark_time(21);
            eth_wait_for_receiver_channel_done(j);
            channels_active[j] = 0;
        }
        kernel_profiler::mark_time(22);
        eth_send_bytes_over_channel(
            local_eth_l1_src_addr, // + (j * num_bytes),
            remote_eth_l1_dst_addr, // + (j * num_bytes),
            num_bytes,
            j,
            num_bytes_per_send,
            num_bytes_per_send_word_size);
        channels_active[j] = 1;
        kernel_profiler::mark_time(23);
        kernel_profiler::mark_time(77);
        kernel_profiler::mark_time(88);
        // }
        j = (j + 1) & CHANNEL_MASK;
    }

    for (uint32_t j = 0; j < MAX_NUM_CHANNELS; j++) {
        kernel_profiler::mark_time(24);
        if (channels_active[j] != 0) {
            eth_wait_for_receiver_channel_done(j);
        }
        kernel_profiler::mark_time(25);
    }
    kernel_profiler::mark_time(11);

    // This helps flush out the "end" timestamp
    // eth_setup_handshake(remote_eth_l1_dst_addr, true);
    // kernel_profiler::mark_time(100);
    // kernel_profiler::mark_time(100);
    // for (int i = 0; i < 30000; i++);
}
