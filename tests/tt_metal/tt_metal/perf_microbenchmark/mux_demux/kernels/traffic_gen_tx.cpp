// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_metal/impl/dispatch/kernels/packet_queue.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/mux_demux/kernels/traffic_gen.hpp"

packet_input_queue_state_t input_queues[MAX_SWITCH_FAN_IN];
packet_output_queue_state_t output_queues[MAX_SWITCH_FAN_OUT];

tt_l1_ptr uint32_t* debug_buf;
uint32_t debug_buf_index;
uint32_t debug_buf_size;

constexpr uint32_t src_endpoint_id = get_compile_time_arg_val(0);

constexpr uint32_t input_queue_id = 0;
constexpr uint32_t output_queue_id = 1;

constexpr uint32_t num_dest_endpoints = get_compile_time_arg_val(1);

constexpr uint32_t queue_start_addr_words = get_compile_time_arg_val(2);
constexpr uint32_t queue_size_words = get_compile_time_arg_val(3);
constexpr uint32_t queue_size_bytes = queue_size_words*PACKET_WORD_SIZE_BYTES;

constexpr uint32_t remote_rx_queue_start_addr_words = get_compile_time_arg_val(4);
constexpr uint32_t remote_rx_queue_size_words = get_compile_time_arg_val(5);

constexpr uint32_t remote_rx_x = get_compile_time_arg_val(6);
constexpr uint32_t remote_rx_y = get_compile_time_arg_val(7);
constexpr uint32_t remote_rx_queue_id = get_compile_time_arg_val(8);

constexpr DispatchRemoteNetworkType
    tx_network_type =
        static_cast<DispatchRemoteNetworkType>(get_compile_time_arg_val(9));

constexpr uint32_t debug_buf_addr_arg = get_compile_time_arg_val(10);
constexpr uint32_t debug_buf_size_arg = get_compile_time_arg_val(11);

constexpr uint32_t prng_seed = get_compile_time_arg_val(12);

constexpr uint32_t total_data_kb = get_compile_time_arg_val(13);
constexpr uint64_t total_data_words = ((uint64_t)total_data_kb) * 1024 / PACKET_WORD_SIZE_BYTES;

constexpr uint32_t max_packet_size_words = get_compile_time_arg_val(14);

constexpr uint32_t src_endpoint_start_id = get_compile_time_arg_val(15);
constexpr uint32_t dest_endpoint_start_id = get_compile_time_arg_val(16);

constexpr packet_input_queue_state_t* input_queue_ptr = &(input_queues[input_queue_id]);
constexpr packet_output_queue_state_t* output_queue_ptr = &(output_queues[output_queue_id]);

constexpr uint64_t max_time = ((uint64_t)0x1024) * 1024 * 1024;

input_queue_rnd_state_t input_queue_rnd_state;

inline void input_queue_handler() {

    uint32_t free_words = input_queue_ptr->get_queue_data_num_words_free();
    if (free_words == 0) {
        return;
    }

    uint32_t byte_wr_addr = input_queue_ptr->get_queue_wptr_addr_bytes();
    uint32_t words_to_init = std::min(free_words,
                                      input_queue_ptr->get_queue_words_before_wptr_wrap());
    uint32_t words_initialized = 0;

    while (words_initialized < words_to_init) {
        if (!input_queue_rnd_state.packet_active()) {
            if (input_queue_rnd_state.last_packet_done()) {
                break;
            }
            input_queue_rnd_state.next_packet_rnd(num_dest_endpoints,
                                                  dest_endpoint_start_id,
                                                  max_packet_size_words,
                                                  total_data_words);

            tt_l1_ptr dispatch_packet_header_t* header_ptr =
                reinterpret_cast<tt_l1_ptr dispatch_packet_header_t*>(byte_wr_addr);
            header_ptr->packet_size_words = input_queue_rnd_state.curr_packet_size_words;
            header_ptr->packet_src = src_endpoint_id;
            header_ptr->packet_dest = input_queue_rnd_state.curr_packet_dest;
            header_ptr->packet_flags = 0;
            header_ptr->num_cmds = 0;
            header_ptr->tag = input_queue_rnd_state.packet_rnd_seed;
            words_initialized++;
            input_queue_rnd_state.curr_packet_words_remaining--;
            byte_wr_addr += PACKET_WORD_SIZE_BYTES;
        }
        else {
            uint32_t words_remaining = words_to_init - words_initialized;
            uint32_t num_words = std::min(words_remaining, input_queue_rnd_state.curr_packet_words_remaining);
            uint32_t start_val =
                (input_queue_rnd_state.packet_rnd_seed & 0xFFFF0000) +
                (input_queue_rnd_state.curr_packet_size_words - input_queue_rnd_state.curr_packet_words_remaining);
            fill_packet_data(reinterpret_cast<tt_l1_ptr uint32_t*>(byte_wr_addr),
                             num_words,
                             start_val);
            words_initialized += num_words;
            input_queue_rnd_state.curr_packet_words_remaining -= num_words;
            byte_wr_addr += num_words*PACKET_WORD_SIZE_BYTES;
        }
    }
    input_queue_ptr->advance_queue_local_wptr(words_initialized);
}


void kernel_main() {

    input_queue_rnd_state.init(prng_seed, src_endpoint_id);
    noc_init(0xF);

    debug_set_buf(reinterpret_cast<tt_l1_ptr uint32_t*>(debug_buf_addr_arg), debug_buf_size_arg);
    zero_queue_data(queue_start_addr_words, queue_size_words);

    // uint32_t s = prng_seed ^ src_endpoint_id;
    // for (uint32_t i = 0; i < 32; i++) {
    //     debug_log(s);
    //     s = prng_next(s);
    // }

    debug_log(0xff000000);

    input_queue_ptr->init(input_queue_id, queue_start_addr_words, queue_size_words,
                          // remote_x, remote_y, remote_queue_id, remote_update_network_type
                          0, 0, 0, DispatchRemoteNetworkType::NONE);

    output_queue_ptr->init(output_queue_id, remote_rx_queue_start_addr_words, remote_rx_queue_size_words,
                           remote_rx_x, remote_rx_y, remote_rx_queue_id, tx_network_type,
                           input_queues);

    output_queue_ptr->wait_for_dest_ready();

    debug_log(0xff000001);

    uint64_t data_words_sent = 0;
    uint64_t start_time = c_tensix_core::read_wall_clock();
    bool timeout = false;
    uint32_t iter = 0;
    uint64_t words_flushed = 0;
    while ((data_words_sent < total_data_words) ||
           (input_queue_ptr->get_queue_data_num_words_available_to_send() > 0)) {
        debug_log_index(0, iter);
        uint64_t curr_time = c_tensix_core::read_wall_clock() - start_time;
        if (curr_time > max_time) {
            timeout = true;
            break;
        }
        if (!input_queue_rnd_state.last_packet_done()) {
            input_queue_handler();
        }
        words_flushed += output_queue_ptr->prev_words_in_flight_check_flush();
        data_words_sent += output_queue_ptr->forward_data_from_input(input_queue_id);
        iter++;
    }

    if (!timeout) {
        debug_log(0xff00002);
        if (!output_queue_ptr->output_barrier(max_time)) {
            timeout = true;
        }
    }

    if (!timeout) {
        debug_log(0xff00003);
        debug_log(data_words_sent>>32);
        debug_log(data_words_sent);
        debug_log(0xcccccccc);
        input_queue_ptr->debug_log_object();
        debug_log(0xdddddddd);
        output_queue_ptr->debug_log_object();
    } else {
        debug_log(0xff00004);
        debug_log(data_words_sent>>32);
        debug_log(data_words_sent);
        debug_log(words_flushed>>32);
        debug_log(words_flushed);
        debug_log(total_data_words>>32);
        debug_log(total_data_words);
        debug_log(0xcccccccc);
        input_queue_ptr->debug_log_object();
        debug_log(0xdddddddd);
        output_queue_ptr->debug_log_object();
    }


}
