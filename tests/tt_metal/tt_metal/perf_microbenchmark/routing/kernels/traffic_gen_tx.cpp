// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_metal/impl/dispatch/kernels/packet_queue.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/traffic_gen.hpp"

packet_input_queue_state_t input_queues[MAX_SWITCH_FAN_IN];
packet_output_queue_state_t output_queues[MAX_SWITCH_FAN_OUT];

tt_l1_ptr uint32_t* debug_buf;
uint32_t debug_buf_index;
uint32_t debug_buf_size;

constexpr uint32_t src_endpoint_id = get_compile_time_arg_val(0);
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

constexpr uint64_t timeout_cycles = ((uint64_t)(get_compile_time_arg_val(17))) * 1000 * 1000;

constexpr uint32_t debug_output_verbose = get_compile_time_arg_val(18);

constexpr uint32_t input_queue_id = 0;
constexpr uint32_t output_queue_id = 1;

constexpr packet_input_queue_state_t* input_queue_ptr = &(input_queues[input_queue_id]);
constexpr packet_output_queue_state_t* output_queue_ptr = &(output_queues[output_queue_id]);

input_queue_rnd_state_t input_queue_rnd_state;


// generates packets with ranom size and payload on the input side
inline void input_queue_handler() {

    uint32_t free_words = input_queue_ptr->get_queue_data_num_words_free();
    if (free_words == 0) {
        return;
    }

    // Each call to input_queue_handler initializes only up to the end
    // of the queue buffer, so we don't need to handle wrapping.
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

    debug_advance_index(16);
    debug_log_index(0, PACKET_QUEUE_TEST_STARTED);
    debug_log_index(1, 0xff000000);
    debug_log_index(2, 0xcc000000 | src_endpoint_id);

    input_queue_ptr->init(input_queue_id, queue_start_addr_words, queue_size_words,
                          // remote_x, remote_y, remote_queue_id, remote_update_network_type:
                          0, 0, 0, DispatchRemoteNetworkType::NONE);

    output_queue_ptr->init(output_queue_id, remote_rx_queue_start_addr_words, remote_rx_queue_size_words,
                           remote_rx_x, remote_rx_y, remote_rx_queue_id, tx_network_type,
                           input_queues);

    output_queue_ptr->wait_for_dest_ready(timeout_cycles);

    debug_log_index(1, 0xff000001);

    uint64_t data_words_sent = 0;
    uint64_t iter = 0;
    uint64_t words_flushed = 0;
    bool timeout = false;
    uint64_t start_timestamp = c_tensix_core::read_wall_clock();
    uint64_t cycles_elapsed = 0;


    // debug_log(0xcccccccc);
    // input_queue_ptr->debug_log_object();
    // debug_log(0xdddddddd);
    // output_queue_ptr->debug_log_object();

    while ((data_words_sent < total_data_words) ||
           (input_queue_ptr->get_queue_data_num_words_available_to_send() > 0)) {

        debug_log_index(3, iter>>32);
        debug_log_index(4, iter);
        cycles_elapsed = c_tensix_core::read_wall_clock() - start_timestamp;
        if (cycles_elapsed > timeout_cycles) {
            timeout = true;
            break;
        }

        // debug_log_index(1, 0xff000021);

        if (!input_queue_rnd_state.last_packet_done()) {
            input_queue_handler();
        }

        // debug_log_index(1, 0xff000031);

        words_flushed += output_queue_ptr->prev_words_in_flight_check_flush();

        // debug_log_index(1, 0xff000041);

        data_words_sent += output_queue_ptr->forward_data_from_input(input_queue_id);

        // debug_log_index(1, 0xff000051);

        iter++;
    }

    if (!timeout) {
        debug_log_index(1, 0xff00002);
        if (!output_queue_ptr->output_barrier(timeout_cycles)) {
            timeout = true;
        }
    }

    if (!timeout) {
        debug_log_index(1, 0xff00003);
        while (!output_queue_ptr->is_remote_finished()) {
            cycles_elapsed = c_tensix_core::read_wall_clock() - start_timestamp;
            if (cycles_elapsed > timeout_cycles) {
                timeout = true;
                break;
            }
        }
    }

    cycles_elapsed = c_tensix_core::read_wall_clock() - start_timestamp;
    debug_log_index(5, data_words_sent>>32);
    debug_log_index(6, data_words_sent);
    debug_log_index(7, cycles_elapsed>>32);
    debug_log_index(8, cycles_elapsed);

    if (!timeout) {
        debug_log_index(0, PACKET_QUEUE_TEST_PASS);
        debug_log_index(1, 0xff00004);
        if (debug_output_verbose) {
            debug_log(0xcccccccc);
            input_queue_ptr->debug_log_object();
            debug_log(0xdddddddd);
            output_queue_ptr->debug_log_object();
        }
    } else {
        debug_log_index(0, PACKET_QUEUE_TEST_TIMEOUT);
        debug_log_index(1, 0xff00005);
        debug_log(output_queue_ptr->is_remote_finished());
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
