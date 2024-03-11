// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_metal/impl/dispatch/kernels/packet_queue.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/mux_demux/kernels/traffic_gen.hpp"

packet_input_queue_state_t input_queues[MAX_SWITCH_FAN_IN];

tt_l1_ptr uint32_t* debug_buf;
uint32_t debug_buf_index;
uint32_t debug_buf_size;

constexpr uint32_t endpoint_id = get_compile_time_arg_val(0);

constexpr uint32_t num_src_endpoints = get_compile_time_arg_val(1);
constexpr uint32_t num_dest_endpoints = get_compile_time_arg_val(2);

constexpr uint32_t input_queue_id = 0;

constexpr uint32_t queue_start_addr_words = get_compile_time_arg_val(3);
constexpr uint32_t queue_size_words = get_compile_time_arg_val(4);

constexpr uint32_t remote_tx_x = get_compile_time_arg_val(5);
constexpr uint32_t remote_tx_y = get_compile_time_arg_val(6);
constexpr uint32_t remote_tx_queue_id = get_compile_time_arg_val(7);

constexpr DispatchRemoteNetworkType rx_rptr_update_network_type = static_cast<DispatchRemoteNetworkType>(get_compile_time_arg_val(8));

constexpr uint32_t debug_buf_addr_arg = get_compile_time_arg_val(9);
constexpr uint32_t debug_buf_size_arg = get_compile_time_arg_val(10);

constexpr uint32_t prng_seed = get_compile_time_arg_val(11);

constexpr uint32_t total_data_kb = get_compile_time_arg_val(12);
constexpr uint64_t total_data_words = ((uint64_t)total_data_kb) * 1024 / PACKET_WORD_SIZE_BYTES;

constexpr uint32_t max_packet_size_words = get_compile_time_arg_val(13);

constexpr uint32_t disable_data_check = get_compile_time_arg_val(14);

constexpr uint32_t src_endpoint_start_id = get_compile_time_arg_val(15);
constexpr uint32_t dest_endpoint_start_id = get_compile_time_arg_val(16);

constexpr uint64_t max_time = ((uint64_t)0x1024) * 1024 * 1024;

input_queue_rnd_state_t src_rnd_state[num_src_endpoints];

void debug_log_packet(dispatch_packet_header_t* header) {
    uint32_t* header_ptr = reinterpret_cast<uint32_t*>(header);
    for (uint32_t i = 0; i <= header->packet_size_words; i++) {
        for (uint32_t j = 0; j < 4; j++) {
            debug_log(header_ptr[i*4 + j]);
        }
    }
}

void kernel_main() {

    noc_init(0xF);

    for (uint32_t i = 0; i < num_src_endpoints; i++) {
        src_rnd_state[i].init(prng_seed, src_endpoint_start_id+i);
    }

    debug_set_buf(reinterpret_cast<tt_l1_ptr uint32_t*>(debug_buf_addr_arg), debug_buf_size_arg);
    zero_queue_data(queue_start_addr_words, queue_size_words);

    debug_log(0xee000000);

    packet_input_queue_state_t* input_queue = &(input_queues[input_queue_id]);

    input_queue->init(input_queue_id, queue_start_addr_words, queue_size_words,
                      remote_tx_x, remote_tx_y, remote_tx_queue_id,
                      rx_rptr_update_network_type);

    input_queue->wait_for_src_ready();

    debug_log(0xee000001);

    uint64_t num_words_checked = 0;
    uint32_t curr_packet_payload_words_remaining_to_check = 0;

    uint32_t iter = 0;
    bool timeout = false;
    bool check_failed = false;
    uint32_t mismatch_addr, mismatch_val, expected_val;
    tt_l1_ptr dispatch_packet_header_t* curr_packet_header_ptr;
    input_queue_rnd_state_t* src_endpoint_rnd_state;
    uint32_t words_sent = 0;
    uint32_t words_cleared = 0;
    uint64_t start_time = c_tensix_core::read_wall_clock();
    while (num_words_checked < total_data_words) {

        bool packet_available = false;
        do {
            uint64_t curr_time = c_tensix_core::read_wall_clock() - start_time;
            debug_log_index(0, 0xb0000000 + iter);
            if (curr_time > max_time) {
                timeout = true;
                break;
            }
            packet_available = input_queue->input_queue_full_packet_available_to_send();
            if (!packet_available) {
                uint32_t num_words_available = input_queue->input_queue_curr_packet_num_words_available_to_send();
                // mark works as "sent" immediately to keep pipeline from stalling
                input_queue->input_queue_advance_words_sent(num_words_available);
                words_sent += num_words_available;
            }
        } while (!packet_available);

        if (timeout) {
            debug_log_index(1, 0xbb000000 + iter);
            break;
        }

        debug_log_index(2, 0xc0000000 + iter);

        curr_packet_header_ptr = input_queue->get_curr_packet_header_ptr();
        uint32_t src_endpoint_id = input_queue->get_curr_packet_src();
        uint32_t src_endpoint_index = src_endpoint_id - src_endpoint_start_id;
        src_endpoint_rnd_state = &(src_rnd_state[src_endpoint_index]);
        uint32_t curr_packet_size_words = input_queue->get_curr_packet_size_words();
        uint32_t curr_packet_dest = input_queue->get_curr_packet_dest();
        uint32_t curr_packet_tag = input_queue->get_curr_packet_tag();

        src_endpoint_rnd_state->next_packet_rnd_to_dest(num_dest_endpoints, endpoint_id, dest_endpoint_start_id,
                                                        max_packet_size_words, total_data_words);

        uint32_t num_words_available = input_queue->input_queue_curr_packet_num_words_available_to_send();
        // we have the packet header info for checking, input queue can now switch to the next packet
        input_queue->input_queue_advance_words_sent(num_words_available);
        words_sent += num_words_available;

        // move rptr_cleared to the packet payload
        input_queue->input_queue_advance_words_cleared(1);
        words_cleared++;

        debug_log_index(3, 0xd0000000 + iter);

        if (src_endpoint_rnd_state->curr_packet_size_words != curr_packet_size_words ||
            endpoint_id != curr_packet_dest ||
            src_endpoint_rnd_state->packet_rnd_seed != curr_packet_tag) {
                check_failed = true;
                debug_log_index(4, 0xdd000000 + iter);
                debug_log_index(5, src_endpoint_rnd_state->curr_packet_size_words);
                debug_log_index(6, curr_packet_size_words);
                debug_log_index(7, src_endpoint_rnd_state->packet_rnd_seed);
                debug_log_index(8, curr_packet_tag);
                debug_log_index(9, endpoint_id);
                debug_log_index(10, curr_packet_dest);
                break;
        }
        debug_log_index(5, 0xe0000000 + iter);
        if (!disable_data_check) {
            uint32_t words_before_wrap = std::min(curr_packet_size_words, input_queue->get_queue_words_before_rptr_cleared_wrap());
            uint32_t words_after_wrap = 0;
            if (words_before_wrap < curr_packet_size_words) {
                words_after_wrap = curr_packet_size_words - words_before_wrap;
            }
            debug_log_index(5, 0xe1000000 + words_before_wrap);
            if (!check_packet_data(reinterpret_cast<tt_l1_ptr uint32_t*>(input_queue->get_queue_rptr_cleared_addr_bytes()),
                                   words_before_wrap,
                                   curr_packet_tag & 0xFFFF0000,
                                   mismatch_addr, mismatch_val, expected_val)) {
                check_failed = true;
                debug_log_index(6, curr_packet_size_words);
                debug_log_index(7, curr_packet_tag);
                debug_log_index(8, curr_packet_dest);
                debug_log_index(10, input_queue->get_queue_rptr_cleared_addr_bytes());
                break;
            }
            input_queue->input_queue_advance_words_cleared(words_before_wrap);
            words_cleared += words_before_wrap;
            if (words_after_wrap > 0) {
                debug_log_index(5, 0xe2000000 + words_after_wrap);
                if (!check_packet_data(reinterpret_cast<tt_l1_ptr uint32_t*>(input_queue->get_queue_rptr_cleared_addr_bytes()),
                                       words_after_wrap,
                                       (curr_packet_tag & 0xFFFF0000) + words_before_wrap,
                                       mismatch_addr, mismatch_val, expected_val)) {
                    check_failed = true;
                    debug_log_index(6, curr_packet_size_words);
                    debug_log_index(7, curr_packet_tag);
                    debug_log_index(8, curr_packet_dest);
                    debug_log_index(10, input_queue->get_queue_rptr_cleared_addr_bytes());
                    break;
                }
                input_queue->input_queue_advance_words_cleared(words_after_wrap);
                words_cleared += words_after_wrap;
            }
            if (check_failed) {
                break;
            }
        } else {
            input_queue->input_queue_advance_words_cleared(curr_packet_size_words);
            words_cleared += curr_packet_size_words;
        }
        num_words_checked += (curr_packet_size_words+1);
        debug_log_index(7, 0xf0000000 + iter);
        iter++;
    }

    debug_advance_index(16);

    debug_log(0xccccdddd);
    debug_log(timeout);
    debug_log(check_failed);

    if (timeout) {
        debug_log(0xee000003);
        debug_log(num_words_checked>>32);
        debug_log(num_words_checked);
        debug_log(total_data_words>>32);
        debug_log(total_data_words);
        debug_log(words_sent);
        debug_log(words_cleared);
        debug_log(0xdddddddd);
        input_queue->debug_log_object();
    } else if (check_failed) {
        debug_log(0xee000004);
        debug_log(num_words_checked>>32);
        debug_log(num_words_checked);
        debug_log(total_data_words>>32);
        debug_log(total_data_words);
        debug_log(words_sent);
        debug_log(words_cleared);
        debug_log(mismatch_addr);
        debug_log(mismatch_val);
        debug_log(expected_val);
        debug_log(0xdddddddd);
        src_endpoint_rnd_state->debug_log_object();
        debug_log(0xeeeeeeee);
        debug_log_packet(curr_packet_header_ptr);
        debug_log(0xffffffff);
        input_queue->debug_log_object();
    } else {
        debug_log(0xee000002);
        debug_log(num_words_checked>>32);
        debug_log(num_words_checked);
        debug_log(words_sent);
        debug_log(words_cleared);
        debug_log(0xffffffff);
        input_queue->debug_log_object();    }
}
