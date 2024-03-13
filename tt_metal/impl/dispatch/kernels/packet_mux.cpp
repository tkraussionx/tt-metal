// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_metal/impl/dispatch/kernels/packet_queue.hpp"

packet_input_queue_state_t input_queues[MAX_SWITCH_FAN_IN];
packet_output_queue_state_t output_queue;

tt_l1_ptr uint32_t* debug_buf;
uint32_t debug_buf_index;
uint32_t debug_buf_size;

constexpr uint32_t reserved = get_compile_time_arg_val(0);

// assume up to MAX_SWITCH_FAN_IN queues with contiguous storage,
// starting at rx_queue_start_addr
constexpr uint32_t rx_queue_start_addr_words = get_compile_time_arg_val(1);
constexpr uint32_t rx_queue_size_words = get_compile_time_arg_val(2);
constexpr uint32_t rx_queue_size_bytes = rx_queue_size_words*PACKET_WORD_SIZE_BYTES;

constexpr uint32_t mux_fan_in = get_compile_time_arg_val(3);

// FIXME imatosevic - is there a way to do this without explicit indexes?
static_assert(mux_fan_in <= MAX_SWITCH_FAN_IN,
    "mux fan-in higher than MAX_SWITCH_FAN_IN");
static_assert(MAX_SWITCH_FAN_IN == 4,
    "MAX_SWITCH_FAN_IN must be 4 for the initialization below to work");

constexpr uint32_t remote_rx_x[MAX_SWITCH_FAN_IN] =
    {
        (get_compile_time_arg_val(4) & 0xFF),
        (get_compile_time_arg_val(5) & 0xFF),
        (get_compile_time_arg_val(6) & 0xFF),
        (get_compile_time_arg_val(7) & 0xFF)
    };

constexpr uint32_t remote_rx_y[MAX_SWITCH_FAN_IN] =
    {
        (get_compile_time_arg_val(4) >> 8) & 0xFF,
        (get_compile_time_arg_val(5) >> 8) & 0xFF,
        (get_compile_time_arg_val(6) >> 8) & 0xFF,
        (get_compile_time_arg_val(7) >> 8) & 0xFF
    };

constexpr uint32_t remote_rx_queue_id[MAX_SWITCH_FAN_IN] =
    {
        (get_compile_time_arg_val(4) >> 16) & 0xFF,
        (get_compile_time_arg_val(5) >> 16) & 0xFF,
        (get_compile_time_arg_val(6) >> 16) & 0xFF,
        (get_compile_time_arg_val(7) >> 16) & 0xFF
    };

constexpr DispatchRemoteNetworkType remote_rx_network_type[MAX_SWITCH_FAN_IN] =
    {
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(4) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(5) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(6) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(7) >> 24) & 0xFF)
    };

constexpr uint32_t remote_tx_queue_start_addr_words = get_compile_time_arg_val(8);
constexpr uint32_t remote_tx_queue_size_words = get_compile_time_arg_val(9);
constexpr uint32_t remote_tx_x = get_compile_time_arg_val(10);
constexpr uint32_t remote_tx_y = get_compile_time_arg_val(11);
constexpr uint32_t remote_tx_queue_id = get_compile_time_arg_val(12);
constexpr DispatchRemoteNetworkType
    tx_network_type =
        static_cast<DispatchRemoteNetworkType>(get_compile_time_arg_val(13));

constexpr uint32_t debug_buf_addr_arg = get_compile_time_arg_val(14);
constexpr uint32_t debug_buf_size_arg = get_compile_time_arg_val(15);
constexpr uint32_t debug_output_verbose = get_compile_time_arg_val(16);

constexpr uint64_t timeout_cycles = ((uint64_t)(get_compile_time_arg_val(17))) * 1000 * 1000;


void kernel_main() {

    noc_init(0xF);
    debug_set_buf(reinterpret_cast<tt_l1_ptr uint32_t*>(debug_buf_addr_arg), debug_buf_size_arg);
    debug_advance_index(16);
    debug_log_index(0, PACKET_QUEUE_TEST_STARTED);
    debug_log_index(1, 0xff000000);
    debug_log_index(2, 0xaa000000 | mux_fan_in);

    for (uint32_t i = 0; i < mux_fan_in; i++) {
        input_queues[i].init(i, rx_queue_start_addr_words + i*rx_queue_size_words, rx_queue_size_words,
                            remote_rx_x[i], remote_rx_y[i], remote_rx_queue_id[i], remote_rx_network_type[i]);
    }
    output_queue.init(0, remote_tx_queue_start_addr_words, remote_tx_queue_size_words,
                      remote_tx_x, remote_tx_y, remote_tx_queue_id, tx_network_type,
                      input_queues);

    wait_all_src_dest_ready(input_queues, mux_fan_in, &output_queue, 1, timeout_cycles);

    debug_log_index(1, 0xff000001);

    uint64_t progress_timestamp = 0;
    bool timeout = false;
    bool dest_finished = false;
    if (timeout_cycles > 0) {
        progress_timestamp = c_tensix_core::read_wall_clock();
    }
    while (!dest_finished) {
        for (uint32_t i = 0; i < mux_fan_in; i++) {
            if (timeout_cycles > 0) {
                uint64_t cycles_elapsed = c_tensix_core::read_wall_clock() - progress_timestamp;
                if (cycles_elapsed > timeout_cycles) {
                    timeout = true;
                    break;
                }
            }
            if (input_queues[i].get_queue_data_num_words_available_to_send() > 0) {
                if (!output_queue.forward_packet_from_input(i, timeout_cycles)) {
                    timeout = true;
                    break;
                } else if (timeout_cycles > 0) {
                    progress_timestamp = c_tensix_core::read_wall_clock();
                }
            }
            output_queue.prev_words_in_flight_check_flush();
        }
        dest_finished = output_queue.is_remote_finished();
    }

    debug_log_index(1, 0xff000002);
    if (!timeout) {
        for (uint32_t i = 0; i < mux_fan_in; i++) {
            input_queues[i].send_remote_finished_notification();
        }
    }

    if (timeout) {
        debug_log_index(0, PACKET_QUEUE_TEST_TIMEOUT);
        debug_log_index(1, 0xff000003);
    } else {
        debug_log_index(0, PACKET_QUEUE_TEST_PASS);
        debug_log_index(1, 0xff000004);
    }
    if (debug_output_verbose) {
        for (uint32_t i = 0; i < mux_fan_in; i++) {
            debug_log(0xaaaaaa00 + i);
            input_queues[i].debug_log_object();
            debug_log(0xbbbbbbbb);
            output_queue.debug_log_object();
        }
    }
}
