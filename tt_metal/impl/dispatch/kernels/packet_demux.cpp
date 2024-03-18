// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_metal/impl/dispatch/kernels/packet_queue.hpp"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/routing/kernels/traffic_gen.hpp"


packet_input_queue_state_t input_queue;
packet_output_queue_state_t output_queues[MAX_SWITCH_FAN_OUT];

tt_l1_ptr uint32_t* debug_buf;
uint32_t debug_buf_index;
uint32_t debug_buf_size;

constexpr uint32_t endpoint_id_start_index = get_compile_time_arg_val(0);

constexpr uint32_t rx_queue_start_addr_words = get_compile_time_arg_val(1);
constexpr uint32_t rx_queue_size_words = get_compile_time_arg_val(2);
constexpr uint32_t rx_queue_size_bytes = rx_queue_size_words*PACKET_WORD_SIZE_BYTES;

static_assert(is_power_of_2(rx_queue_size_words), "rx_queue_size_words must be a power of 2");

constexpr uint32_t demux_fan_out = get_compile_time_arg_val(3);

// FIXME imatosevic - is there a way to do this without explicit indexes?
static_assert(demux_fan_out <= MAX_SWITCH_FAN_OUT,
    "demux fan-out higher than MAX_SWITCH_FAN_OUT");
static_assert(MAX_SWITCH_FAN_OUT == 4,
    "MAX_SWITCH_FAN_OUT must be 4 for the initialization below to work");

constexpr uint32_t remote_tx_x[MAX_SWITCH_FAN_OUT] =
    {
        (get_compile_time_arg_val(4) & 0xFF),
        (get_compile_time_arg_val(5) & 0xFF),
        (get_compile_time_arg_val(6) & 0xFF),
        (get_compile_time_arg_val(7) & 0xFF)
    };

constexpr uint32_t remote_tx_y[MAX_SWITCH_FAN_OUT] =
    {
        (get_compile_time_arg_val(4) >> 8) & 0xFF,
        (get_compile_time_arg_val(5) >> 8) & 0xFF,
        (get_compile_time_arg_val(6) >> 8) & 0xFF,
        (get_compile_time_arg_val(7) >> 8) & 0xFF
    };

constexpr uint32_t remote_tx_queue_id[MAX_SWITCH_FAN_OUT] =
    {
        (get_compile_time_arg_val(4) >> 16) & 0xFF,
        (get_compile_time_arg_val(5) >> 16) & 0xFF,
        (get_compile_time_arg_val(6) >> 16) & 0xFF,
        (get_compile_time_arg_val(7) >> 16) & 0xFF
    };

constexpr DispatchRemoteNetworkType remote_tx_network_type[MAX_SWITCH_FAN_OUT] =
    {
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(4) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(5) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(6) >> 24) & 0xFF),
        static_cast<DispatchRemoteNetworkType>((get_compile_time_arg_val(7) >> 24) & 0xFF)
    };

constexpr uint32_t remote_tx_queue_start_addr_words[MAX_SWITCH_FAN_OUT] =
    {
        get_compile_time_arg_val(8),
        get_compile_time_arg_val(10),
        get_compile_time_arg_val(12),
        get_compile_time_arg_val(14)
    };

constexpr uint32_t remote_tx_queue_size_words[MAX_SWITCH_FAN_OUT] =
    {
        get_compile_time_arg_val(9),
        get_compile_time_arg_val(11),
        get_compile_time_arg_val(13),
        get_compile_time_arg_val(15)
    };

static_assert(is_power_of_2(remote_tx_queue_size_words[0]), "remote_tx_queue_size_words must be a power of 2");
static_assert(is_power_of_2(remote_tx_queue_size_words[1]), "remote_tx_queue_size_words must be a power of 2");
static_assert(is_power_of_2(remote_tx_queue_size_words[2]), "remote_tx_queue_size_words must be a power of 2");
static_assert(is_power_of_2(remote_tx_queue_size_words[3]), "remote_tx_queue_size_words must be a power of 2");

constexpr uint32_t remote_rx_x = get_compile_time_arg_val(16);
constexpr uint32_t remote_rx_y = get_compile_time_arg_val(17);
constexpr uint32_t remote_rx_queue_id = get_compile_time_arg_val(18);
constexpr DispatchRemoteNetworkType
    remote_rx_network_type =
        static_cast<DispatchRemoteNetworkType>(get_compile_time_arg_val(19));

static_assert(MAX_DEST_ENDPOINTS <= 32 && MAX_SWITCH_FAN_OUT <= 4,
    "We assume MAX_DEST_ENDPOINTS <= 32 and MAX_SWITCH_FAN_OUT <= 4 for the initialization below to work");

constexpr uint64_t dest_endpoint_output_map =
    ((uint64_t)(get_compile_time_arg_val(20)) << 32) |
    ((uint64_t)(get_compile_time_arg_val(21)));

constexpr uint32_t output_queue_index_bits = 2;
constexpr uint32_t output_queue_index_mask = (1 << output_queue_index_bits) - 1;

constexpr uint32_t debug_buf_addr_arg = get_compile_time_arg_val(22);
constexpr uint32_t debug_buf_size_arg = get_compile_time_arg_val(23);
constexpr uint32_t debug_output_verbose = get_compile_time_arg_val(24);

constexpr uint64_t timeout_cycles = ((uint64_t)(get_compile_time_arg_val(25))) * 1000 * 1000;


void kernel_main() {

    noc_init(0xF);
    debug_set_buf(reinterpret_cast<tt_l1_ptr uint32_t*>(debug_buf_addr_arg), debug_buf_size_arg);
    debug_advance_index(32);
    debug_log_index(0, PACKET_QUEUE_TEST_STARTED);
    debug_log_index(1, 0xff000000);
    debug_log_index(2, 0xbb000000 | demux_fan_out);
    debug_log_index(3, dest_endpoint_output_map >> 32);
    debug_log_index(4, dest_endpoint_output_map);
    debug_log_index(5, endpoint_id_start_index);

    for (uint32_t i = 0; i < demux_fan_out; i++) {
       output_queues[i].init(i, remote_tx_queue_start_addr_words[i], remote_tx_queue_size_words[i],
                             remote_tx_x[i], remote_tx_y[i], remote_tx_queue_id[i], remote_tx_network_type[i],
                             &input_queue);
    }
    input_queue.init(demux_fan_out, rx_queue_start_addr_words, rx_queue_size_words,
                     remote_rx_x, remote_rx_y, remote_rx_queue_id, remote_rx_network_type);

    wait_all_src_dest_ready(&input_queue, 1, output_queues, demux_fan_out, timeout_cycles);

    debug_log_index(1, 0xff000001);

    bool timeout = false;
    bool all_outputs_finished = false;
    uint64_t data_words_sent = 0;
    uint64_t start_timestamp = c_tensix_core::read_wall_clock();
    uint64_t progress_timestamp = start_timestamp;
    while (!all_outputs_finished && !timeout) {
        if (timeout_cycles > 0) {
            uint64_t cycles_elapsed = c_tensix_core::read_wall_clock() - progress_timestamp;
            if (cycles_elapsed > timeout_cycles) {
                timeout = true;
                break;
            }
        }
        if (input_queue.get_curr_packet_valid()) {
            uint32_t dest = input_queue.get_curr_packet_dest();
            uint32_t output_queue_id = packet_switch_dest_unpack(dest, endpoint_id_start_index,
                                                                 dest_endpoint_output_map);
            bool full_packet_sent;
            uint32_t words_sent = output_queues[output_queue_id].forward_data_from_input(0, full_packet_sent);
            data_words_sent += words_sent;
            if ((words_sent > 0) && (timeout_cycles > 0)) {
                progress_timestamp = c_tensix_core::read_wall_clock();
            }
        }
        all_outputs_finished = true;
        for (uint32_t i = 0; i < demux_fan_out; i++) {
            output_queues[i].prev_words_in_flight_check_flush();
            all_outputs_finished &= output_queues[i].is_remote_finished();
        }
    }

    if (!timeout) {
        debug_log_index(1, 0xff000002);
        for (uint32_t i = 0; i < demux_fan_out; i++) {
            if (!output_queues[i].output_barrier(timeout_cycles)) {
                timeout = true;
                break;
            }
        }
    }

    uint64_t cycles_elapsed = c_tensix_core::read_wall_clock() - start_timestamp;
    if (!timeout) {
        debug_log_index(1, 0xff000003);
        input_queue.send_remote_finished_notification();
    }

    debug_log_index(16, data_words_sent>>32);
    debug_log_index(17, data_words_sent);
    debug_log_index(18, cycles_elapsed>>32);
    debug_log_index(19, cycles_elapsed);

    if (timeout) {
        debug_log_index(0, PACKET_QUEUE_TEST_TIMEOUT);
    } else {
        debug_log_index(0, PACKET_QUEUE_TEST_PASS);
        debug_log_index(1, 0xff000005);
    }
    if (debug_output_verbose) {
        debug_log(0xaaaaaaaa);
        input_queue.debug_log_object();
        for (uint32_t i = 0; i < demux_fan_out; i++) {
            debug_log(0xbbbbbb00 + i);
            output_queues[i].debug_log_object();
        }
    }
}
