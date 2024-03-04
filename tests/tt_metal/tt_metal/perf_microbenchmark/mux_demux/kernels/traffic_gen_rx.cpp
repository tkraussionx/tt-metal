// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/dprint.h"
#include "tt_metal/impl/dispatch/kernels/packet_queue.hpp"

packet_input_queue_state_t input_queues[MAX_SWITCH_FAN_IN];
packet_output_queue_state_t output_queues[MAX_SWITCH_FAN_OUT];

volatile uint32_t* debug_buf;
uint32_t debug_buf_index;
uint32_t debug_buf_size;


void kernel_main() {

    constexpr uint32_t input_queue_id = get_compile_time_arg_val(0);
    constexpr uint32_t output_queue_id = get_compile_time_arg_val(1);

    constexpr uint32_t queue_start_addr_words = get_compile_time_arg_val(2);
    constexpr uint32_t queue_size_words = get_compile_time_arg_val(3);

    constexpr uint32_t remote_tx_x = get_compile_time_arg_val(4);
    constexpr uint32_t remote_tx_y = get_compile_time_arg_val(5);
    constexpr uint32_t remote_tx_queue_id = get_compile_time_arg_val(6);

    constexpr DispatchRemoteNetworkType rx_rptr_update_network_type = static_cast<DispatchRemoteNetworkType>(get_compile_time_arg_val(7));

    constexpr uint32_t debug_buf_addr = get_compile_time_arg_val(8);
    constexpr uint32_t debug_buf_size = get_compile_time_arg_val(9);

    noc_init(0xF);

    debug_set_buf(reinterpret_cast<volatile uint32_t*>(debug_buf_addr), debug_buf_size);

    debug_log(0xee000000);

    packet_input_queue_state_t* input_queue = &(input_queues[input_queue_id]);
    packet_output_queue_state_t* output_queue = &(output_queues[output_queue_id]);

    input_queue->init(input_queue_id, queue_start_addr_words, queue_size_words,
                      remote_tx_x, remote_tx_y, remote_tx_queue_id,
                      rx_rptr_update_network_type);

    output_queue->init(output_queue_id, queue_start_addr_words, queue_size_words,
                       0, 0, 0, DispatchRemoteNetworkType::NONE, // remote_rx_x, remote_rx_y, remote_rx_queue_id, tx_network_type,
                       input_queues);


    debug_log(0xee000001);

}
