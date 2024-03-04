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

void init_queue_data(uint32_t queue_start_addr_words, uint32_t queue_size_words) {
    constexpr uint32_t PACKET_SIZE_BYTES = 1024;
    volatile uint32_t* queue_ptr = reinterpret_cast<volatile uint32_t*>(queue_start_addr_words*PACKET_WORD_SIZE_BYTES);
    for (uint32_t i = 0; i < queue_size_words*PACKET_WORD_SIZE_BYTES/4; i++) {
        queue_ptr[i] = 0xbbcc0000 + i;
    }
    for (uint32_t i = 0; i < queue_size_words; i++) {
       if (i % (PACKET_SIZE_BYTES/PACKET_WORD_SIZE_BYTES) == 0) {
           dispatch_packet_header_t* header = reinterpret_cast<dispatch_packet_header_t*>(queue_start_addr_words*PACKET_WORD_SIZE_BYTES + i*PACKET_WORD_SIZE_BYTES);
           header->packet_size_words = PACKET_SIZE_BYTES/PACKET_WORD_SIZE_BYTES;
       }
    }
}


void kernel_main() {

    constexpr uint32_t input_queue_id = get_compile_time_arg_val(0);
    constexpr uint32_t output_queue_id = get_compile_time_arg_val(1);

    constexpr uint32_t queue_start_addr_words = get_compile_time_arg_val(2);
    constexpr uint32_t queue_size_words = get_compile_time_arg_val(3);

    constexpr uint32_t remote_rx_queue_start_addr_words = get_compile_time_arg_val(4);
    constexpr uint32_t remote_rx_queue_size_words = get_compile_time_arg_val(5);

    constexpr uint32_t remote_rx_x = get_compile_time_arg_val(6);
    constexpr uint32_t remote_rx_y = get_compile_time_arg_val(7);
    constexpr uint32_t remote_rx_queue_id = get_compile_time_arg_val(8);

    constexpr DispatchRemoteNetworkType tx_network_type = static_cast<DispatchRemoteNetworkType>(get_compile_time_arg_val(9));

    constexpr uint32_t debug_buf_addr = get_compile_time_arg_val(10);
    constexpr uint32_t debug_buf_size = get_compile_time_arg_val(11);

    noc_init(0xF);

    debug_set_buf(reinterpret_cast<volatile uint32_t*>(debug_buf_addr), debug_buf_size);

    debug_log(0xff000000);

    packet_input_queue_state_t* input_queue = &(input_queues[input_queue_id]);
    packet_output_queue_state_t* output_queue = &(output_queues[output_queue_id]);

    input_queue->init(input_queue_id, queue_start_addr_words, queue_size_words,
                      0, 0, 0, DispatchRemoteNetworkType::NONE); // remote_x, remote_y, remote_queue_id, remote_update_network_type

    output_queue->init(output_queue_id, remote_rx_queue_start_addr_words, remote_rx_queue_size_words,
                       remote_rx_x, remote_rx_y, remote_rx_queue_id, tx_network_type,
                       input_queues);


    init_queue_data(queue_start_addr_words, queue_size_words);

    debug_log(0xff000001);
    input_queue->debug_log_object();
    debug_log(0xff000002);
    output_queue->debug_log_object();

    input_queue->advance_queue_local_wptr(queue_size_words);
    output_queue->forward_data_from_input(input_queue_id);

    debug_log(0xff000003);
    input_queue->debug_log_object();
    debug_log(0xff000004);
    output_queue->debug_log_object();

    debug_log(0xff000004);


}
