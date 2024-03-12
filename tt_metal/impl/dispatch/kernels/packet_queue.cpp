// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "packet_queue.hpp"

bool wait_all_src_dest_ready(packet_input_queue_state_t* input_queue_array, uint32_t num_input_queues,
                             packet_output_queue_state_t* output_queue_array, uint32_t num_output_queues,
                             uint64_t timeout_cycles = 0) {

    bool all_src_dest_ready = false;
    bool src_ready[MAX_SWITCH_FAN_IN] = {false};
    bool dest_ready[MAX_SWITCH_FAN_OUT] = {false};
    uint64_t start_timestamp = c_tensix_core::read_wall_clock();
    while (!all_src_dest_ready) {
        if (timeout_cycles > 0) {
            uint64_t cycles_since_start = c_tensix_core::read_wall_clock() - start_timestamp;
            if (cycles_since_start > timeout_cycles) {
                return false;
            }
        }
        all_src_dest_ready = true;
        for (uint32_t i = 0; i < num_input_queues; i++) {
            if (!src_ready[i]) {
                src_ready[i] = input_queue_array[i].is_remote_ready();
                if (!src_ready[i]) {
                    input_queue_array[i].send_remote_ready_notification();
                    all_src_dest_ready = false;
                }
            }
        }
        for (uint32_t i = 0; i < num_output_queues; i++) {
            if (!dest_ready[i]) {
                dest_ready[i] = output_queue_array[i].is_remote_ready();
                if (dest_ready[i]) {
                    output_queue_array[i].send_remote_ready_notification();
                } else {
                    all_src_dest_ready = false;
                }
            }
        }
    }
    return true;
}
