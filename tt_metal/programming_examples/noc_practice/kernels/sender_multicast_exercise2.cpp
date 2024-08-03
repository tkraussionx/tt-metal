// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    uint32_t l1_buffer_src_addr = get_arg_val<uint32_t>(0);
    uint32_t dram_input_buffer_addr = get_arg_val<uint32_t>(1);
    uint32_t dram_input_noc_x = get_arg_val<uint32_t>(2);
    uint32_t dram_input_noc_y = get_arg_val<uint32_t>(3);
    uint32_t dram_buffer_size      = get_arg_val<uint32_t>(4);
    uint32_t l1_buffer_dest_addr = get_arg_val<uint32_t>(5);
    uint32_t dest_noc_x_start = get_arg_val<uint32_t>(6);
    uint32_t dest_noc_y_start = get_arg_val<uint32_t>(7);
    uint32_t dest_noc_x_end = get_arg_val<uint32_t>(8);
    uint32_t dest_noc_y_end = get_arg_val<uint32_t>(9);
    uint32_t num_dests = get_arg_val<uint32_t>(10);
    uint32_t mcast_sender_semaphore_addr    = get_arg_val<uint32_t>(11);
    uint32_t mcast_receiver_semaphore_addr  = get_arg_val<uint32_t>(12);

    volatile tt_l1_ptr uint32_t* mcast_receiver_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_receiver_semaphore_addr);
    *(mcast_receiver_semaphore_addr_ptr) = VALID;
    volatile tt_l1_ptr uint32_t* mcast_sender_semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_sender_semaphore_addr);

    // read dram input buffer into local L1 buffer
    uint64_t dram_buffer_input_noc_addr = get_noc_addr(dram_input_noc_x, dram_input_noc_y, dram_input_buffer_addr);
    noc_async_read(dram_buffer_input_noc_addr, l1_buffer_src_addr, dram_buffer_size);
    noc_async_read_barrier();

    noc_semaphore_wait(mcast_sender_semaphore_addr_ptr, num_dests);
    noc_semaphore_set(mcast_sender_semaphore_addr_ptr, 0);

    // multicast local L1 buffer to all destination cores
    uint64_t dest_noc_multicast_addr =
        get_noc_multicast_addr(dest_noc_x_start, dest_noc_y_start, dest_noc_x_end, dest_noc_y_end, l1_buffer_dest_addr);
    noc_async_write_multicast(l1_buffer_src_addr, dest_noc_multicast_addr, dram_buffer_size, num_dests);
    noc_async_write_barrier();

    uint64_t mcast_receiver_semaphore_noc_addr = get_noc_multicast_addr(
        dest_noc_x_start, dest_noc_y_start, dest_noc_x_end, dest_noc_y_end, mcast_receiver_semaphore_addr);
    noc_semaphore_set_multicast(mcast_receiver_semaphore_addr, mcast_receiver_semaphore_noc_addr, num_dests);
}
