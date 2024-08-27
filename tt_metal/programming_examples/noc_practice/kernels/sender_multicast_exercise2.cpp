// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

// This section is reserved for watcher tool practice session.
#if 0
#include "debug/assert.h"
#include "debug/pause.h"
#include "debug/ring_buffer.h"
#include "debug/status.h"
#endif

void kernel_main() {
    uint32_t l1_buffer_src_addr = get_arg_val<uint32_t>(0);
    uint32_t input_dram_buffer_addr = get_arg_val<uint32_t>(1);
    uint32_t input_dram_noc_x = get_arg_val<uint32_t>(2);
    uint32_t input_dram_noc_y = get_arg_val<uint32_t>(3);
    uint32_t dram_buffer_size = get_arg_val<uint32_t>(4);
    uint32_t l1_buffer_dest_addr = get_arg_val<uint32_t>(5);
    uint32_t dest_noc_x_start = get_arg_val<uint32_t>(6);
    uint32_t dest_noc_y_start = get_arg_val<uint32_t>(7);
    uint32_t dest_noc_x_end = get_arg_val<uint32_t>(8);
    uint32_t dest_noc_y_end = get_arg_val<uint32_t>(9);
    uint32_t num_dests = get_arg_val<uint32_t>(10);
    uint32_t mcast_sender_semaphore_addr = get_arg_val<uint32_t>(11);
    uint32_t mcast_receiver_semaphore_addr = get_arg_val<uint32_t>(12);

    volatile tt_l1_ptr uint32_t* mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_receiver_semaphore_addr);
    *(mcast_receiver_semaphore_addr_ptr) = VALID;
    volatile tt_l1_ptr uint32_t* mcast_sender_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_sender_semaphore_addr);

    // read input dram buffer into local L1 buffer
    uint64_t input_dram_buffer_noc_addr = get_noc_addr(input_dram_noc_x, input_dram_noc_y, input_dram_buffer_addr);
    noc_async_read(input_dram_buffer_noc_addr, l1_buffer_src_addr, dram_buffer_size);
    noc_async_read_barrier();

// This section is reserved for watcher tool practice session.
#if 0
    for (uint32_t idx = 0; idx < 40; idx++) {
        WATCHER_RING_BUFFER_PUSH(idx + 1);
    }
#endif

// This section is reserved for watcher tool practice session.
#if 0
    PAUSE();
#endif

// This section is reserved for watcher tool practice session.
#if 0
    DEBUG_STATUS("AST1");
    ASSERT(false);
#endif

    /* TODO: fill in the parameters */
    noc_semaphore_wait(mcast_sender_semaphore_addr_ptr, num_dests);

    // multicast local L1 buffer to all destination cores
    uint64_t dest_noc_multicast_addr =
        get_noc_multicast_addr(dest_noc_x_start, dest_noc_y_start, dest_noc_x_end, dest_noc_y_end, l1_buffer_dest_addr);
    noc_async_write_multicast(l1_buffer_src_addr, dest_noc_multicast_addr, dram_buffer_size, num_dests);
    noc_async_write_barrier();

    uint64_t mcast_receiver_semaphore_noc_addr = get_noc_multicast_addr(
        dest_noc_x_start, dest_noc_y_start, dest_noc_x_end, dest_noc_y_end, mcast_receiver_semaphore_addr);
    noc_semaphore_set_multicast(mcast_receiver_semaphore_addr, mcast_receiver_semaphore_noc_addr, num_dests);
}
