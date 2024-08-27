// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {
    uint32_t mcast_sender_noc_x = get_arg_val<uint32_t>(0);
    uint32_t mcast_sender_noc_y = get_arg_val<uint32_t>(1);
    uint32_t mcast_sender_semaphore_addr = get_arg_val<uint32_t>(2);
    uint32_t mcast_receiver_semaphore_addr = get_arg_val<uint32_t>(3);

    volatile tt_l1_ptr uint32_t* mcast_receiver_semaphore_addr_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(mcast_receiver_semaphore_addr);
    /* TODO: fill in the parameters */
    noc_semaphore_set(mcast_receiver_semaphore_addr_ptr, INVALID);

    // Atomic increment source core counter
    uint64_t mcast_sender_semaphore_noc_addr =
        get_noc_addr(mcast_sender_noc_x, mcast_sender_noc_y, mcast_sender_semaphore_addr);

    noc_semaphore_inc(mcast_sender_semaphore_noc_addr, 1);

    noc_semaphore_wait(mcast_receiver_semaphore_addr_ptr, VALID);
}
