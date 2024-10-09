// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {

    uint32_t rt_args_idx = 0;
    uint32_t noc_sem_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_transfers = get_arg_val<uint32_t>(rt_args_idx++);

    volatile tt_l1_ptr uint32_t* noc_sem_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(noc_sem_id));

    noc_semaphore_wait(noc_sem_addr, num_transfers);

}
