// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void kernel_main() {

    uint32_t rt_args_idx = 0;
    uint32_t target_noc_x = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t target_noc_y = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t target_noc_sem_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_tiles = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_transfers = get_arg_val<uint32_t>(rt_args_idx++);

    constexpr uint32_t cb0_id = 0;
    uint32_t transfer_size = get_tile_size(cb0_id) * num_tiles;

    cb_reserve_back(cb0_id, num_tiles);
    uint32_t cb0_addr = get_write_ptr(cb0_id);

    uint64_t target_noc_addr = get_noc_addr(target_noc_x, target_noc_y, cb0_addr);
    uint64_t target_noc_sem_addr = get_noc_addr(target_noc_x, target_noc_y, get_semaphore(target_noc_sem_id));

    for (uint32_t i = 0; i < num_transfers; i++) {
        noc_async_write(cb0_addr, target_noc_addr, transfer_size);
    }
    noc_semaphore_inc(target_noc_sem_addr, num_transfers);
}
