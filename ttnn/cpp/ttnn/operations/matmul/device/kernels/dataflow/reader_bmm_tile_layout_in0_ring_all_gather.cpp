// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "debug/dprint.h"

FORCE_INLINE void set(uint32_t addr, uint32_t val) {
    volatile tt_l1_ptr uint32_t* l1_addr1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    *l1_addr1 = val;
}

FORCE_INLINE void spin_wait_min(uint32_t addr, uint32_t val) {
    volatile tt_l1_ptr uint32_t* l1_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    while (*l1_addr < val) { // The first byte contains the local semaphore value
        invalidate_l1_cache();
    }
}

FORCE_INLINE void inc(uint32_t addr) {
    volatile tt_l1_ptr uint32_t* l1_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
    (*l1_addr)++;
}

// A function that calls increment, and then uses the new value to write it into the remote address of another core
FORCE_INLINE void inc_and_write(uint32_t addr, uint64_t remote_addr, uint32_t size) {
    inc(addr);
    noc_async_write(addr, remote_addr, size);
}

void kernel_main() {

    constexpr bool SKIP = false;

    // Compile time args
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t in0_block_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);

    constexpr uint32_t shard_width_in_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t shard_height_in_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(5);

    constexpr uint32_t batch = get_compile_time_arg_val(6);

    // All Gather specific
    constexpr uint32_t ring_size = get_compile_time_arg_val(7);
    uint32_t signal_semaphore_addr = get_semaphore(get_compile_time_arg_val(8));

    // Runtime args
    uint32_t rt_args_idx = 0;
    uint32_t ring_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t next_core_noc_x = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t next_core_noc_y = get_arg_val<uint32_t>(rt_args_idx++);

    volatile tt_l1_ptr uint32_t* l1_signal_sem_addr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(signal_semaphore_addr);
    uint64_t remote_signal_semaphore_addr = get_noc_addr(next_core_noc_x, next_core_noc_y, signal_semaphore_addr);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in2 = 2;  // Sharded cb

    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    constexpr uint32_t shard_size_in_tiles = shard_width_in_tiles * shard_height_in_tiles;
    constexpr uint32_t shard_size_bytes = shard_size_in_tiles* in0_single_tile_size_bytes;
    constexpr uint32_t shard_read_stride = shard_width_in_tiles * in0_single_tile_size_bytes;
    constexpr uint32_t shard_read_width = in0_single_tile_size_bytes * in0_block_w;

    // // Dprint some info
    // DPRINT << "[" << ring_idx << "] ring_size: " << ring_size << ENDL();
    // DPRINT << "[" << ring_idx << "] signal_semaphore_addr: " << signal_semaphore_addr << ENDL();
    // DPRINT << "[" << ring_idx << "] to: (" << next_core_noc_x << ", " << next_core_noc_y << ")" << ENDL();
    // DPRINT << "[" << ring_idx << "] in0_block_num_tiles: " << in0_block_num_tiles << ENDL();
    // DPRINT << "[" << ring_idx << "] in0_single_tile_size_bytes: " << in0_single_tile_size_bytes << ENDL();
    // DPRINT << "[" << ring_idx << "] shard_width_in_tiles: " << shard_width_in_tiles << ENDL();
    // DPRINT << "[" << ring_idx << "] shard_height_in_tiles: " << shard_height_in_tiles << ENDL();
    // DPRINT << "[" << ring_idx << "] shard_size_bytes: " << shard_size_bytes << ENDL();
    // DPRINT << "[" << ring_idx << "] shard_read_stride: " << shard_read_stride << ENDL();
    // DPRINT << "[" << ring_idx << "] shard_read_width: " << shard_read_width << ENDL();


    cb_reserve_back(cb_id_in2, batch * shard_size_in_tiles);
    cb_reserve_back(cb_id_in0, batch * ring_size * shard_size_in_tiles + 1);


    uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
    uint32_t local_shard_read_addr = get_read_ptr(cb_id_in2);

    // Set the signal semaphores to 0
    for (uint32_t i = 1; i <= ring_size; i++) {
        uint32_t local_signal_addr = l1_write_addr_in0 + shard_size_bytes * i;
        set(local_signal_addr, 0);
    }

    for (uint32_t b = 0; b < batch; ++b) {

        for (uint32_t shard_cnt = 0; shard_cnt < ring_size && !SKIP; shard_cnt++) {

            uint32_t curr_shard_write_addr = l1_write_addr_in0 + shard_size_bytes * (shard_cnt + 1);
            uint64_t remote_curr_shard_write_addr = get_noc_addr(next_core_noc_x, next_core_noc_y, curr_shard_write_addr);
            uint32_t curr_shard_read_addr = l1_write_addr_in0 + shard_size_bytes * shard_cnt;

            if (shard_cnt == 0) { // Need to load the local shard from cb2 to cb0 in the correct place
                noc_async_read(get_noc_addr(local_shard_read_addr), curr_shard_read_addr, shard_size_bytes);
                noc_async_read_barrier();
            } else {
                // Wait for signal from previous core that data has been added to this core's in0
                uint32_t local_signal_addr = curr_shard_read_addr + shard_size_bytes;
                spin_wait_min(local_signal_addr, 1);
            }

            /* Here, assume cb0 has the data the data ready in the correct place. */

            // Send data to next core
            if (shard_cnt < ring_size - 1) { // Skip sending the last shard
                uint32_t next_local_signal_addr = curr_shard_read_addr + shard_size_bytes;
                inc(next_local_signal_addr);
                noc_async_write(curr_shard_read_addr, remote_curr_shard_write_addr, shard_size_bytes + 4); // 4 bytes for the signal
            }

            // Do stuff for matmul fusion here
            cb_push_back(cb_id_in0, shard_size_in_tiles);
       }
    }
}
