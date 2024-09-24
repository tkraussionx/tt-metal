// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "debug/dprint.h"

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

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in2 = 2;  // Sharded cb

    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    constexpr uint32_t shard_size_bytes = shard_width_in_tiles * shard_height_in_tiles * in0_single_tile_size_bytes;
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


    cb_reserve_back(cb_id_in2, batch * shard_width_in_tiles * shard_height_in_tiles);
    cb_reserve_back(cb_id_in0, batch * in0_block_num_tiles);

    uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
    uint32_t local_shard_read_addr = get_read_ptr(cb_id_in2);

    for (uint32_t b = 0; b < batch; ++b) {

        uint32_t curr_ring_idx = ring_idx;
        for (uint32_t shard_cnt = 0; shard_cnt < ring_size && !SKIP; shard_cnt++) {

            uint32_t curr_shard_write_addr = l1_write_addr_in0 + shard_size_bytes * curr_ring_idx;
            uint64_t remote_curr_shard_write_addr = get_noc_addr(next_core_noc_x, next_core_noc_y, curr_shard_write_addr);
            uint32_t curr_shard_read_addr = curr_shard_write_addr;

            // Wait for signal from previous core that data has been added to this core's in0
            noc_semaphore_wait_min(l1_signal_sem_addr, shard_cnt + 1);

            if (shard_cnt == 0) { // Need to load the local shard from cb2 to cb0 in the correct place
                noc_async_read(get_noc_addr(local_shard_read_addr), curr_shard_write_addr, shard_size_bytes);
                noc_async_read_barrier();
            }

            // Do stuff for matmul fusion here


            /* Here, assume cb0 has the data the data ready in the correct place. */

            // Send data to next core
            if (shard_cnt < ring_size - 1) { // Skip sending the last shard
                noc_async_write(curr_shard_read_addr, remote_curr_shard_write_addr, shard_size_bytes);

                // Signal the next core that data is ready
                uint64_t remote_signal_semaphore_addr = get_noc_addr(next_core_noc_x, next_core_noc_y, signal_semaphore_addr);
                noc_semaphore_inc(remote_signal_semaphore_addr, 1);
            }

            // Update the ring index
            curr_ring_idx = (curr_ring_idx + 1) % ring_size;
       }
        cb_push_back(cb_id_in0, in0_block_num_tiles);
    }
}
