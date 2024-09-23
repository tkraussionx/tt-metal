// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"
#include "hostdevcommon/common_values.hpp"

void kernel_main() {

    // Compile time args
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t in0_block_size_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(2);

    constexpr uint32_t shard_width_in_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t shard_height_in_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(5);

    constexpr uint32_t batch = get_compile_time_arg_val(6);

    // Runtime args
    uint32_t rt_args_idx = 0;

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in2 = 2;  // Sharded cb

    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);
    constexpr DataFormat in0_data_format = get_dataformat(cb_id_in0);

    constexpr uint32_t num_blocks_per_shard = shard_width_in_tiles / in0_block_w;
    constexpr uint32_t shard_read_stride = shard_width_in_tiles * in0_single_tile_size_bytes;
    constexpr uint32_t shard_read_width = in0_single_tile_size_bytes * in0_block_w;

    cb_reserve_back(cb_id_in2, batch * in0_block_num_tiles);

    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t block = 0; block < num_blocks; ++block) {

            cb_reserve_back(cb_id_in0, in0_block_num_tiles);

            uint64_t noc_shard_read_start_addr = get_noc_addr(get_read_ptr(cb_id_in2));
            uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);

            uint32_t l1_write_extract_shard_in0 = l1_write_addr_in0;
            uint64_t noc_shard_read_addr = noc_shard_read_start_addr;

            for (uint32_t i = 0; i < shard_height_in_tiles; i++) {
                noc_async_read(noc_shard_read_addr, l1_write_extract_shard_in0, shard_read_width);

                l1_write_extract_shard_in0 += shard_read_width;
                noc_shard_read_addr += shard_read_stride;
            }
            noc_async_read_barrier();

            cb_push_back(cb_id_in0, in0_block_num_tiles);
       }
    }
}
