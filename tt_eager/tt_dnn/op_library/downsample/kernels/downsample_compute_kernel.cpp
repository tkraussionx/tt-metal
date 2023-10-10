// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/tilize.h"
//#include "debug_print.h"

namespace NAMESPACE {
void MAIN {

    uint32_t input_cb_index = get_compile_time_arg_val(0); // sharded tiled
    uint32_t untilize_cb_index = get_compile_time_arg_val(1); // 1 row of tiles (untilized)
    uint32_t untilize_downsampled_cb_index = get_compile_time_arg_val(2); // full output size
    uint32_t tilize_out_cb_index = get_compile_time_arg_val(3); // final output = sharded output
    uint32_t num_input_blocks = get_compile_time_arg_val(4);
    uint32_t num_input_tiles_per_block = get_compile_time_arg_val(5);
    uint32_t num_output_blocks = get_compile_time_arg_val(6);
    uint32_t num_output_tiles_per_block = get_compile_time_arg_val(7);
    uint32_t num_output_tiles = num_output_blocks * num_output_tiles_per_block;

    untilize_init(input_cb_index, untilize_cb_index);

    // Untilize input
    //cb_wait_front(input_cb_index, num_input_blocks * num_input_tiles_per_block);
    for(uint32_t b = 0; b < num_input_blocks; ++ b) {

        cb_reserve_back(untilize_cb_index, num_input_tiles_per_block);

        untilize_block(input_cb_index, num_input_tiles_per_block, untilize_cb_index);

        cb_push_back(untilize_cb_index, num_input_tiles_per_block);
        cb_pop_front(input_cb_index, num_input_tiles_per_block);
    }

    // Tilize downsampled input
    cb_wait_front(untilize_downsampled_cb_index, num_output_tiles);
    cb_reserve_back(tilize_out_cb_index, num_output_tiles);

    tilize_init(untilize_downsampled_cb_index, num_output_tiles_per_block, tilize_out_cb_index);

    for(uint32_t b=0;b<num_output_blocks;++b)
    {
        cb_wait_front(untilize_downsampled_cb_index, num_output_tiles_per_block);
        cb_reserve_back(tilize_out_cb_index, num_output_tiles_per_block);

        tilize_block(untilize_downsampled_cb_index, num_output_tiles_per_block, tilize_out_cb_index);

        cb_push_back(tilize_out_cb_index, num_output_tiles_per_block);
        cb_pop_front(untilize_downsampled_cb_index, num_output_tiles_per_block);
    }
}
}
