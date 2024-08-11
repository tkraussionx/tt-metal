// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"

namespace NAMESPACE {
void MAIN {

    uint32_t in0_block_w = get_compile_time_arg_val(0); // inner block size in tiles
    uint32_t in0_num_subblocks = get_compile_time_arg_val(1); // outer row block size (in inner row blocks)
    uint32_t in0_block_num_tiles = get_compile_time_arg_val(2); // out_subblock_h*in0_block_w*in0_num_subblocks;
    uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);  // out_subblock_h*in0_block_w
    uint32_t in1_num_subblocks = get_compile_time_arg_val(4); // outer column block size (in inner column blocks)
    uint32_t in1_block_num_tiles = get_compile_time_arg_val(5); //out_subblock_w*in0_block_w* in1_num_subblocks;
    uint32_t in1_per_core_w = get_compile_time_arg_val(6); // out_subblock_w*in1_num_subblocks
    uint32_t num_blocks = get_compile_time_arg_val(7);  // outer inner dim (in inner dim blocks)
    uint32_t out_subblock_h = get_compile_time_arg_val(8); // inner row block size in tiles
    uint32_t out_subblock_w = get_compile_time_arg_val(9); // inner column block size in tiles
    uint32_t out_subblock_num_tiles = get_compile_time_arg_val(10); // out_subblock_h * out_subblock_w;

    constexpr uint32_t in0_cb_id = tt::CB::c_in0;
    constexpr uint32_t in1_cb_id = tt::CB::c_in1;
    constexpr uint32_t out_cb_id = tt::CB::c_out0;
    constexpr uint32_t mm_partials_cb_id = tt::CB::c_intermed0;

    // TODO: Initialize mm using the `mm_init` API
    // mm_init();

    bool spill = num_blocks > 1;
    bool enable_reload = false;
    uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

    ///////////////////////////////////////////////
    // TODO: Fill the /* */ with appropriate code
    ///////////////////////////////////////////////
    for(uint32_t block = 0; block < /* */; block++)
    {
        bool last_out = block == (num_blocks-1);

        cb_wait_front(in0_cb_id, /* */);
        cb_wait_front(in1_cb_id, /* */);
        int in0_index_subblock_offset = 0;
        for (uint32_t in0_subblock = 0; in0_subblock < /* */; in0_subblock++) {
            int in1_index_subblock_offset = 0;
            for (uint32_t in1_subblock = 0; in1_subblock < /* */; in1_subblock++) {

                tile_regs_acquire();
                if (enable_reload) {
                    copy_tile_to_dst_init_short();
                    cb_wait_front(mm_partials_cb_id, out_subblock_num_tiles);
                    // TODO: Copy tiles using `copy_tile` with a loop
                    cb_pop_front(mm_partials_cb_id, out_subblock_num_tiles);
                    mm_init_short();
                }

                // Compute output sub-block from in0_subblock x in1_subblock
                int dst_index = 0;
                int in0_index_h_offset = 0;
                for (uint32_t h = 0; h < /* */; h++) {
                    for (uint32_t w = 0; w < /* */; w++) {
                        int in1_index_inner_dim_offset = 0;
                        for (uint32_t inner_dim = 0; inner_dim < /* */; inner_dim++) {
                            // TODO: Update `in0_index` and `in1_index`
                            int in0_index = /* */ + /* */ + /* */;
                            int in1_index = /* */ + /* */ + /* */;
                            matmul_tiles(/* */, /* */, /* */, /* */, /* */, false /* transpose */);
                            in1_index_inner_dim_offset += /* */;
                        }
                        /* */++;
                    }
                    in0_index_h_offset += /* */;
                }

                tile_regs_commit();

                tile_regs_wait();
                if (last_out) {
                    // Pack out to output buffer
                    cb_reserve_back(out_cb_id, out_subblock_num_tiles);
                    // TODO: Pack tiles using `pack_tile` with a loop
                    cb_push_back(out_cb_id, out_subblock_num_tiles);
                } else {
                    // Wait for tiles in output buffer to be written out since interm and output share memory
                    if (block == 0) {
                        cb_reserve_back(out_cb_id, out_num_tiles_to_wait);
                        out_num_tiles_to_wait += out_subblock_num_tiles;
                    }
                    // Move partial result to interm buffer
                    cb_reserve_back(mm_partials_cb_id, out_subblock_num_tiles);
                    // TODO: Pack tiles using `pack_tile` with a loop
                    cb_push_back(mm_partials_cb_id, out_subblock_num_tiles);
                }

                tile_regs_release();
                in1_index_subblock_offset += /* */;
            }
            in0_index_subblock_offset += /* */;
        }

        if (spill) enable_reload = true;

        cb_pop_front(in0_cb_id, in0_block_num_tiles);
        cb_pop_front(in1_cb_id, in1_block_num_tiles);

    }
}
}
