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
    uint32_t out_block_num_tiles = get_compile_time_arg_val(11); // per_core_Mt * per_core_Nt

    constexpr uint32_t in0_cb_id = tt::CB::c_in0;
    constexpr uint32_t in1_cb_id = tt::CB::c_in1;
    constexpr uint32_t out_cb_id = tt::CB::c_out0;
    constexpr uint32_t mm_partials_cb_id = tt::CB::c_intermed0;

    mm_block_init(in0_cb_id, in1_cb_id, mm_partials_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);

    bool spill = num_blocks > 1;
    bool enable_reload = false;
    uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

    for(uint32_t block = 0; block < num_blocks; block++)
    {
        bool last_out = block == (num_blocks-1);

        cb_wait_front(in0_cb_id, in0_block_num_tiles);
        cb_wait_front(in1_cb_id, in1_block_num_tiles);
        int in0_index_subblock_offset = 0;
        for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
            int in1_index_subblock_offset = 0;
            for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {

                tile_regs_acquire();
                if (enable_reload) {
                    copy_tile_to_dst_init_short();
                    cb_wait_front(mm_partials_cb_id, out_subblock_num_tiles);
                    uint32_t start_dst_index = 0;
                    uint32_t start_tile_index = 0;
                    copy_block_matmul_partials(
                        mm_partials_cb_id, start_tile_index, start_dst_index, out_subblock_num_tiles);
                    cb_pop_front(mm_partials_cb_id, out_subblock_num_tiles);
                    mm_block_init_short(in0_cb_id, in1_cb_id, 0, out_subblock_w, out_subblock_h, in0_block_w);
                }

                // Compute output sub-block from in0_subblock x in1_subblock
                uint32_t dst_index = 0;  // start at 0, each call to matmul_block internally increments dst_index
                uint32_t in0_index = in0_index_subblock_offset;  // offset into in0 block
                uint32_t in1_index = in1_index_subblock_offset;  // offset into in1 block
                // inner dim that we accumualte is the inner dim of in0/in1, which is in0_block_w
                for (uint32_t inner_dim_idx = 0; inner_dim_idx < in0_block_w; ++inner_dim_idx) {
                    // matmul outer product of (out_subblock_h x out_subblock_w) tiles that fill dst
                    // accumulation is done by iterating matmul_block across inner dim
                    // in0_block_w is passed as innder dim (kt) to matmul_block, interally used to stride in0
                    matmul_block(
                        in0_cb_id,
                        in1_cb_id,
                        in0_index,
                        in1_index,
                        dst_index,
                        false,
                        out_subblock_w,
                        out_subblock_h,
                        in0_block_w);
                    in0_index++;                  // stride right by 1
                    in1_index += in1_per_core_w;  // to stride down by 1 need to stride by in_per_core_w (should be
                                                  // called in1_block_w)
                }

                tile_regs_commit();

                tile_regs_wait();
                if (last_out) {
                    // Pack out to output buffer
                    PACK((llk_pack_reconfig_l1_acc(0)));
                    cb_reserve_back(out_cb_id, out_subblock_num_tiles);
                    uint32_t start_dst_index = 0;
                    matmul_pack_tile(start_dst_index, mm_partials_cb_id, out_subblock_num_tiles);
                    cb_push_back(out_cb_id, out_subblock_num_tiles);
                } else {
                    // Wait for tiles in output buffer to be written out since interm and output share memory
                    if (block == 0) {
                        cb_reserve_back(out_cb_id, out_num_tiles_to_wait);
                        out_num_tiles_to_wait += out_subblock_num_tiles;
                    }
                    // Move partial result to interm buffer
                    if (block == 0) {  // no accumulation for first iteration
                        PACK((llk_pack_reconfig_l1_acc(0)));
                    } else if (block == 1) {
                        PACK((llk_pack_reconfig_l1_acc(1)));
                    }
                    cb_reserve_back(mm_partials_cb_id, out_subblock_num_tiles);
                    uint32_t start_dst_index = 0;
                    matmul_pack_tile(start_dst_index, mm_partials_cb_id, out_subblock_num_tiles);
                    cb_push_back(mm_partials_cb_id, out_subblock_num_tiles);
                }

                tile_regs_release();
                in1_index_subblock_offset += out_subblock_w;
            }
            in0_index_subblock_offset += in0_subblock_num_tiles;
        }

        // Last iteration does spill and reload to output buffer
        if (block < num_blocks - 2) {
            cb_wait_front(mm_partials_cb_id, out_block_num_tiles);
            cb_pop_front(mm_partials_cb_id, out_block_num_tiles);
        }
        if (block == num_blocks - 2) {
            enable_reload = true;
        }  // reload when last iteration

        cb_pop_front(in0_cb_id, in0_block_num_tiles);
        cb_pop_front(in1_cb_id, in1_block_num_tiles);

    }
}
}
