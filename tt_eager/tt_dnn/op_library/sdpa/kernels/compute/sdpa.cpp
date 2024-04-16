// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "debug/dprint.h"

namespace NAMESPACE {


// void matmul_blocks()

void MAIN {

    uint32_t B         = get_arg_val<uint32_t>(0);
    uint32_t NQH         = get_arg_val<uint32_t>(1);
    uint32_t NKH       = get_arg_val<uint32_t>(2);
    uint32_t St       = get_arg_val<uint32_t>(3);
    uint32_t DHt      = get_arg_val<uint32_t>(4);
    uint32_t S_chunk_t    = get_arg_val<uint32_t>(5);
    uint32_t num_chunks    = get_arg_val<uint32_t>(6);

    // PACK(DPRINT all the above variables
    // PACK(DPRINT << "COMPUTE: B=" << B << ENDL());
    // PACK(DPRINT << "COMPUTE: NQH=" << NQH << ENDL());
    // PACK(DPRINT << "COMPUTE: NKH=" << NKH << ENDL());
    // PACK(DPRINT << "COMPUTE: St=" << St << ENDL());
    // PACK(DPRINT << "COMPUTE: DHt=" << DHt << ENDL());
    // PACK(DPRINT << "COMPUTE: S_chunk_t=" << S_chunk_t << ENDL());
    // PACK(DPRINT << "COMPUTE: num_chunks=" << num_chunks << ENDL());


    const uint32_t q_chunk_tiles = S_chunk_t * DHt;
    const uint32_t k_chunk_tiles = S_chunk_t * DHt;
    const uint32_t qk_chunk_tiles = S_chunk_t * S_chunk_t;

    constexpr uint32_t in0_block_w = 1;
    constexpr uint32_t subblock_w = 1;
    constexpr uint32_t subblock_h = 1;
    constexpr uint32_t out_subblock_num_tiles = 1;


    // PACK(DPRINT << "COMPUTE: q_chunk_tiles=" << q_chunk_tiles << ENDL());
    // PACK(DPRINT << "COMPUTE: k_chunk_tiles=" << k_chunk_tiles << ENDL());
    // PACK(DPRINT << "COMPUTE: qk_chunk_tiles=" << qk_chunk_tiles << ENDL());

    constexpr uint32_t cb_q_in = tt::CB::c_in0;
    constexpr uint32_t cb_k_in = tt::CB::c_in1;
    constexpr uint32_t cb_v_in = tt::CB::c_in2;

    constexpr uint32_t cb_qk_im = tt::CB::c_intermed0;
    constexpr uint32_t cb_out = tt::CB::c_out0;

    mm_init();

    for (uint32_t nb = 0; nb < B; ++nb) {
        PACK(DPRINT << "COMPUTE: "  << "nb=" << nb << ENDL());
        for (uint32_t nq = 0; nq < NQH; ++nq) {
            PACK(DPRINT << "COMPUTE: "  << "nq=" << nq << ENDL());
            for (uint32_t q_chunk = 0; q_chunk < num_chunks; ++q_chunk) {
                PACK(DPRINT << "COMPUTE: "  << "q_chunk=" << q_chunk << ENDL());
                // Get Q chunk
                cb_wait_front(cb_q_in, q_chunk_tiles);
                cb_reserve_back(cb_out, q_chunk_tiles);

                for (uint32_t k_chunk = 0; k_chunk < num_chunks; ++k_chunk) {
                    PACK(DPRINT << "COMPUTE: "  << "k_chunk=" << k_chunk << ENDL());
                    // K chunk
                    cb_wait_front(cb_k_in, k_chunk_tiles);

                    /*
                    cb_q: [S_chunk_t * DHt]
                    cb_K: [DHt * S_chunk_t]
                    */
                    const uint32_t in0_num_subblocks = S_chunk_t / subblock_h;
                    const uint32_t in1_num_subblocks = S_chunk_t / subblock_w;
                    const uint32_t num_blocks = DHt / in0_block_w;
                    bool spill = num_blocks > 1;
                    bool enable_reload = false;

                    for (uint32_t block = 0; block < num_blocks; ++block) {
                        PACK(DPRINT << "COMPUTE: "  << "block=" << block << ENDL());
                        // bool last_out = block == (num_blocks - 1);
                        cb_reserve_back(cb_qk_im, qk_chunk_tiles);

                        for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
                            PACK(DPRINT << "COMPUTE: "  << "in0_subblock=" << in0_subblock << ENDL());
                            for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
                                PACK(DPRINT << "COMPUTE: "  << "in1_subblock=" << in1_subblock << ENDL());
                                acquire_dst(tt::DstMode::Half);
                                // Reload partial
                                if (enable_reload) {
                                    copy_tile_to_dst_init_short();
                                    // cb_wait_front(cb_qk_im, out_subblock_num_tiles);
                                    for (uint32_t i = 0; i < subblock_h; i++) {
                                        for (uint32_t j = 0; j < subblock_w; j++) {
                                            uint32_t im_i = in0_subblock * subblock_h + i;
                                            uint32_t im_j = in1_subblock * subblock_w + j;
                                            uint32_t im_index = im_i * S_chunk_t + im_j;
                                            copy_tile(cb_qk_im, im_index, i);
                                        }
                                    }
                                    // cb_pop_front(cb_qk_im, out_subblock_num_tiles);
                                    mm_init_short();
                                }
                                // Loop over in0_subblock, in1_subblock, and in0_block_w
                                int dst_index = 0;
                                for (uint32_t h = 0; h < subblock_h; h++) {
                                    for (uint32_t w = 0; w < subblock_w; w++) {
                                        // int in1_index_inner_dim_offset = 0;
                                        for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                                            uint32_t i_idx = in0_subblock * subblock_h + h;
                                            uint32_t j_idx = in1_subblock * subblock_w + w;
                                            uint32_t k_idx = block * in0_block_w + inner_dim;
                                            uint32_t in0_index = i_idx * DHt + k_idx;
                                            uint32_t in1_index = k_idx * S_chunk_t + j_idx;
                                            // int in0_index = in0_subblock * subblock_h + in0_index_h_offset + inner_dim;

                                            // int in1_index = in1_subblock * subblock_w + in1_index_inner_dim_offset + w;
                                            matmul_tiles(cb_q_in, cb_k_in, in0_index, in1_index, dst_index, false /* transpose */);
                                        }
                                        dst_index++;
                                    }
                                }

                                // Move partial result to interm buffer (and output will show up here in last iteration)
                                // cb_reserve_back(cb_qk_im, out_subblock_num_tiles);
                                for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                                    pack_tile(i, cb_qk_im);
                                }
                                // cb_push_back(cb_qk_im, out_subblock_num_tiles);
                                release_dst(tt::DstMode::Half);
                            }
                        }
                        if (spill) {
                            enable_reload = true;
                        }
                        // Free up space in intermediate CB
                        cb_push_back(cb_qk_im, qk_chunk_tiles);
                        cb_wait_front(cb_qk_im, qk_chunk_tiles);
                        cb_pop_front(cb_qk_im, qk_chunk_tiles);
                    }

                    // cb_qk_im contains row-major output of Q @ Kt
                    cb_pop_front(cb_k_in, k_chunk_tiles);

                    // V chunk
                    cb_wait_front(cb_v_in, k_chunk_tiles);
                    // Multiply Q @ Kt with V

                    cb_pop_front(cb_v_in, k_chunk_tiles);
                }

                // Commit Q chunk output)
                cb_push_back(cb_out, q_chunk_tiles);
                cb_pop_front(cb_q_in, q_chunk_tiles);

            }
        }
    }

    // for (uint32_t b = 0; b < batch; b++){
    //     bool spill = num_blocks > 1;
    //     bool enable_reload = false;
    //     uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

    //     for(uint32_t block = 0; block < num_blocks; block++)
    //     {
    //         bool last_out = block == (num_blocks-1);

    //         cb_wait_front(tt::CB::c_in0, in0_block_num_tiles);
    //         cb_wait_front(tt::CB::c_in1, in1_block_num_tiles);
    //         int in0_index_subblock_offset = 0;
    //         for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
    //             int in1_index_subblock_offset = 0;
    //             for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; in1_subblock++) {

    //                 acquire_dst(tt::DstMode::Half);

    //                 if (enable_reload) {
    //                     copy_tile_to_dst_init_short();
    //                     cb_wait_front(tt::CB::c_intermed0, out_subblock_num_tiles);
    //                     for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
    //                         copy_tile(tt::CB::c_intermed0, i, i);
    //                     }
    //                     cb_pop_front(tt::CB::c_intermed0, out_subblock_num_tiles);
    //                     mm_init_short();
    //                 }

    //                 // Compute output sub-block from in0_subblock x in1_subblock
    //                 int dst_index = 0;
    //                 int in0_index_h_offset = 0;
    //                 for (uint32_t h = 0; h < out_subblock_h; h++) {
    //                     for (uint32_t w = 0; w < out_subblock_w; w++) {
    //                         int in1_index_inner_dim_offset = 0;
    //                         for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
    //                             int in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
    //                             int in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
    //                             matmul_tiles(tt::CB::c_in0, tt::CB::c_in1, in0_index, in1_index, dst_index, false /* transpose */);
    //                             in1_index_inner_dim_offset += in1_per_core_w;
    //                         }
    //                         dst_index++;
    //                     }
    //                     in0_index_h_offset += in0_block_w;
    //                 }

    //                 if (last_out) {
    //                     // Pack out to output buffer
    //                     cb_reserve_back(tt::CB::c_out0, out_subblock_num_tiles);
    //                     for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
    //                         pack_tile(i, tt::CB::c_out0);
    //                     }
    //                     cb_push_back(tt::CB::c_out0, out_subblock_num_tiles);
    //                 } else {
    //                     // Wait for tiles in output buffer to be written out since interm and output share memory
    //                     if (block == 0) {
    //                         cb_reserve_back(tt::CB::c_out0, out_num_tiles_to_wait);
    //                         out_num_tiles_to_wait += out_subblock_num_tiles;
    //                     }
    //                     // Move partial result to interm buffer
    //                     cb_reserve_back(tt::CB::c_intermed0, out_subblock_num_tiles);
    //                     for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
    //                         pack_tile(i, tt::CB::c_intermed0);
    //                     }
    //                     cb_push_back(tt::CB::c_intermed0, out_subblock_num_tiles);
    //                 }

    //                 release_dst(tt::DstMode::Half);
    //                 in1_index_subblock_offset += out_subblock_w;
    //             }
    //             in0_index_subblock_offset += in0_subblock_num_tiles;
    //         }

    //         if (spill) enable_reload = true;

    //         cb_pop_front(tt::CB::c_in0, in0_block_num_tiles);
    //         cb_pop_front(tt::CB::c_in1, in1_block_num_tiles);

    //     }
    // }
}
}
