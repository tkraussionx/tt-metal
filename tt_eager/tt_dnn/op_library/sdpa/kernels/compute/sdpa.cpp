// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "debug/dprint.h"

namespace NAMESPACE {

void mul_block_bcast_scalar_inplace(uint32_t in0_cb, uint32_t in1_scalar_cb, uint32_t num_tiles) {
    // Precondition: in0_cb has num_tiles produced
    // Precondition: in1_scalar_cb has 1 produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_scalar_cb has 1 produced
    unpack_reconfig_data_format(in0_cb, in1_scalar_cb);
    pack_reconfig_data_format(in0_cb);
    cb_wait_front(in1_scalar_cb, 1);
    mul_tiles_bcast_scalar_init_short();
    for (uint32_t i = 0; i < num_tiles; ++i) {
        PACK(DPRINT << "COMPUTE: MUL_BCAST i: " << i << ENDL());
        acquire_dst(tt::DstMode::Half);
        // cb_wait_front(in0_cb, 1);
        PACK(DPRINT << "COMPUTE: MUL_BCAST wait_front i: " << i << ENDL());
        mul_tiles_bcast_scalar(in0_cb, in1_scalar_cb, 0, 0, 0);
        cb_pop_front(in0_cb, 1);
        // cb_reserve_back(in0_cb, 1);
        PACK(DPRINT << "COMPUTE: MUL_BCAST reserve_back i: " << i << ENDL());
        pack_tile(0, in0_cb);
        cb_push_back(in0_cb, 1);
        release_dst(tt::DstMode::Half);
    }
}

void add_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles consumed

    add_tiles_init();

    cb_wait_front(in1_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst(tt::DstMode::Half);
        cb_wait_front(in0_cb, 1);
        add_tiles(in0_cb, in1_cb, 0, i, 0);
        cb_pop_front(in0_cb, 1);
        cb_reserve_back(in0_cb, 1);
        pack_tile(0, in0_cb);
        cb_push_back(in0_cb, 1);
        release_dst(tt::DstMode::Half);
    }

    cb_pop_front(in1_cb, num_tiles);
}

void copy_block(uint32_t in_cb, uint32_t out_cb, uint32_t num_tiles) {
    // Precondition: in_cb has num_tiles produced
    // Precondition: out_cb has num_tiles free
    // Postcondition: in_cb has num_tiles consumed
    // Postcondition: out_cb has num_tiles produced

    copy_tile_to_dst_init_short();

    cb_wait_front(in_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);

    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst(tt::DstMode::Half);
        copy_tile(in_cb, i, 0/*dst*/);
        pack_tile(0, out_cb);
        cb_push_back(out_cb, 1);
        release_dst(tt::DstMode::Half);
    }
    cb_pop_front(in_cb, num_tiles);
}

void matmul_blocks(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t M, uint32_t N, uint32_t K, uint32_t num_blocks, uint32_t in0_num_subblocks, uint32_t in1_num_subblocks,
                    uint32_t in0_block_w, uint32_t subblock_h, uint32_t subblock_w) {
    bool spill = num_blocks > 1;
    bool enable_reload = false;
    mm_init_short();

    uint32_t output_num_tiles = M * N;
    uint32_t out_subblock_num_tiles = subblock_h * subblock_w;

    for (uint32_t block = 0; block < num_blocks; ++block) {
        PACK(DPRINT << "COMPUTE: "  << "block=" << block << ENDL());

        for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
            // PACK(DPRINT << "COMPUTE: "  << "in0_subblock=" << in0_subblock << ENDL());
            for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
                // PACK(DPRINT << "COMPUTE: "  << "in1_subblock=" << in1_subblock << ENDL());
                acquire_dst(tt::DstMode::Half);
                // Reload partial
                if (enable_reload) {
                    copy_tile_to_dst_init_short();
                    cb_wait_front(out_cb, out_subblock_num_tiles);
                    // TODO: Does out_subblock have to be one row for this to work?
                    for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
                        copy_tile(out_cb, i, i);
                    }
                    cb_pop_front(out_cb, out_subblock_num_tiles);
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
                            uint32_t in0_index = i_idx * K + k_idx;
                            uint32_t in1_index = k_idx * N + j_idx;
                            matmul_tiles(in0_cb, in1_cb, in0_index, in1_index, dst_index, false /* transpose */);
                        }
                        dst_index++;
                    }
                }

                // Move partial result to interm buffer (and output will show up here in last iteration)
                cb_reserve_back(out_cb, out_subblock_num_tiles);
                for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                    pack_tile(i, out_cb);
                }
                cb_push_back(out_cb, out_subblock_num_tiles);
                release_dst(tt::DstMode::Half);
            }
        }
        if (spill) {
            enable_reload = true;
        }
    }

    // Free up out_cb. Inner loop writes `output_num_tiles` times more than it reads.
    cb_wait_front(out_cb, output_num_tiles);
    cb_pop_front(out_cb, output_num_tiles);
}

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
    const uint32_t out_chunk_tiles = S_chunk_t * DHt;

    constexpr uint32_t in0_block_w = 1;
    constexpr uint32_t subblock_w = 1;
    constexpr uint32_t subblock_h = 1;
    constexpr uint32_t out_subblock_num_tiles = 1;

    const uint32_t qk_in0_num_subblocks = S_chunk_t / subblock_h;
    const uint32_t qk_in1_num_subblocks = S_chunk_t / subblock_w;
    const uint32_t qk_num_blocks = DHt / in0_block_w;

    const uint32_t out_in0_num_subblocks = S_chunk_t / subblock_h;
    const uint32_t out_in1_num_subblocks = DHt / subblock_w;
    const uint32_t out_num_blocks = S_chunk_t / in0_block_w;

    constexpr uint32_t DST_SIZE = 8;


    // PACK(DPRINT << "COMPUTE: q_chunk_tiles=" << q_chunk_tiles << ENDL());
    // PACK(DPRINT << "COMPUTE: k_chunk_tiles=" << k_chunk_tiles << ENDL());
    // PACK(DPRINT << "COMPUTE: qk_chunk_tiles=" << qk_chunk_tiles << ENDL());

    constexpr uint32_t cb_q_in = tt::CB::c_in0;
    constexpr uint32_t cb_k_in = tt::CB::c_in1;
    constexpr uint32_t cb_v_in = tt::CB::c_in2;
    constexpr uint32_t cb_mask_in = tt::CB::c_in3;
    constexpr uint32_t cb_scale_in = tt::CB::c_in4;

    constexpr uint32_t cb_qk_im = tt::CB::c_intermed0;
    constexpr uint32_t cb_out_im = tt::CB::c_intermed1;
    constexpr uint32_t cb_out_accumulate_im = tt::CB::c_intermed2;
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

                for (uint32_t k_chunk = 0; k_chunk < num_chunks; ++k_chunk) {
                    PACK(DPRINT << "COMPUTE: "  << "k_chunk=" << k_chunk << ENDL());

                    /* QK = Q_CHUNK @ K_CHUNK */
                    cb_wait_front(cb_k_in, k_chunk_tiles);
                    matmul_blocks(cb_q_in, cb_k_in, cb_qk_im, S_chunk_t, S_chunk_t, DHt, qk_num_blocks, qk_in0_num_subblocks, qk_in1_num_subblocks, in0_block_w, subblock_h, subblock_w);
                    cb_pop_front(cb_k_in, k_chunk_tiles);

                    /* QK *= SCALE */
                    cb_push_back(cb_qk_im, qk_chunk_tiles);
                    mul_block_bcast_scalar_inplace(cb_qk_im, cb_scale_in, qk_chunk_tiles);
                    cb_pop_front(cb_qk_im, qk_chunk_tiles);

                    /* OUT_IM = QK @ V_CHUNK */
                    cb_wait_front(cb_v_in, k_chunk_tiles);
                    matmul_blocks(cb_qk_im, cb_v_in, cb_out_im, S_chunk_t, DHt, S_chunk_t, out_num_blocks, out_in0_num_subblocks, out_in1_num_subblocks, in0_block_w, subblock_h, subblock_w);
                    cb_pop_front(cb_v_in, k_chunk_tiles);
                    // cb_out_im points to end of CB
                    // reset read_ptr for cb_out_im
                    cb_reserve_back(cb_out_im, out_chunk_tiles);
                    cb_push_back(cb_out_im, out_chunk_tiles);


                    /* OUT_ACC += OUT_IM */
                    if (k_chunk == 0) {
                        copy_block(cb_out_im, cb_out_accumulate_im, out_chunk_tiles);
                    } else {
                        add_block_inplace(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);
                    }

                }


                copy_block(cb_out_accumulate_im, cb_out, out_chunk_tiles);

                cb_pop_front(cb_q_in, q_chunk_tiles);

            }
        }
    }


}
}
