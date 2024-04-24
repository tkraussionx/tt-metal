// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"
#include "debug/dprint.h"
#include "tools/profiler/kernel_profiler.hpp"


namespace NAMESPACE {


void reduce_max_c(uint32_t in0_cb, uint32_t scale_cb, uint32_t out_cb, uint32_t rows, uint32_t cols) {
    // //DeviceZoneScopedN("reduce max");
    // TODO: Fold in maxmium(prev_max, cur_max) ? Might require a bunch of reconfigs...

    // Precondition: in0_cb has rows*cols produced. in0_cb has tiles in row-major order
    // Precondition: scale_cb has 1 produced
    // Precondition: out_cb has rows free
    // Postcondition: in0_cb has rows*cols consumed
    // Precondition: scale_cb has 1 produced
    // Postcondition: out_cb has rows produced

    reduce_init_delta<false, PoolType::MAX, ReduceDim::REDUCE_ROW>(in0_cb, scale_cb, out_cb);

    // DEBUG: Does reduce_init_delta mess up matmul config? YES! Need to revert

    const uint32_t num_tiles = rows * cols;
    cb_wait_front(scale_cb, 1);
    cb_wait_front(in0_cb, num_tiles);
    cb_reserve_back(out_cb, rows);

    constexpr uint32_t reduce_dst_idx = 0;

    for (uint32_t i = 0; i < rows; i++) {
        acquire_dst(tt::DstMode::Half);
        for (uint32_t j = 0; j < cols; j++) {
            cb_wait_front(in0_cb, 1);
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(in0_cb, scale_cb, 0, 0, reduce_dst_idx);
            cb_pop_front(in0_cb, 1);
        }

        cb_reserve_back(out_cb, 1);
        pack_tile(reduce_dst_idx, out_cb);
        cb_push_back(out_cb, 1);
        release_dst(tt::DstMode::Half);
    }

   reduce_revert_delta<ReduceDim::REDUCE_ROW>(out_cb);
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
    //DeviceZoneScopedN("matmul_blocks");
    bool spill = num_blocks > 1;
    bool enable_reload = false;
    mm_init_short();

    uint32_t output_num_tiles = M * N;
    uint32_t out_subblock_num_tiles = subblock_h * subblock_w;

    for (uint32_t block = 0; block < num_blocks; ++block) {
        // PACK(DPRINT << "COMPUTE: "  << "block=" << block << ENDL());

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
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t St = get_compile_time_arg_val(3);
    constexpr uint32_t DHt = get_compile_time_arg_val(4);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(6);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(7);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(8);

    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(9);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(10);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(11);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(12);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(13);
    constexpr uint32_t qk_num_blocks = get_compile_time_arg_val(14);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(15);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(16);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(17);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(18);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(19);
    constexpr uint32_t out_num_blocks = get_compile_time_arg_val(20);


    const uint32_t core_id    = get_arg_val<uint32_t>(0);
    const uint32_t num_cores    = get_arg_val<uint32_t>(1);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(2);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(3);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(4);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(5);
    const uint32_t local_q_start = get_arg_val<uint32_t>(6);
    const uint32_t local_q_end = get_arg_val<uint32_t>(7);

    // constexpr uint32_t num_local_q_chunks = q_num_chunks / q_parallel_factor;
    // const uint32_t local_batch = core_id / (NQH * q_parallel_factor);
    // const uint32_t local_q_head = (core_id / q_parallel_factor) % NQH;
    // const uint32_t local_q_chunk_start = num_local_q_chunks * (core_id % q_parallel_factor);
    // const uint32_t local_q_chunk_end = local_q_chunk_start + num_local_q_chunks;

    // PACK(DPRINT all the above variables
    // PACK(DPRINT << "COMPUTE: B=" << B << ENDL());
    // PACK(DPRINT << "COMPUTE: NQH=" << NQH << ENDL());
    // PACK(DPRINT << "COMPUTE: NKH=" << NKH << ENDL());
    // PACK(DPRINT << "COMPUTE: St=" << St << ENDL());
    // PACK(DPRINT << "COMPUTE: DHt=" << DHt << ENDL());
    // PACK(DPRINT << "COMPUTE: S_chunk_t=" << S_chunk_t << ENDL());
    // PACK(DPRINT << "COMPUTE: num_chunks=" << num_chunks << ENDL());


    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * DHt;


    constexpr uint32_t DST_SIZE = 8;


    // PACK(DPRINT << "COMPUTE: q_chunk_tiles=" << q_chunk_tiles << ENDL());
    // PACK(DPRINT << "COMPUTE: k_chunk_tiles=" << k_chunk_tiles << ENDL());
    // PACK(DPRINT << "COMPUTE: qk_chunk_tiles=" << qk_chunk_tiles << ENDL());

    constexpr uint32_t cb_q_in = tt::CB::c_in0;
    constexpr uint32_t cb_k_in = tt::CB::c_in1;
    constexpr uint32_t cb_v_in = tt::CB::c_in2;
    constexpr uint32_t cb_mask_in = tt::CB::c_in3;
    constexpr uint32_t cb_scale_in = tt::CB::c_in4;
    constexpr uint32_t cb_identity_scale_in = tt::CB::c_in5;

    constexpr uint32_t cb_qk_im = tt::CB::c_intermed0;
    constexpr uint32_t cb_out_im = tt::CB::c_intermed1;
    constexpr uint32_t cb_out_accumulate_im = tt::CB::c_intermed2;
    constexpr uint32_t cb_cur_max = tt::CB::c_intermed3;
    constexpr uint32_t cb_prev_max = tt::CB::c_intermed4;
    constexpr uint32_t cb_cur_sum = tt::CB::c_intermed5;
    constexpr uint32_t cb_prev_sum = tt::CB::c_intermed6;
    constexpr uint32_t cb_exp_max_diff = tt::CB::c_intermed7;

    constexpr uint32_t cb_out = tt::CB::c_out0;


    mm_init();

    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {

        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {

            for (uint32_t q_chunk = local_q_start; q_chunk < local_q_end; ++q_chunk) {

                // Get Q chunk
                const uint32_t q_low_idx = q_chunk * Sq_chunk_t; // This is the sequence index of the first tile of this chunk
                const uint32_t q_high_idx = q_low_idx + Sq_chunk_t;
                cb_wait_front(cb_q_in, q_chunk_tiles);

                // loop while k_low < q_high
                for (uint32_t k_chunk = 0; (k_chunk * Sk_chunk_t) < q_high_idx; ++k_chunk) {

                    const uint32_t k_low_idx = k_chunk * Sk_chunk_t;
                    const uint32_t k_high_idx = k_low_idx + Sk_chunk_t;

                    /* QK = Q_CHUNK @ K_CHUNK */
                    cb_wait_front(cb_k_in, k_chunk_tiles);
                    matmul_blocks(cb_q_in, cb_k_in, cb_qk_im, Sq_chunk_t, Sk_chunk_t, DHt, qk_num_blocks, qk_in0_num_subblocks, qk_in1_num_subblocks, qk_in0_block_w, qk_subblock_h, qk_subblock_w);
                    cb_pop_front(cb_k_in, k_chunk_tiles);



                    if (!(q_low_idx >= k_high_idx || k_low_idx >= q_high_idx)) {

                        cb_wait_front(cb_mask_in, qk_chunk_tiles);
                        cb_pop_front(cb_mask_in, qk_chunk_tiles);

                    }

                    // START @reem: comment out this following block to get determinism
                    cb_push_back(cb_qk_im, qk_chunk_tiles);
                    reduce_max_c(cb_qk_im, cb_identity_scale_in, cb_cur_max, Sq_chunk_t, Sk_chunk_t);
                    // END

                    cb_pop_front(cb_cur_max, Sq_chunk_t);


                    /* OUT_IM = QK @ V_CHUNK */
                    cb_wait_front(cb_v_in, k_chunk_tiles);
                    matmul_blocks(cb_qk_im, cb_v_in, cb_out_im, Sq_chunk_t, DHt, Sk_chunk_t, out_num_blocks, out_in0_num_subblocks, out_in1_num_subblocks, out_in0_block_w, out_subblock_h, out_subblock_w);
                    cb_pop_front(cb_v_in, k_chunk_tiles);
                    // cb_out_im points to end of CB
                    // reset read_ptr for cb_out_im
                    cb_reserve_back(cb_out_im, out_chunk_tiles);
                    cb_push_back(cb_out_im, out_chunk_tiles);

                    if (k_chunk == 0) {
                        copy_block(cb_out_im, cb_out_accumulate_im, out_chunk_tiles);
                    } else {
                        cb_pop_front(cb_out_im, out_chunk_tiles);
                    }
                }

                copy_block(cb_out_accumulate_im, cb_out, out_chunk_tiles);

                cb_pop_front(cb_q_in, q_chunk_tiles);
            }
        }
    }


}
}
