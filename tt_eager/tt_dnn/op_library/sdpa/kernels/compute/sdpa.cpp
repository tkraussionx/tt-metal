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

void max_block_inplace(uint32_t in0, uint32_t in1, uint32_t num_tiles) {
    // inputs come in full, outputs go out full

    constexpr uint32_t dst_reg_0 = 0;
    constexpr uint32_t dst_reg_1 = 1;
    cb_wait_front(in0, num_tiles);
    cb_wait_front(in1, num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        acquire_dst(tt::DstMode::Half);
        cb_wait_front(in0, num_tiles);
        copy_tile_to_dst_init_short(in0);
        copy_tile(in0, 0, dst_reg_0);
        copy_tile(in1, i, dst_reg_1);
        cb_pop_front(in0, 1);
        max_tile_init();
        cb_reserve_back(in0, 1);
        max_tile(dst_reg_0, dst_reg_1);
        pack_tile(dst_reg_0, in0);
        cb_push_back(in0, 1);
        release_dst(tt::DstMode::Half);
    }
}

void reduce_max_c(uint32_t in0_cb, uint32_t scale_cb, uint32_t out_cb, uint32_t rows, uint32_t cols) {
    // DeviceZoneScopedN("reduce max");
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

void reduce_sum_c(uint32_t in0_cb, uint32_t scale_cb, uint32_t out_cb, uint32_t rows, uint32_t cols) {
    // Precondition: in0_cb has rows*cols produced. in0_cb has tiles in row-major order
    // Precondition: scale_cb has 1 produced
    // Precondition: out_cb has rows free
    // Postcondition: in0_cb has rows*cols consumed
    // Precondition: scale_cb has 1 produced
    // Postcondition: out_cb has rows produced

    reduce_init_delta<false, PoolType::SUM, ReduceDim::REDUCE_ROW>(in0_cb, scale_cb, out_cb);

    // DEBUG: Does reduce_init_delta mess up matmul config? YES! Need to revert

    const uint32_t num_tiles = rows * cols;
    cb_wait_front(scale_cb, 1);
    cb_wait_front(in0_cb, num_tiles);

    constexpr uint32_t reduce_dst_idx = 0;

    for (uint32_t i = 0; i < rows; i++) {
        acquire_dst(tt::DstMode::Half);
        for (uint32_t j = 0; j < cols; j++) {
            cb_wait_front(in0_cb, 1);
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(in0_cb, scale_cb, 0, 0, reduce_dst_idx);
            cb_pop_front(in0_cb, 1);
        }

        cb_reserve_back(out_cb, 1);
        pack_tile(reduce_dst_idx, out_cb);
        cb_push_back(out_cb, 1);
        release_dst(tt::DstMode::Half);
    }

   reduce_revert_delta<ReduceDim::REDUCE_ROW>(out_cb);
}

void recip_block_inplace(uint32_t in_cb, uint32_t num_tiles) {
    // Precondition: in_cb has num_tiles produced
    // Postcondition: in_cb has num_tiles produced
    cb_wait_front(in_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        acquire_dst(tt::DstMode::Half);
        cb_wait_front(in_cb, num_tiles);
        copy_tile_to_dst_init_short(in_cb); // TODO: might move out of loop
        copy_tile(in_cb, 0, 0);
        cb_pop_front(in_cb, 1);
        recip_tile_init(); // TODO: might move out of loop
        recip_tile(0);
        cb_reserve_back(in_cb, 1);
        pack_tile(0, in_cb);
        cb_push_back(in_cb, 1);
        release_dst(tt::DstMode::Half);
    }
}

void sub_exp_block_bcast_cols_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t rows, uint32_t cols) {
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Postcondition: in0_cb has rows*cols produced
    // Postcondition: in1_cb has rows produced

    // unpack_reconfig_data_format(in0_cb, in1_cb);
    // pack_reconfig_data_format(in0_cb);
    cb_wait_front(in0_cb, rows*cols);
    cb_wait_front(in1_cb, rows);

    constexpr uint32_t dst_tiles = SUB_EXP_GRANULARITY;
    const uint32_t granularity = cols >> LOG2_SUB_EXP_GRANULARITY;
    for (uint32_t i = 0; i < rows; ++i) {
        for(uint u = 0; u < granularity; u++) {
            cb_wait_front(in0_cb, rows*cols);
            sub_bcast_cols_init_short(in0_cb, in1_cb);
            tile_regs_acquire();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                sub_tiles_bcast_cols(in0_cb, in1_cb, j, i, j);
            }
            cb_pop_front(in0_cb, dst_tiles);
            cb_reserve_back(in0_cb, dst_tiles);

            exp_tile_init(true);
            for (uint32_t i = 0; i < dst_tiles; i++) {
                exp_tile(i, true);
            }
            tile_regs_commit();

            tile_regs_wait();
            for (uint32_t j = 0; j < dst_tiles; ++j) {
                pack_tile(j, in0_cb);
            }
            tile_regs_release();
            cb_push_back(in0_cb, dst_tiles);
        }
    }
}

void mul_block_bcast_cols_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t rows, uint32_t cols) {
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Postcondition: in0_cb has rows*cols produced
    // Postcondition: in1_cb has rows consumed

    // unpack_reconfig_data_format(in0_cb, in1_cb);
    // pack_reconfig_data_format(in0_cb);
    uint32_t num_tiles = rows * cols;
    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, rows);
    mul_bcast_cols_init_short(in0_cb, in1_cb);
    for (uint32_t i = 0; i < rows; ++i) {
        for (uint32_t j = 0; j < cols; ++j) {
            acquire_dst(tt::DstMode::Half);
            cb_wait_front(in0_cb, num_tiles);
            mul_tiles_bcast_cols(in0_cb, in1_cb, 0, i, 0);
            cb_pop_front(in0_cb, 1);
            cb_reserve_back(in0_cb, 1);
            pack_tile(0, in0_cb);
            cb_push_back(in0_cb, 1);
            release_dst(tt::DstMode::Half);
        }
    }
    cb_pop_front(in1_cb, rows);
}

void mul_block_bcast_scalar_inplace(uint32_t in0_cb, uint32_t in1_scalar_cb, uint32_t num_tiles) {
    // Precondition: in0_cb has num_tiles produced
    // Precondition: in1_scalar_cb has 1 produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_scalar_cb has 1 produced
    // unpack_reconfig_data_format(in0_cb, in1_scalar_cb);
    // pack_reconfig_data_format(in0_cb);
    cb_wait_front(in1_scalar_cb, 1);
    cb_wait_front(in0_cb, num_tiles);
    mul_tiles_bcast_scalar_init_short();
    for (uint32_t i = 0; i < num_tiles; ++i) {
        // PACK(DPRINT << "COMPUTE: MUL_BCAST i: " << i << ENDL());
        acquire_dst(tt::DstMode::Half);
        // ISSUE: unpacker is not blocking
        // Might be correct because of timing
        cb_wait_front(in0_cb, num_tiles);
        // PACK(DPRINT << "COMPUTE: MUL_BCAST wait_front i: " << i << ENDL());
        mul_tiles_bcast_scalar(in0_cb, in1_scalar_cb, 0, 0, 0);
        cb_pop_front(in0_cb, 1);
        // ISSUE: packer doesn't block
        cb_reserve_back(in0_cb, 1);
        // PACK(DPRINT << "COMPUTE: MUL_BCAST reserve_back i: " << i << ENDL());
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
        cb_wait_front(in0_cb, num_tiles);
        add_tiles(in0_cb, in1_cb, 0, i, 0);
        cb_pop_front(in0_cb, 1);
        cb_reserve_back(in0_cb, 1);
        pack_tile(0, in0_cb);
        cb_push_back(in0_cb, 1);
        release_dst(tt::DstMode::Half);
    }

    cb_pop_front(in1_cb, num_tiles);
}

void mul_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles consumed

    mul_tiles_init();

    cb_wait_front(in1_cb, num_tiles);
    for (uint32_t i = 0; i < num_tiles; i++) {
        acquire_dst(tt::DstMode::Half);
        cb_wait_front(in0_cb, num_tiles);
        mul_tiles(in0_cb, in1_cb, 0, i, 0);
        cb_pop_front(in0_cb, 1);
        cb_reserve_back(in0_cb, 1);
        pack_tile(0, in0_cb);
        cb_push_back(in0_cb, 1);
        release_dst(tt::DstMode::Half);
        PACK(DPRINT << "COMPUTE: MUL i: " << i << ENDL());
    }

    cb_pop_front(in1_cb, num_tiles);
}

/*
void mul_block_bcast_scalar_inplace(uint32_t in0_cb, uint32_t in1_scalar_cb, uint32_t num_tiles) {
    // Precondition: in0_cb has num_tiles produced
    // Precondition: in1_scalar_cb has 1 produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_scalar_cb has 1 produced
    // unpack_reconfig_data_format(in0_cb, in1_scalar_cb);
    // pack_reconfig_data_format(in0_cb);
    constexpr uint32_t dst_tiles = MUL_BCAST_GRANULARITY;
    const uint32_t granularity = num_tiles >> LOG2_MUL_BCAST_GRANULARITY;
    cb_wait_front(in1_scalar_cb, 1);
    cb_wait_front(in0_cb, num_tiles);
    mul_tiles_bcast_scalar_init_short();
    for (uint32_t i = 0; i < granularity; ++i) {
        // PACK(DPRINT << "COMPUTE: MUL_BCAST i: " << i << ENDL());
        // acquire_dst(tt::DstMode::Half);

        cb_wait_front(in0_cb, num_tiles);
        tile_regs_acquire();
        for (uint32_t u = 0; u < dst_tiles; u++) {
            mul_tiles_bcast_scalar(in0_cb, in1_scalar_cb, u, 0, u);
        }
        cb_pop_front(in0_cb, dst_tiles);
        cb_reserve_back(in0_cb, dst_tiles);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t u = 0; u < dst_tiles; u++) {
            pack_tile(u, in0_cb);
        }
        tile_regs_release();
        cb_push_back(in0_cb, dst_tiles);
        // pack_tile(0, in0_cb);
        // cb_push_back(in0_cb, 1);
        // release_dst(tt::DstMode::Half);
    }
*/

void sub_exp_block(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: out_cb has num_tiles produced
    // Postcondition: in0_cb and in1_cb has num_tiles consumed

    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);

    const uint dst_tiles = STATS_GRANULARITY;
    const uint granularity = num_tiles >> LOG2_STATS_GRANULARITY;
    for (uint32_t u = 0; u < granularity; u++) {
        sub_tiles_init();
        tile_regs_acquire();
        for (uint32_t i = 0; i < dst_tiles; i++) {
            sub_tiles(in0_cb, in1_cb, (u*dst_tiles) + i, (u*dst_tiles) + i, i);
        }

        exp_tile_init(true);
        for (uint32_t i = 0; i < dst_tiles; i++) {
            exp_tile(i, true);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < dst_tiles; i++) {
            pack_tile(i, out_cb);
        }
        tile_regs_release();
        cb_push_back(out_cb, dst_tiles);
    }

    cb_pop_front(in0_cb, num_tiles);
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
    DeviceZoneScopedN("matmul_blocks");
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

    uint32_t B         = get_arg_val<uint32_t>(0);
    uint32_t NQH         = get_arg_val<uint32_t>(1);
    uint32_t NKH       = get_arg_val<uint32_t>(2);
    uint32_t St       = get_arg_val<uint32_t>(3);
    uint32_t DHt      = get_arg_val<uint32_t>(4);
    uint32_t Sq_chunk_t    = get_arg_val<uint32_t>(5);
    uint32_t q_num_chunks    = get_arg_val<uint32_t>(6);
    uint32_t Sk_chunk_t    = get_arg_val<uint32_t>(7);
    uint32_t k_num_chunks    = get_arg_val<uint32_t>(8);
    uint32_t core_id    = get_arg_val<uint32_t>(9);
    uint32_t num_cores    = get_arg_val<uint32_t>(10);
    uint32_t q_parallel_factor    = get_arg_val<uint32_t>(11);

    const uint32_t num_local_q_chunks = q_num_chunks / q_parallel_factor;
    const uint32_t local_batch = core_id / (NQH * q_parallel_factor);
    const uint32_t local_q_head = (core_id / q_parallel_factor) % NQH;
    const uint32_t local_q_chunk_start = num_local_q_chunks * (core_id % q_parallel_factor);
    const uint32_t local_q_chunk_end = local_q_chunk_start + num_local_q_chunks;

    // PACK(DPRINT all the above variables
    // PACK(DPRINT << "COMPUTE: B=" << B << ENDL());
    // PACK(DPRINT << "COMPUTE: NQH=" << NQH << ENDL());
    // PACK(DPRINT << "COMPUTE: NKH=" << NKH << ENDL());
    // PACK(DPRINT << "COMPUTE: St=" << St << ENDL());
    // PACK(DPRINT << "COMPUTE: DHt=" << DHt << ENDL());
    // PACK(DPRINT << "COMPUTE: S_chunk_t=" << S_chunk_t << ENDL());
    // PACK(DPRINT << "COMPUTE: num_chunks=" << num_chunks << ENDL());


    const uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    const uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    const uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    const uint32_t out_chunk_tiles = Sq_chunk_t * DHt;


    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(0);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(1);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(2);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(3);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(4);
    constexpr uint32_t qk_num_blocks = get_compile_time_arg_val(5);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(6);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(7);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(8);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(9);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(10);
    constexpr uint32_t out_num_blocks = get_compile_time_arg_val(11);


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

    for (uint32_t nb = 0; nb < B; ++nb) {
        if (nb != local_batch) {
            continue;
        }
        // PACK(DPRINT << "COMPUTE: "  << "nb=" << nb << ENDL());
        for (uint32_t nq = 0; nq < NQH; ++nq) {
            if (nq != local_q_head) {
                continue;
            }
            // PACK(DPRINT << "COMPUTE: "  << "nq=" << nq << ENDL());
            for (uint32_t q_chunk = local_q_chunk_start; q_chunk < local_q_chunk_end; ++q_chunk) {
                // PACK(DPRINT << "COMPUTE: "  << "q_chunk=" << q_chunk << ENDL());
                // Get Q chunk
                const uint32_t q_low_idx = q_chunk * Sq_chunk_t; // This is the sequence index of the first tile of this chunk
                const uint32_t q_high_idx = q_low_idx + Sq_chunk_t;
                cb_wait_front(cb_q_in, q_chunk_tiles);

                // loop while k_low < q_high
                for (uint32_t k_chunk = 0; (k_chunk * Sk_chunk_t) < q_high_idx; ++k_chunk) {
                    DeviceZoneScopedN("K inner loop");
                    // PACK(DPRINT << "COMPUTE: "  << "k_chunk=" << k_chunk << ENDL());
                    const uint32_t k_low_idx = k_chunk * Sk_chunk_t;
                    const uint32_t k_high_idx = k_low_idx + Sk_chunk_t;

                    /* QK = Q_CHUNK @ K_CHUNK */
                    cb_wait_front(cb_k_in, k_chunk_tiles);
                    matmul_blocks(cb_q_in, cb_k_in, cb_qk_im, Sq_chunk_t, Sk_chunk_t, DHt, qk_num_blocks, qk_in0_num_subblocks, qk_in1_num_subblocks, qk_in0_block_w, qk_subblock_h, qk_subblock_w);
                    cb_pop_front(cb_k_in, k_chunk_tiles);

                    {
                        DeviceZoneScopedN("stats 1");
                        /* QK *= SCALE */
                        cb_push_back(cb_qk_im, qk_chunk_tiles);
                        mul_block_bcast_scalar_inplace(cb_qk_im, cb_scale_in, qk_chunk_tiles);
                        cb_pop_front(cb_qk_im, qk_chunk_tiles); // TODO: Fold into following push_back

                        // Finding the diagonal is harder now that q_chunk_size and k_chunk_size can differ
                        // Q-range = [q_low, q_high)
                        // K-range = [k_low, k_high)
                        // does_overlap = not (q_low >= k_high or k_low >= q_high)
                        // TODO: Due to loop bounds, we should never have k_low >= q_high. Can simplify this conditional check
                        if (!(q_low_idx >= k_high_idx || k_low_idx >= q_high_idx)) {
                            /* QK += MASK */
                            cb_push_back(cb_qk_im, qk_chunk_tiles);
                            add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);
                            cb_pop_front(cb_qk_im, qk_chunk_tiles);
                        }


                        /* cb_cur_max = max(QK, dim=-1)*/
                        cb_push_back(cb_qk_im, qk_chunk_tiles);
                        reduce_max_c(cb_qk_im, cb_identity_scale_in, cb_cur_max, Sq_chunk_t, Sk_chunk_t);

                        if (k_chunk > 0) {
                            /* cb_cur_max = maximum(cb_prev_max, cb_cur_max) */
                            // cb_prev_max and cb_cur_max are full
                            max_block_inplace(cb_cur_max, cb_prev_max, Sq_chunk_t);
                            // cb_prev_max and cb_cur_max are full
                        }

                    }

                    {
                        DeviceZoneScopedN("exp");
                        /* QK -= cb_cur_max */
                        /* QK = exp(QK)*/
                        cb_push_back(cb_qk_im, qk_chunk_tiles);
                        sub_exp_block_bcast_cols_inplace(cb_qk_im, cb_cur_max, Sq_chunk_t, Sk_chunk_t);
                        cb_pop_front(cb_qk_im, qk_chunk_tiles);
                    }

                    {
                        DeviceZoneScopedN("stats 2");
                        /* cb_cur_sum = sum(cb_qk_im, dim=-1) */
                        cb_push_back(cb_qk_im, qk_chunk_tiles);
                        reduce_sum_c(cb_qk_im, cb_identity_scale_in, cb_cur_sum, Sq_chunk_t, Sk_chunk_t);

                        // cb_cur_sum full, cb_qk_im empty
                        // DEBUG: free cb_cur_sum
                        // cb_pop_front(cb_cur_sum, S_chunk_t);

                        if (k_chunk > 0) {
                            /* cb_exp_max_diff = torch.exp(cb_prev_max - cb_cur_max) */
                            sub_exp_block(cb_prev_max, cb_cur_max, cb_exp_max_diff, Sq_chunk_t);
                            // make cb_prev_max and cb_cur_max full again
                            cb_push_back(cb_prev_max, Sq_chunk_t);
                            cb_push_back(cb_cur_max, Sq_chunk_t);
                            /* cb_prev_sum *= cb_exp_max_diff */
                            mul_block_inplace(cb_prev_sum, cb_exp_max_diff, Sq_chunk_t);
                            // cb_prev_sum full, cb_exp_max_diff empty
                            cb_push_back(cb_exp_max_diff, Sq_chunk_t);
                            // cb_exp_max_diff full
                            /* cb_cur_sum += cb_prev_sum */
                            add_block_inplace(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
                            // cb_cur_sum full, cb_prev_sum empty
                        }

                    }


                    /* OUT_IM = QK @ V_CHUNK */
                    cb_wait_front(cb_v_in, k_chunk_tiles);
                    matmul_blocks(cb_qk_im, cb_v_in, cb_out_im, Sq_chunk_t, DHt, Sk_chunk_t, out_num_blocks, out_in0_num_subblocks, out_in1_num_subblocks, out_in0_block_w, out_subblock_h, out_subblock_w);
                    cb_pop_front(cb_v_in, k_chunk_tiles);
                    // cb_out_im points to end of CB
                    // reset read_ptr for cb_out_im
                    cb_reserve_back(cb_out_im, out_chunk_tiles);
                    cb_push_back(cb_out_im, out_chunk_tiles);


                    {
                        DeviceZoneScopedN("Finalize loop");
                        /* OUT_ACC += OUT_IM */
                        if (k_chunk == 0) {
                            copy_block(cb_out_im, cb_out_accumulate_im, out_chunk_tiles);
                        } else {
                            /* cb_out_accumulate_im *= cb_exp_max_diff */
                            mul_block_bcast_cols_inplace(cb_out_accumulate_im, cb_exp_max_diff, Sq_chunk_t, DHt);
                            // cb_exp_max_diff is now empty

                            /* cb_out_accumulate_im += cb_out_im */
                            add_block_inplace(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);
                        }



                        // Set cb_prev_sum and cb_prev_max

                        if (k_chunk > 0) {
                            // Free up prev_max
                            cb_pop_front(cb_prev_max, Sq_chunk_t);

                        }
                        // cb_cur_max is full, cb_prev_max is empty
                        copy_block(cb_cur_max, cb_prev_max, Sq_chunk_t);
                        // cb_cur_max is empty, cb_prev_max is full

                        // cb_cur_sum is full, cb_prev_sum is empty
                        copy_block(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
                        // cb_cur_sum is empty, cb_prev_sum is full
                    }


                }

                {
                    DeviceZoneScopedN("Finalize Q loop");
                    // free up cb_prev_max after K chunks
                    cb_pop_front(cb_prev_max, Sq_chunk_t);
                    cb_pop_front(cb_prev_sum, Sq_chunk_t);

                    /* cb_cur_sum = 1.0 / cb_cur_sum */
                    cb_push_back(cb_cur_sum, Sq_chunk_t);
                    recip_block_inplace(cb_cur_sum, Sq_chunk_t);
                    // cb_cur_sum is full

                    /* cb_out_accumulate_im *= cb_cur_sum */
                    mul_block_bcast_cols_inplace(cb_out_accumulate_im, cb_cur_sum, Sq_chunk_t, DHt);
                    // cb_cur_sum is empty, cb_out_accumulate_im is full

                    copy_block(cb_out_accumulate_im, cb_out, out_chunk_tiles);

                    cb_pop_front(cb_q_in, q_chunk_tiles);
                }


            }
        }
    }


}
}
