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
    // cb_wait_front(in0, num_tiles);
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
            // cb_wait_front(in0_cb, 1);
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW>(in0_cb, scale_cb, i*cols+j, 0, reduce_dst_idx);
            // cb_pop_front(in0_cb, 1);
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
            // cb_wait_front(in0_cb, 1);
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(in0_cb, scale_cb, i*cols+j, 0, reduce_dst_idx);
            // cb_pop_front(in0_cb, 1);
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
    // cb_wait_front(in_cb, num_tiles);
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

// void recip_block_inplace(uint32_t in_cb, uint32_t num_tiles) {
//     // Precondition: in_cb has num_tiles produced
//     // Postcondition: in_cb has num_tiles produced
//     // cb_wait_front(in_cb, num_tiles);
//     constexpr uint32_t dst_tiles = STATS_GRANULARITY;
//     const uint32_t granularity = num_tiles >> LOG2_STATS_GRANULARITY;

//     for (uint32_t u = 0; u < granularity; ++u) {
//         acquire_dst(tt::DstMode::Half);
//         cb_wait_front(in_cb, num_tiles);
//         copy_tile_to_dst_init_short(in_cb);
//         tile_regs_acquire();
//         for (uint32_t d = 0; d < dst_tiles; ++d) {
//             copy_tile(in_cb, d, d);
//         }

//         recip_tile_init();
//         for (uint32_t d = 0; d < dst_tiles; ++d) {
//             recip_tile(d);
//         }
//         tile_regs_commit();
//         cb_pop_front(in_cb, dst_tiles);

//         cb_reserve_back(in_cb, dst_tiles);
//         tile_regs_wait();
//         for (uint32_t d = 0; d < dst_tiles; ++d) {
//             pack_tile(d, in_cb);
//         }
//         tile_regs_release();
//         cb_push_back(in_cb, dst_tiles);
//     }
// }


void sub_exp_block_bcast_cols_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t rows, uint32_t cols) {
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Postcondition: in0_cb has rows*cols produced
    // Postcondition: in1_cb has rows produced

    // unpack_reconfig_data_format(in0_cb, in1_cb);
    // pack_reconfig_data_format(in0_cb);
    // cb_wait_front(in0_cb, rows*cols);
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

// void mul_block_bcast_cols_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t rows, uint32_t cols) {
//     // Precondition: in0_cb has rows*cols produced
//     // Precondition: in1_cb has rows produced
//     // Postcondition: in0_cb has rows*cols produced
//     // Postcondition: in1_cb has rows consumed

//     // unpack_reconfig_data_format(in0_cb, in1_cb);
//     // pack_reconfig_data_format(in0_cb);
//     constexpr uint32_t dst_tiles = DHT_GRANULARITY;
//     const uint32_t granularity = cols >> DHT_GRANULARITY;
//     uint32_t num_tiles = rows * cols;
//     // cb_wait_front(in0_cb, num_tiles); /* COMMENT THIS TO GET NON-DETERMINISM :) */
//     cb_wait_front(in1_cb, rows);
//     mul_bcast_cols_init_short(in0_cb, in1_cb);
//     for (uint32_t i = 0; i < rows; ++i) {
//         for (uint32_t j = 0; j < granularity; ++j) {
//             acquire_dst(tt::DstMode::Half);
//             cb_wait_front(in0_cb, num_tiles);
//             for (uint32_t d = 0; d < dst_tiles; ++d) {
//                 mul_tiles_bcast_cols(in0_cb, in1_cb, d, i, d);
//             }
//             cb_pop_front(in0_cb, dst_tiles);
//             cb_reserve_back(in0_cb, dst_tiles);
//             for (uint32_t d = 0; d < dst_tiles; ++d) {
//                 pack_tile(d, in0_cb);
//             }
//             cb_push_back(in0_cb, dst_tiles);
//             release_dst(tt::DstMode::Half);
//         }
//     }
//     cb_pop_front(in1_cb, rows);
// }

void mul_block_bcast_cols_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t rows, uint32_t cols) {
    // Precondition: in0_cb has rows*cols produced
    // Precondition: in1_cb has rows produced
    // Postcondition: in0_cb has rows*cols produced
    // Postcondition: in1_cb has rows consumed

    // unpack_reconfig_data_format(in0_cb, in1_cb);
    // pack_reconfig_data_format(in0_cb);
    uint32_t num_tiles = rows * cols;
    // cb_wait_front(in0_cb, num_tiles); /* COMMENT THIS TO GET NON-DETERMINISM :) */
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

/* USE THIS VERSION TO GET NON-DETERMINISM :) */
// void mul_block_bcast_scalar_inplace(uint32_t in0_cb, uint32_t in1_scalar_cb, uint32_t num_tiles) {
//     // Precondition: in0_cb has num_tiles produced
//     // Precondition: in1_scalar_cb has 1 produced
//     // Postcondition: in0_cb has num_tiles produced
//     // Postcondition: in1_scalar_cb has 1 produced
//     // unpack_reconfig_data_format(in0_cb, in1_scalar_cb);
//     // pack_reconfig_data_format(in0_cb);
//     constexpr uint32_t dst_tiles = MUL_BCAST_GRANULARITY;
//     const uint32_t granularity = num_tiles >> LOG2_MUL_BCAST_GRANULARITY;
//     cb_wait_front(in1_scalar_cb, 1);
//     // cb_wait_front(in0_cb, num_tiles);
//     mul_tiles_bcast_scalar_init_short();
//     for (uint32_t i = 0; i < granularity; ++i) {
//         // PACK(DPRINT << "COMPUTE: MUL_BCAST i: " << i << ENDL());

//         cb_wait_front(in0_cb, num_tiles);
//         tile_regs_acquire();
//         for (uint32_t u = 0; u < dst_tiles; u++) {
//             mul_tiles_bcast_scalar(in0_cb, in1_scalar_cb, u, 0, u);
//         }
//         cb_pop_front(in0_cb, dst_tiles);
//         cb_reserve_back(in0_cb, dst_tiles);
//         tile_regs_commit();
//         tile_regs_wait();
//         for (uint32_t u = 0; u < dst_tiles; u++) {
//             pack_tile(u, in0_cb);
//         }
//         tile_regs_release();
//         cb_push_back(in0_cb, dst_tiles);
//     }
// }

void mul_block_bcast_scalar_inplace(uint32_t in0_cb, uint32_t in1_scalar_cb, uint32_t num_tiles) {
    // Precondition: in0_cb has num_tiles produced
    // Precondition: in1_scalar_cb has 1 produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_scalar_cb has 1 produced
    // unpack_reconfig_data_format(in0_cb, in1_scalar_cb);
    // pack_reconfig_data_format(in0_cb);
    cb_wait_front(in1_scalar_cb, 1);
    // cb_wait_front(in0_cb, num_tiles);
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


// void add_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
//     // Precondition: in0_cb and in1_cb have num_tiles produced
//     // Postcondition: in0_cb has num_tiles produced
//     // Postcondition: in1_cb has num_tiles consumed

//     constexpr uint32_t dst_tiles = STATS_GRANULARITY;
//     const uint32_t granularity = num_tiles >> LOG2_STATS_GRANULARITY;
//     add_tiles_init();
//     // cb_wait_front(in0_cb, num_tiles);
//     cb_wait_front(in1_cb, num_tiles);
//     for (uint32_t g = 0; g < granularity; ++g) {
//         cb_wait_front(in0_cb, num_tiles);
//         tile_regs_acquire();
//         for (uint32_t i = 0; i < dst_tiles; i++) {
//             add_tiles(in0_cb, in1_cb, i, (g*dst_tiles) + i, i);
//         }
//         tile_regs_commit();
//         cb_pop_front(in0_cb, dst_tiles);
//         cb_reserve_back(in0_cb, dst_tiles);
//         tile_regs_wait();
//         for (uint32_t i = 0; i < dst_tiles; i++) {
//             pack_tile(i, in0_cb);
//         }
//         tile_regs_release();
//         cb_push_back(in0_cb, dst_tiles);
//     }

//     cb_pop_front(in1_cb, num_tiles);
// }


void add_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: in0_cb has num_tiles produced
    // Postcondition: in1_cb has num_tiles consumed

    add_tiles_init();
    // cb_wait_front(in0_cb, num_tiles);
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
    // Postcondition: in1_cb has num_tiles produced

    mul_tiles_init();
    // cb_wait_front(in0_cb, num_tiles);
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
        // PACK(DPRINT << "COMPUTE: MUL i: " << i << ENDL());
    }
}

// void sub_exp_block(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
//     // Precondition: in0_cb and in1_cb have num_tiles produced
//     // Postcondition: out_cb has num_tiles produced
//     // Postcondition: in0_cb and in1_cb has num_tiles produced

//     cb_wait_front(in0_cb, num_tiles);
//     cb_wait_front(in1_cb, num_tiles);
//     cb_reserve_back(out_cb, num_tiles);

//     const uint dst_tiles = STATS_GRANULARITY;
//     const uint granularity = num_tiles >> LOG2_STATS_GRANULARITY;
//     for (uint32_t u = 0; u < granularity; u++) {
//         sub_tiles_init();
//         tile_regs_acquire();
//         for (uint32_t i = 0; i < dst_tiles; i++) {
//             sub_tiles(in0_cb, in1_cb, (u*dst_tiles) + i, (u*dst_tiles) + i, i);
//         }

//         exp_tile_init(true);
//         for (uint32_t i = 0; i < dst_tiles; i++) {
//             exp_tile(i, true);
//         }
//         tile_regs_commit();
//         tile_regs_wait();
//         for (uint32_t i = 0; i < dst_tiles; i++) {
//             pack_tile(i, out_cb);
//         }
//         tile_regs_release();
//         cb_push_back(out_cb, dst_tiles);
//     }
// }

void sub_exp_block(uint32_t in0_cb, uint32_t in1_cb, uint32_t out_cb, uint32_t num_tiles) {
    // Precondition: in0_cb and in1_cb have num_tiles produced
    // Postcondition: out_cb has num_tiles produced
    // Postcondition: in0_cb and in1_cb has num_tiles produced

    cb_wait_front(in0_cb, num_tiles);
    cb_wait_front(in1_cb, num_tiles);
    cb_reserve_back(out_cb, num_tiles);

    for (uint32_t i = 0; i < num_tiles; i++) {
        sub_tiles_init();
        // tile_regs_acquire();
        acquire_dst(tt::DstMode::Half);

        sub_tiles(in0_cb, in1_cb, i, i, 0);

        exp_tile_init(true);
        exp_tile(0, true);

        // tile_regs_commit();
        // tile_regs_wait();

        pack_tile(0, out_cb);
        // tile_regs_release();
        cb_push_back(out_cb, 1);
        release_dst(tt::DstMode::Half);
    }
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

    // preconditino: in0_cb has M*K produced
    // preconditino: in1_cb has K*N produced
    // postcondition: out_cb has M*N produced

    mm_block_init_short(in0_cb, in1_cb, 0 /*transpose*/, subblock_w /*ct_dim*/, subblock_h /*rt_dim*/, in0_block_w /*kt_dim*/);

    uint32_t output_num_tiles = M * N;
    uint32_t out_subblock_num_tiles = subblock_h * subblock_w;
    uint32_t in0_index_offset = 0;
    uint32_t in1_index_offset = 0;

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
        for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
            tile_regs_acquire();

            int dst_index = 0;
            uint32_t in0_index = in0_index_offset;
            uint32_t in1_index = in1_index_offset;

            for (uint32_t inner_dim = 0; inner_dim < in0_block_w; inner_dim++) {
                matmul_block(in0_cb, in1_cb, in0_index, in1_index, dst_index, false, subblock_w, subblock_h, in0_block_w);
                in0_index++;
                in1_index += N;

            }
            tile_regs_commit();

            cb_reserve_back(out_cb, out_subblock_num_tiles);
            tile_regs_wait();
            for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                pack_tile(i, out_cb);
            }
            tile_regs_release();
            cb_push_back(out_cb, out_subblock_num_tiles);
            in1_index_offset += in1_subblock * subblock_w;
        }
        in0_index_offset += subblock_h * in0_block_w;
    }

    // Free up out_cb. Inner loop writes `output_num_tiles` times more than it reads.
    // cb_wait_front(out_cb, output_num_tiles);
    // cb_pop_front(out_cb, output_num_tiles);
}

void MAIN {
    constexpr uint16_t B = get_compile_time_arg_val(0);
    constexpr uint16_t NQH = get_compile_time_arg_val(1);
    constexpr uint16_t NKH = get_compile_time_arg_val(2);
    constexpr uint16_t St = get_compile_time_arg_val(3);
    constexpr uint16_t DHt = get_compile_time_arg_val(4);
    constexpr uint16_t Sq_chunk_t = get_compile_time_arg_val(5);
    constexpr uint16_t q_num_chunks = get_compile_time_arg_val(6);
    constexpr uint16_t Sk_chunk_t = get_compile_time_arg_val(7);
    constexpr uint16_t k_num_chunks = get_compile_time_arg_val(8);

    constexpr uint16_t qk_in0_block_w = get_compile_time_arg_val(9);
    constexpr uint16_t qk_subblock_w = get_compile_time_arg_val(10);
    constexpr uint16_t qk_subblock_h = get_compile_time_arg_val(11);
    constexpr uint16_t qk_in0_num_subblocks = get_compile_time_arg_val(12);
    constexpr uint16_t qk_in1_num_subblocks = get_compile_time_arg_val(13);
    constexpr uint16_t qk_num_blocks = get_compile_time_arg_val(14);
    constexpr uint16_t out_in0_block_w = get_compile_time_arg_val(15);
    constexpr uint16_t out_subblock_w = get_compile_time_arg_val(16);
    constexpr uint16_t out_subblock_h = get_compile_time_arg_val(17);
    constexpr uint16_t out_in0_num_subblocks = get_compile_time_arg_val(18);
    constexpr uint16_t out_in1_num_subblocks = get_compile_time_arg_val(19);
    constexpr uint16_t out_num_blocks = get_compile_time_arg_val(20);

    const uint32_t core_id    = get_arg_val<uint32_t>(0);
    const uint32_t num_cores    = get_arg_val<uint32_t>(1);
    const uint32_t local_batch_start = get_arg_val<uint32_t>(2);
    const uint32_t local_batch_end = get_arg_val<uint32_t>(3);
    const uint32_t local_nh_start = get_arg_val<uint32_t>(4);
    const uint32_t local_nh_end = get_arg_val<uint32_t>(5);
    const uint32_t local_q_start = get_arg_val<uint32_t>(6);
    const uint32_t local_q_end = get_arg_val<uint32_t>(7);

    const uint32_t q_chunks_per_core = local_q_end - local_q_start;


    constexpr uint16_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint16_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint16_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint16_t out_chunk_tiles = Sq_chunk_t * DHt;

    constexpr uint16_t cb_q_in = tt::CB::c_in0;
    constexpr uint16_t cb_k_in = tt::CB::c_in1;
    constexpr uint16_t cb_v_in = tt::CB::c_in2;
    constexpr uint16_t cb_mask_in = tt::CB::c_in3;
    constexpr uint16_t cb_scale_in = tt::CB::c_in4;
    constexpr uint16_t cb_identity_scale_in = tt::CB::c_in5;

    constexpr uint16_t cb_qk_im = tt::CB::c_intermed0;
    constexpr uint16_t cb_out_im = tt::CB::c_intermed1;
    constexpr uint16_t cb_out_accumulate_im = tt::CB::c_intermed2;
    constexpr uint16_t cb_cur_max = tt::CB::c_intermed3;
    constexpr uint16_t cb_prev_max = tt::CB::c_intermed4;
    constexpr uint16_t cb_cur_sum = tt::CB::c_intermed5;
    constexpr uint16_t cb_prev_sum = tt::CB::c_intermed6;
    constexpr uint16_t cb_exp_max_diff = tt::CB::c_intermed7;

    constexpr uint16_t cb_out = tt::CB::c_out0;


    mm_init();

    for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {
        // PACK(DPRINT << "COMPUTE: "  << "nb=" << nb << ENDL());
        for (uint32_t nq = local_nh_start; nq < local_nh_end; ++nq) {
            // PACK(DPRINT << "COMPUTE: "  << "nq=" << nq << ENDL());
            // for (uint32_t q_chunk = local_q_start; q_chunk < local_q_end; ++q_chunk) {
            for (uint32_t q_iter = 0; q_iter < q_chunks_per_core; ++q_iter) {
                uint32_t q_chunk;
                if (q_iter < q_chunks_per_core / 2) {
                    q_chunk = local_q_start + q_iter;
                } else {
                    uint32_t back_q_iter = q_iter - q_chunks_per_core / 2; // Back half should start at 0
                    q_chunk = q_num_chunks - 1 - (local_q_start + back_q_iter);
                }
                // PACK(DPRINT << "COMPUTE: "  << "q_chunk=" << q_chunk << ENDL());
                // Get Q chunk
                const uint16_t q_low_idx = q_chunk * Sq_chunk_t; // This is the sequence index of the first tile of this chunk
                const uint32_t q_high_idx = q_low_idx + Sq_chunk_t;
                cb_wait_front(cb_q_in, q_chunk_tiles);

                // loop while k_low < q_high
                for (uint32_t k_chunk = 0; (k_chunk * Sk_chunk_t) < q_high_idx; ++k_chunk) {
                    // DeviceZoneScopedN("K inner loop");
                    // PACK(DPRINT << "COMPUTE: "  << "k_chunk=" << k_chunk << ENDL());
                    const uint32_t k_low_idx = k_chunk * Sk_chunk_t;
                    const uint32_t k_high_idx = k_low_idx + Sk_chunk_t;

                    /* QK = Q_CHUNK @ K_CHUNK */
                    cb_wait_front(cb_k_in, k_chunk_tiles);
                    matmul_blocks(cb_q_in, cb_k_in, cb_qk_im, Sq_chunk_t, Sk_chunk_t, DHt, qk_num_blocks, qk_in0_num_subblocks, qk_in1_num_subblocks, qk_in0_block_w, qk_subblock_h, qk_subblock_w);
                    cb_pop_front(cb_k_in, k_chunk_tiles);

                    {
                        // DeviceZoneScopedN("stats 1");
                        /* QK *= SCALE */
                        // cb_push_back(cb_qk_im, qk_chunk_tiles);
                        mul_block_bcast_scalar_inplace(cb_qk_im, cb_scale_in, qk_chunk_tiles);
                        // cb_pop_front(cb_qk_im, qk_chunk_tiles); // TODO: Fold into following push_back

                        // Finding the diagonal is harder now that q_chunk_size and k_chunk_size can differ
                        // Q-range = [q_low, q_high)
                        // K-range = [k_low, k_high)
                        // does_overlap = not (q_low >= k_high or k_low >= q_high)
                        // Due to loop bounds, we should never have k_low >= q_high. Can simplify this conditional check
                        if (!(q_low_idx >= k_high_idx)) {
                            // DeviceZoneScopedN("stats 1.1");
                            /* QK += MASK */
                            // cb_push_back(cb_qk_im, qk_chunk_tiles);
                            add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);
                            // cb_pop_front(cb_qk_im, qk_chunk_tiles);
                        }


                        /* cb_cur_max = max(QK, dim=-1)*/
                        // cb_push_back(cb_qk_im, qk_chunk_tiles);
                        reduce_max_c(cb_qk_im, cb_identity_scale_in, cb_cur_max, Sq_chunk_t, Sk_chunk_t);

                        if (k_chunk > 0) {
                            /* cb_cur_max = maximum(cb_prev_max, cb_cur_max) */
                            // cb_prev_max and cb_cur_max are full
                            max_block_inplace(cb_cur_max, cb_prev_max, Sq_chunk_t);
                            // cb_prev_max and cb_cur_max are full
                        }

                    }

                    {
                        // DeviceZoneScopedN("exp");
                        /* QK -= cb_cur_max */
                        /* QK = exp(QK)*/
                        // cb_push_back(cb_qk_im, qk_chunk_tiles);
                        sub_exp_block_bcast_cols_inplace(cb_qk_im, cb_cur_max, Sq_chunk_t, Sk_chunk_t);
                        // cb_pop_front(cb_qk_im, qk_chunk_tiles);
                    }

                    {
                        // DeviceZoneScopedN("stats 2");
                        /* cb_cur_sum = sum(cb_qk_im, dim=-1) */
                        // cb_push_back(cb_qk_im, qk_chunk_tiles);
                        reduce_sum_c(cb_qk_im, cb_identity_scale_in, cb_cur_sum, Sq_chunk_t, Sk_chunk_t);

                        // cb_cur_sum full, cb_qk_im empty
                        // DEBUG: free cb_cur_sum
                        // cb_pop_front(cb_cur_sum, S_chunk_t);

                        if (k_chunk > 0) {
                            /* cb_exp_max_diff = torch.exp(cb_prev_max - cb_cur_max) */
                            sub_exp_block(cb_prev_max, cb_cur_max, cb_exp_max_diff, Sq_chunk_t);
                            cb_pop_front(cb_prev_max, Sq_chunk_t);
                            // make cb_prev_max and cb_cur_max full again
                            // cb_push_back(cb_prev_max, Sq_chunk_t);
                            // cb_push_back(cb_cur_max, Sq_chunk_t);
                            /* cb_prev_sum *= cb_exp_max_diff */
                            mul_block_inplace(cb_prev_sum, cb_exp_max_diff, Sq_chunk_t);

                            /* cb_cur_sum += cb_prev_sum */
                            add_block_inplace(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
                            // cb_cur_sum full, cb_prev_sum empty
                        }

                    }


                    /* OUT_IM = QK @ V_CHUNK */
                    cb_wait_front(cb_v_in, k_chunk_tiles);
                    matmul_blocks(cb_qk_im, cb_v_in, cb_out_im, Sq_chunk_t, DHt, Sk_chunk_t, out_num_blocks, out_in0_num_subblocks, out_in1_num_subblocks, out_in0_block_w, out_subblock_h, out_subblock_w);
                    cb_pop_front(cb_qk_im, qk_chunk_tiles);
                    cb_pop_front(cb_v_in, k_chunk_tiles);
                    // cb_out_im points to end of CB
                    // reset read_ptr for cb_out_im
                    // cb_reserve_back(cb_out_im, out_chunk_tiles);
                    // cb_push_back(cb_out_im, out_chunk_tiles);


                    {
                        // DeviceZoneScopedN("Finalize loop");
                        /* OUT_ACC += OUT_IM */
                        if (k_chunk == 0) {
                            copy_block(cb_out_im, cb_out_accumulate_im, out_chunk_tiles);
                        } else {
                            /* cb_out_accumulate_im *= cb_exp_max_diff */
                            // cb_wait_front(cb_out_accumulate_im, out_chunk_tiles); // DEBUG ND!
                            mul_block_bcast_cols_inplace(cb_out_accumulate_im, cb_exp_max_diff, Sq_chunk_t, DHt);
                            // cb_exp_max_diff is now empty

                            /* cb_out_accumulate_im += cb_out_im */
                            add_block_inplace(cb_out_accumulate_im, cb_out_im, out_chunk_tiles);
                        }



                        // Set cb_prev_sum and cb_prev_max

                        // cb_cur_max is full, cb_prev_max is empty
                        copy_block(cb_cur_max, cb_prev_max, Sq_chunk_t);
                        // cb_cur_max is empty, cb_prev_max is full

                        // cb_cur_sum is full, cb_prev_sum is empty
                        copy_block(cb_cur_sum, cb_prev_sum, Sq_chunk_t);
                        // cb_cur_sum is empty, cb_prev_sum is full
                    }


                }

                {
                    //DeviceZoneScopedN("Finalize Q loop");
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
