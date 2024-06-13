// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/layernorm.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"

// SPLIT REDUCE across Cores
namespace NAMESPACE {
void MAIN {

    constexpr uint32_t is_top_row                     = get_compile_time_arg_val(0);
    constexpr uint32_t do_gamma                       = get_compile_time_arg_val(1);
    constexpr uint32_t do_beta                        = get_compile_time_arg_val(2);
    constexpr uint32_t num_blocks                     = get_compile_time_arg_val(3);
    uint32_t block_w                        = get_compile_time_arg_val(5);
    constexpr uint32_t block_h_const                  = get_compile_time_arg_val(4);
    volatile uint32_t block_h_volatile                = get_compile_time_arg_val(4);
    constexpr uint32_t subblock_w_const               = get_compile_time_arg_val(6);
    volatile uint32_t subblock_w_volatile             = get_compile_time_arg_val(6);
    uint32_t num_subblocks_w                = get_compile_time_arg_val(7);
    const bool is_allgather_worker                    = get_compile_time_arg_val(8) == 1;
    uint32_t num_tiles_per_block            = get_compile_time_arg_val(9);
    constexpr bool FLOAT32_DTYPE                    = get_compile_time_arg_val(10) == 1;

    const uint32_t block_w_arg_val = get_arg_val<uint32_t>(0);
    block_w = block_w_arg_val;

    // DPRINT_UNPACK({DPRINT << "Hello from layernorm compute_kernel!" << ENDL(); });

    // DPRINT << "LN compute kernel, block_w runtime_arg = " << block_w_arg_val << " block_w compile_time_arg: " << block_w <<  ENDL();

    const uint32_t num_tiles_per_allgather_worker = is_allgather_worker ? get_arg_val<uint32_t>(2) : 0;

    const uint32_t width_index = get_arg_val<uint32_t>(1);

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t scaler0 = 0;

    constexpr uint32_t cb_in0 = tt::CB::c_in0;
    constexpr uint32_t cb_in1 = tt::CB::c_in1;
    constexpr uint32_t cb_scaler = tt::CB::c_in2;
    constexpr uint32_t cb_eps = tt::CB::c_in3;
    constexpr uint32_t cb_scaler_global = tt::CB::c_in4;
    constexpr uint32_t cb_gamma = tt::CB::c_in5;
    constexpr uint32_t cb_beta = tt::CB::c_in6;
    constexpr uint32_t cb_x = tt::CB::c_intermed0; // x minus mean
    #if defined RMSNORM and not defined FUSE_PRE_ADD
    constexpr uint32_t cb_xmm = cb_in0; // x minus mean
    #else
    constexpr uint32_t cb_xmm = tt::CB::c_intermed1; // x minus mean
    #endif
    constexpr uint32_t cb_ex_partial = tt::CB::dataflow0; // E[x] partial reduce
    constexpr uint32_t cb_ex = tt::CB::dataflow1; // E[x] global reduce
    constexpr uint32_t cb_ex_external = tt::CB::dataflow2;
    constexpr uint32_t cb_ex_partial2 = tt::CB::dataflow3; // E[(x-E[x])^2] partial reduce
    constexpr uint32_t cb_ex2 = tt::CB::dataflow4; // E[(x-E[x])^2] global reduce
    constexpr uint32_t cb_ex_external2 = tt::CB::dataflow5;
    constexpr uint32_t cb_ex_global = tt::CB::dataflow7; // E[x] global reduce
    constexpr uint32_t cb_xmm2 = cb_x; // xmm^2
    constexpr uint32_t cb_ex2pe = tt::CB::c_intermed3; // E[(x-E[x])^2]+eps
    constexpr uint32_t cb_fusion = tt::CB::c_intermed1; // stream gamma/beta
    constexpr uint32_t cb_padding_zero = tt::CB::c_intermed2;
    constexpr uint32_t cb_out = tt::CB::c_out0;

    binary_op_init_common(cb_in0, cb_in0, cb_x);

    // set block_h to volatile to disable automatically unroll of the loops, avoid code overflow
    const uint32_t block_h = (block_w == 1) ? block_h_volatile : block_h_const;
    uint32_t subblock_w = (block_w <= 2) ? subblock_w_volatile : subblock_w_const;
    uint32_t num_valid_tiles_per_block_h = block_w;
    uint32_t num_valid_subblocks_w = num_subblocks_w;
    uint32_t padding_diff = 0;

    if (block_w == 16) {
        subblock_w = 1;
        num_subblocks_w = 18;
        block_w = 18;
        num_valid_subblocks_w = 16;
        padding_diff = num_subblocks_w - num_valid_subblocks_w;
        // num_tiles_per_block = block_w * block_h_const;
        //num_valid_tiles_per_block_h = block_w;

        cb_wait_front(cb_padding_zero, 1);

        // DPRINT << "subblock_w: " << subblock_w << " num_subblocks_w: " << num_subblocks_w << " num_tiles_per_block: " << num_tiles_per_block << ENDL();
    } else if (width_index == 7) {
        DPRINT << "width=7 subblock_w: " << subblock_w << " num_subblocks_w: " << num_subblocks_w << " num_tiles_per_block: " << num_tiles_per_block << ENDL();
    }

    // DPRINT << "subblock_w: " << subblock_w << " num_subblocks_w: " << num_subblocks_w << " num_tiles_per_block: " << num_tiles_per_block << "num_blocks:" << num_blocks << "do_gamma: " << do_gamma << "do_betta: " << do_beta << ENDL();


    // DPRINT << "In LN compute kernel, block_h: " << block_h  << ", block_w: " << block_w << ", num_subblocks_w: " << num_subblocks_w << ", subblock_w: " << subblock_w_volatile << " num_tiles_per_block: " << num_tiles_per_block << ENDL();

    int index_subblock_w_offset = 0;
    int index_h_offset = 0;
    int index = 0;

    #ifdef FUSE_PRE_ADD
    #ifdef RMSNORM
    constexpr uint32_t cb_in = cb_xmm;
    #else
    constexpr uint32_t cb_in = cb_x;
    #endif
    #else
    constexpr uint32_t cb_in = cb_in0;
    #endif
    constexpr uint32_t cb_im = (do_gamma | do_beta) ? cb_x : cb_out;
    constexpr uint32_t cb_outgamma = do_beta ? cb_fusion : cb_out;

    // pre-add x + y
    #ifdef FUSE_PRE_ADD
    unpack_reconfig_data_format_srcb(cb_in0, cb_in1);
    add_tiles_init();
    cb_reserve_back(cb_in, num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                add_tiles(cb_in0, cb_in1, index, index, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_w; i++) {
                pack_tile(i, cb_in);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_w;
        }
        index_h_offset += block_w;
    }
    cb_push_back(cb_in, num_tiles_per_block);
    #ifndef RMSNORM
    unpack_reconfig_data_format(cb_in0, cb_in, cb_in1, cb_scaler);
    #else
    unpack_reconfig_data_format(cb_in0, cb_in, cb_in1, cb_in);
    #endif
    cb_wait_front(cb_in, num_tiles_per_block);
    #else
    #ifndef RMSNORM
    unpack_reconfig_data_format_srcb(cb_in0, cb_scaler);
    #endif // RMSNORM
    #endif // FUSE_PRE_ADD

    #ifndef RMSNORM
    // E[x],
    index_h_offset = 0;
    reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
    cb_wait_front(cb_scaler, 1);
    cb_reserve_back(cb_ex_partial, block_h);
    for (uint32_t i = 0; i < block_h; i++) {
        tile_regs_acquire();
        for (uint32_t w = 0; w < block_w; w++) {
            reduce_tile(cb_in, cb_scaler, w+index_h_offset, scaler0, dst0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_ex_partial);
        tile_regs_release();
        index_h_offset += block_w;
    }
    reduce_revert_delta();
    cb_push_back(cb_ex_partial, block_h);
    cb_wait_front(cb_ex_partial, block_h);

    // constexpr uint32_t tile_index_to_inspect = 0;
    // DPRINT_UNPACK({ DPRINT  << "WidthIndex: " << width_index << " BlockW: " << block_w << "TileRow: " <<  TSLICE(cb_ex_partial, tile_index_to_inspect, SliceRange::h0_w0_32()) << ENDL(); });

    unpack_reconfig_data_format_srca(cb_in, cb_ex_external);

    // global reduce, cb_ex <-- cb_ex_external, cb_ex_partial
    if constexpr(is_allgather_worker) {
        reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
        cb_reserve_back(cb_ex, num_tiles_per_allgather_worker);

        for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {
            cb_wait_front(cb_scaler_global, 1);
            tile_regs_acquire();
            for (uint32_t w = 0; w < num_blocks; w++) {
                cb_wait_front(cb_ex_external, 1);
                //DPRINT_UNPACK({ DPRINT << TSLICE(cb_ex_external, 0, SliceRange::h0_w0_32()) << ENDL(); });
                reduce_tile(cb_ex_external, cb_scaler_global, 0, scaler0, dst0);
                cb_pop_front(cb_ex_external, 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex);
            tile_regs_release();
        }
        reduce_revert_delta();
        cb_push_back(cb_ex, num_tiles_per_allgather_worker);
        cb_wait_front(cb_ex, num_tiles_per_allgather_worker);
    }

    // x - E[x]
    if constexpr (FLOAT32_DTYPE) {
        unpack_reconfig_data_format(cb_in, cb_ex_global);
    }
    index_h_offset = 0;
    unpack_reconfig_data_format_srca(cb_ex_external, cb_in);
    cb_reserve_back(cb_xmm, num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        sub_bcast_cols_init_short();
        index_subblock_w_offset = 0;
        cb_wait_front(cb_ex_global, 1);
        constexpr uint32_t tile_index_to_inspect = 0;
        // DPRINT_UNPACK({ DPRINT  << "cb_ex_global, width_index=" << width_index << "Tile index:" << tile_index_to_inspect << " Tile content: " << TSLICE(cb_ex_global, tile_index_to_inspect, SliceRange::h0_w0_32()) << ENDL(); });
        // DPRINT_UNPACK({ DPRINT  << "checking, width_index=" << width_index << "num_subblocks_w:" << num_subblocks_w << " sublock_w: " << subblock_w << ENDL(); });

        for (uint32_t j = 0; j < num_valid_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset;
                sub_tiles_bcast_cols(cb_in, cb_ex_global, index, 0, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_w; i++) {
                pack_tile(i, cb_xmm);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_w;
        }
        copy_tile_init();
        tile_regs_acquire();
        copy_tile(cb_padding_zero, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < padding_diff; j++) {
            pack_tile(0 , cb_xmm);
        }
        tile_regs_release();

        cb_pop_front(cb_ex_global, 1);
        cb_pop_front(cb_in, block_w);
    }
    cb_push_back(cb_xmm, num_tiles_per_block);
    #ifndef FUSE_PRE_ADD
    unpack_reconfig_data_format_srca(cb_in, cb_xmm);
    #endif
    cb_wait_front(cb_xmm, num_tiles_per_block);
    //constexpr uint32_t tile_index_to_inspect = 17;
    //DPRINT_UNPACK({ DPRINT  << "WI=" << width_index << "TI:" << tile_index_to_inspect << " TC: " << TSLICE(cb_xmm, tile_index_to_inspect, SliceRange::h0_w0_32()) << ENDL(); });
    // for (uint32_t i = 0; i < num_tiles_per_block; i++) {
    //     DPRINT_UNPACK({ DPRINT  << "Xmm, width_index=" << width_index << "Tile content: " << TSLICE(cb_xmm, i, SliceRange::h0_w0_32()) << ENDL(); });
    // }
    // if (width_index == 7 && block_w == 18) {
    //     DPRINT_UNPACK({ DPRINT  << "Regular implementation: " <<  TSLICE(cb_xmm, tile_index_to_inspect, SliceRange::h0_w0_32()) << ENDL(); });
    // }
    // if (block_w == 16) {
    //     DPRINT_UNPACK({ DPRINT  << "Padded version: " <<  TSLICE(cb_xmm, tile_index_to_inspect, SliceRange::h0_w0_32()) << ENDL(); });
    // }
    #endif

    // (x - E[x])^2, cb_mm2 <-- cb_xmm
    index_h_offset = 0;
    cb_reserve_back(cb_xmm2, num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        mul_tiles_init();
        index_subblock_w_offset = 0;
        for (uint32_t j = 0; j < num_valid_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                mul_tiles(cb_xmm, cb_xmm, index, index, w);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_w; i++) {
                pack_tile(i, cb_xmm2);
            }
            tile_regs_release();
            index_subblock_w_offset += subblock_w;
        }
        copy_tile_init();
        tile_regs_acquire();
        copy_tile(cb_padding_zero, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < padding_diff; j++) {
            pack_tile(0 , cb_xmm2);
        }
        tile_regs_release();
        index_h_offset += block_w;
    }
    cb_push_back(cb_xmm2, num_tiles_per_block);

    #if defined RMSNORM and not defined FUSED_PRE_ADD
    unpack_reconfig_data_format(cb_xmm, cb_xmm2, cb_xmm, cb_scaler);
    #else
    if constexpr (FLOAT32_DTYPE) {
        unpack_reconfig_data_format(cb_xmm, cb_xmm2, cb_xmm, cb_scaler);
    }
    #endif

    cb_wait_front(cb_xmm2, num_tiles_per_block);
    // constexpr uint32_t tile_index_to_inspect = 15;
    // DPRINT_UNPACK({ DPRINT  << "XMM2_WI=" << width_index << "TI:" << tile_index_to_inspect << " TC: " << TSLICE(cb_xmm2, tile_index_to_inspect, SliceRange::h0_w0_32()) << ENDL(); });

    // Var(x)
    cb_reserve_back(cb_ex_partial2, block_h);
    reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
    index_h_offset = 0;
    for (uint32_t i = 0; i < block_h; i++) {
        tile_regs_acquire();
        for (uint32_t w = 0; w < block_w; w++) {
            reduce_tile(cb_xmm2, cb_scaler, w+index_h_offset, scaler0, dst0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_ex_partial2);
        tile_regs_release();
        index_h_offset += block_w;
    }
    reduce_revert_delta();
    cb_pop_front(cb_xmm2, num_tiles_per_block);
    cb_push_back(cb_ex_partial2, block_h);

    // constexpr uint32_t tile_index_to_inspect = 0;
    // DPRINT_UNPACK({ DPRINT  << "cb_ex_partial2_WI=" << width_index << "TI:" << tile_index_to_inspect << " TC: " << TSLICE(cb_ex_partial2, tile_index_to_inspect, SliceRange::h0_w0_32()) << ENDL(); });


    // global reduce, cb_ex <-- cb_ex_external, cb_ex_partial
    if constexpr(is_allgather_worker) {
        reduce_init_delta<false>(REDUCE_OP, REDUCE_DIM);
        cb_reserve_back(cb_ex2, num_tiles_per_allgather_worker);

        for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {
            cb_wait_front(cb_scaler_global, 1);

            tile_regs_acquire();
            for (uint32_t w = 0; w < num_blocks; w++) {
                cb_wait_front(cb_ex_external2, 1);
                reduce_tile(cb_ex_external2, cb_scaler_global, 0, scaler0, dst0);
                cb_pop_front(cb_ex_external2, 1);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex2);
            tile_regs_release();
        }
        reduce_revert_delta();
        cb_push_back(cb_ex2, num_tiles_per_allgather_worker);

        for (uint32_t i = 0; i < num_tiles_per_allgather_worker; i++) {
            // 1/[sqrt(Var + eps)],
            cb_wait_front(cb_ex2, 1);
            cb_reserve_back(cb_ex2pe, 1);
            tile_regs_acquire();
            add_tiles_init();
            add_tiles(cb_ex2, cb_eps, i, 0, dst0);
            tile_regs_wait();
            // sqrt(Var + eps)
            sqrt_tile_init();
            sqrt_tile(dst0);
            tile_regs_wait();
            // 1/[sqrt(Var + eps)]
            recip_tile_init();
            recip_tile(dst0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(dst0, cb_ex2pe);
            cb_push_back(cb_ex2pe, 1);
            tile_regs_release();
        }

    }


    if constexpr(do_gamma == 0 && do_beta == 0) {
        pack_reconfig_data_format(cb_out);
    }
    // (x - Ex) * 1/[sqrt(Var + eps)]
    #if defined RMSNORM and not defined FUSE_PRE_ADD
    if constexpr (FLOAT32_DTYPE) {
        unpack_reconfig_data_format(cb_xmm, cb_ex_global);
    } else {
        unpack_reconfig_data_format_srca(cb_ex2, cb_xmm);
    }
    #else
    if constexpr (FLOAT32_DTYPE) {
        unpack_reconfig_data_format(cb_xmm, cb_ex_global);
    }
    #endif
    index_h_offset = 0;
    cb_reserve_back(cb_im, num_tiles_per_block);
    for (uint32_t i = 0; i < block_h; i++) {
        mul_bcast_cols_init_short();
        index_subblock_w_offset = 0;
        cb_wait_front(cb_ex_global, 1);
        constexpr uint32_t tile_index_to_inspect = 0;
        DPRINT_UNPACK({ DPRINT  << "cb_ex_global_WI=" << width_index << "TI:" << tile_index_to_inspect << " TC: " << TSLICE(cb_ex_global, tile_index_to_inspect, SliceRange::h0_w0_32()) << ENDL(); });
        for (uint32_t j = 0; j < num_valid_subblocks_w; j++) {
            tile_regs_acquire();
            for (uint32_t w = 0; w < subblock_w; w++) {
                index = w + index_subblock_w_offset + index_h_offset;
                mul_tiles_bcast_cols(cb_xmm, cb_ex_global, index, 0, w);
            }
            tile_regs_commit();

            tile_regs_wait();
            for (uint32_t i = 0; i < subblock_w; i++) {
                pack_tile(i, cb_im);
            }
            tile_regs_release();

            index_subblock_w_offset += subblock_w;
        }
        copy_tile_init();
        tile_regs_acquire();
        copy_tile(cb_padding_zero, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < padding_diff; j++) {
            pack_tile(0 , cb_im);
        }
        tile_regs_release();
        index_h_offset += block_w;
        cb_pop_front(cb_ex_global, 1);
    }
    cb_push_back(cb_im, num_tiles_per_block);

    cb_pop_front(cb_xmm, num_tiles_per_block);
    cb_wait_front(cb_im, num_tiles_per_block);

    if constexpr(do_gamma) {
        unpack_reconfig_data_format(cb_im, cb_gamma);
        if constexpr(do_beta == 0) {
            pack_reconfig_data_format(cb_out);
        }
        cb_wait_front(cb_gamma, block_w);
        index_h_offset = 0;
        cb_reserve_back(cb_outgamma, num_tiles_per_block);
        for (uint32_t i = 0; i < block_h; i++) {
            index_subblock_w_offset = 0;
            for (uint32_t j = 0; j < num_valid_subblocks_w; j++) {
                mul_bcast_rows_init_short();
                tile_regs_acquire();
                for (uint32_t w = 0; w < subblock_w; w++) {
                    index = w + index_subblock_w_offset;
                    mul_tiles_bcast_rows(cb_im, cb_gamma, index+index_h_offset, index, w);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < subblock_w; i++) {
                    pack_tile(i, cb_outgamma);
                }
                tile_regs_release();
                index_subblock_w_offset += subblock_w;
            }
            copy_tile_init();
            tile_regs_acquire();
            copy_tile(cb_padding_zero, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < padding_diff; j++) {
                pack_tile(0 , cb_outgamma);
            }
            tile_regs_release();
            index_h_offset += block_w;
        }
        cb_push_back(cb_outgamma, num_tiles_per_block);
        cb_pop_front(cb_im, num_tiles_per_block);
        cb_wait_front(cb_outgamma, num_tiles_per_block);
    }

    if constexpr(do_beta) {
        unpack_reconfig_data_format(cb_fusion, cb_beta);
        pack_reconfig_data_format(cb_out);
        cb_wait_front(cb_beta, block_w);
        index_h_offset = 0;
        cb_reserve_back(cb_out, num_tiles_per_block);
        for (uint32_t i = 0; i < block_h; i++) {
            add_bcast_rows_init_short();
            index_subblock_w_offset = 0;
            for (uint32_t j = 0; j < num_valid_subblocks_w; j++) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < subblock_w; w++) {
                    index = w + index_subblock_w_offset;
                    add_tiles_bcast_rows(cb_fusion, cb_beta, index + index_h_offset, index, w);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t i = 0; i < subblock_w; i++) {
                    pack_tile(i, cb_out);
                }
                tile_regs_release();
                index_subblock_w_offset += subblock_w;
            }
            copy_tile_init();
            tile_regs_acquire();
            copy_tile(cb_padding_zero, 0, 0);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < padding_diff; j++) {
                pack_tile(0 , cb_out);
            }
            tile_regs_release();

            index_h_offset += block_w;
        }
        cb_push_back(cb_out, num_tiles_per_block);
        cb_pop_front(cb_fusion, num_tiles_per_block);
        cb_wait_front(cb_out, num_tiles_per_block);
    }

    // if (width_index == 7 && block_w == 18) {
    //     // DPRINT_UNPACK({ DPRINT  << "Regular implementation: " <<  TSLICE(cb_out, tile_index_to_inspect, SliceRange::h0_w0_32()) << ENDL(); });
    // }

    // if (block_w == 16) {
    //    // DPRINT << "BEGIN" << ENDL();
    //     uint32_t pad_remainder = 2 * block_h_const;
    //     unpack_reconfig_data_format_srca(cb_padding_zero);
    //     pack_reconfig_data_format(cb_out);
    //     // DPRINT_UNPACK({ DPRINT  << "before: " <<  TSLICE(cb_out, 17, SliceRange::h0_w0_32()) << ENDL(); });
    //     copy_tile_init();
    //     cb_wait_front(cb_padding_zero, 1);
    //     cb_reserve_back(cb_out, pad_remainder);
    //     //cb_reserve_back(cb_out, 18);
    //     tile_regs_acquire();
    //     copy_tile(cb_padding_zero, 0, 0);
    //     tile_regs_commit();
    //     tile_regs_wait();
    //     for (uint32_t i = 0; i < pad_remainder; i++) {
    //         pack_tile(0, cb_out);
    //     }
    //     tile_regs_release();
    //     cb_push_back(cb_out, pad_remainder);
    //     cb_pop_front(cb_padding_zero, 1);
    //     //DPRINT << "END" << ENDL();
    //     cb_wait_front(cb_out, block_w * block_h_const);
    //     // DPRINT_UNPACK({ DPRINT  << "Padded version: " <<  TSLICE(cb_out, tile_index_to_inspect, SliceRange::h0_w0_32()) << ENDL(); });
    // }

    // constexpr uint32_t tile_index_to_inspect = 36;
    // if (width_index == 7 && block_w == 18) {
    //     DPRINT_UNPACK({ DPRINT  << "   Padding implementation, tile_index: " << tile_index_to_inspect << " -> " <<  TSLICE(cb_out, tile_index_to_inspect, SliceRange::h0_w0_32()) << ENDL(); });
    // }
    // if (width_index == 7 && block_w == 16) {
    //     DPRINT_UNPACK({ DPRINT  << "No padding implementation, tile_index: " << tile_index_to_inspect << " -> " <<  TSLICE(cb_out, tile_index_to_inspect, SliceRange::h0_w0_32()) << ENDL(); });
    // }
}
}
