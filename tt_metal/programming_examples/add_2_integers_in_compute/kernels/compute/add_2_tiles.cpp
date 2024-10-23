// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/pack_untilize.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/tilize.h"
#include "debug/dprint.h"
#include "debug/dprint_tensix.h"

namespace NAMESPACE {

void MAIN {
    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_in1 = tt::CB::c_in1;
    constexpr auto cb_out0 = tt::CB::c_out0;

    // binary_op_init_common(cb_in0, cb_in1, cb_out0);
    // add_tiles_init();

    // reduce_init<true>(cb_in0, cb_in1, cb_out0);
    constexpr uint32_t num_output_tiles = 1;
    constexpr uint32_t num_out_rows = 1;
    constexpr uint32_t num_faces_in_tile = 2;
    constexpr uint32_t window_hw = 16;
    tilizeA_B_reduce_init<false, true>(
        cb_in0, cb_in1, /*block=*/2, cb_out0, /*num_faces=*/num_faces_in_tile, /*face_r_dim=*/window_hw);
    // pack_untilize_dst_init_short<num_output_tiles>(cb_out0, num_out_rows, num_faces_in_tile);

    // wait for a block of tiles in each of input CBs
    cb_wait_front(cb_in0, 1);
    cb_wait_front(cb_in1, 1);

    // cb_reserve_back(cb_out0, 1);

    tile_regs_acquire();  // acquire 8 tile registers

    // add_tiles(cb_in0, cb_in1, 0, 0, 0);
    // copy_tile(cb_in0, 0, 0);

    // reduce_tile(cb_in0, cb_in1, 0, 0, cb_out0);

    unpack_tilizeA_B_block(
        cb_in0,
        cb_in1,
        /*block=*/2,
        /*tile_idx_b=*/0,
        /*num_faces=*/num_faces_in_tile,
        /*srca_face_r_dim =*/window_hw);
    reduce_tile_math(0, /*num_faces=*/num_faces_in_tile);
    reduce_tile_math(1, /*num_faces=*/num_faces_in_tile);
    dprint_tensix_dest_reg(0);
    dprint_tensix_dest_reg(1);

    tile_regs_commit();  // signal the packer

    tile_regs_wait();  // packer waits here
    pack_tile(0, cb_out0);
    // pack_untilize_dst<num_output_tiles>(cb_out0, 1 /*out_subblock_h*/, 0, num_out_rows, num_faces_in_tile);
    tile_regs_release();  // packer releases

    cb_pop_front(cb_in0, 1);
    cb_pop_front(cb_in1, 1);

    cb_push_back(cb_out0, 1);

    /*
    acquire_dst();

    cb_wait_front(tt::CB::c_in0, 1);
    cb_wait_front(tt::CB::c_in1, 1);

    add_tiles(tt::CB::c_in0, tt::CB::c_in1, 0, 0, 0);

    cb_pop_front(tt::CB::c_in0, 1);
    cb_pop_front(tt::CB::c_in1, 1);

    cb_reserve_back(tt::CB::c_out0, 1);
    pack_tile(0, tt::CB::c_out0);
    cb_push_back(tt::CB::c_out0, 1);

    release_dst();
    */
}
}  // namespace NAMESPACE
