// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/pack_untilize.h"

// #include "debug/dprint_tensix.h"
//#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {

    constexpr uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    constexpr uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    constexpr uint32_t num_faces = get_compile_time_arg_val(2);
    constexpr uint32_t num_rows_per_face = get_compile_time_arg_val(3);
    constexpr uint32_t num_output_tiles = 1;
    tilizeA_B_reduce_init(tt::CB::c_in0, tt::CB::c_in1, per_core_block_tile_cnt, tt::CB::c_out0, num_faces, num_rows_per_face);
    pack_untilize_dst_init_short<num_output_tiles>(tt::CB::c_out0, 1, num_faces); /* pack 1 row (1x16 or 1x32) */
    cb_wait_front(tt::CB::c_in1, 1);

    for(uint32_t b=0;b<per_core_block_cnt;++b)
    {
        cb_wait_front(tt::CB::c_in0, per_core_block_tile_cnt);
        cb_reserve_back(tt::CB::c_out0, per_core_block_tile_cnt);
        unpack_tilizeA_B_block(tt::CB::c_in0, tt::CB::c_in1, per_core_block_tile_cnt, 0 /*tile idx for Src b is 0 because only 1 tile of constants is loaded*/, num_faces, num_rows_per_face);

        for(uint i=0; i<per_core_block_tile_cnt; ++i) {
            acquire_dst(tt::DstMode::Half);
            reduce_tile_math(0,  num_faces /* reduce 1 or 2 faces */);
            // dprint_tensix_dest_reg(0);
            pack_untilize_dst<num_output_tiles>(tt::CB::c_out0, 1/*out_subblock_h*/, 0, num_output_tiles, num_faces);  /* pack 1 row (1x16 or 1x32) */
            release_dst(tt::DstMode::Half);
        }

        cb_push_back(tt::CB::c_out0, per_core_block_tile_cnt);
        cb_pop_front(tt::CB::c_in0, per_core_block_tile_cnt);
    }
    cb_pop_front(tt::CB::c_in1, 1);
}
}
