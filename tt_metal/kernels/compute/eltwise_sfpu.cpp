// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);
    const uint32_t a = per_core_block_cnt;
    const uint32_t b = per_core_block_dim;
    DPRINT << "Test string" << "\tINSIDE UNARY KERNEL: " << ENDL();
    DPRINT << "per_core_block_cnt\t" << a << ENDL();
    DPRINT << "per_core_block_dim\t" << b << ENDL();

    init_sfpu(tt::CB::c_in0);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CB::c_out0, per_core_block_dim);
        for(uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            acquire_dst(tt::DstMode::Half);

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CB::c_in0, 1);

            DPRINT << "input tile slice 14" << ENDL();
            // Extract a numpy slice from tile 0 from CB c_in0 and print it.
            DPRINT  << TSLICE(tt::CB::c_in0, 0, SliceRange::h0_w0_32()) << ENDL();

            copy_tile(tt::CB::c_in0, 0, 0);

            #ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
            #endif

            pack_tile(0, tt::CB::c_out0);

            // Extract a numpy slice from tile 0 from CB c_out0 and print it.
            DPRINT << "output tile slice 14" << ENDL();
            DPRINT  << TSLICE(tt::CB::c_out0, 0, SliceRange::h0_w0_32()) << ENDL();

            cb_pop_front(tt::CB::c_in0, 1);

            release_dst(tt::DstMode::Half);
        }
        cb_push_back(tt::CB::c_out0, per_core_block_dim);
    }

}
}
