// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "dprint.h"


inline void RISC_POST_STATUS(uint32_t status) {
    volatile uint32_t *ptr = (volatile uint32_t *)(0xFFB2010C);
    ptr[0] = status;
}

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    init_sfpu(tt::CB::c_in0);
    uint32_t ct = 0;
    RISC_POST_STATUS((0xa << 12) | (per_core_block_dim << 8) | (per_core_block_cnt << 4) | ct);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        // DPRINT << per_core_block_cnt << ENDL();
        cb_reserve_back(tt::CB::c_out0, per_core_block_dim);
        for(uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            acquire_dst(tt::DstMode::Half);

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CB::c_in0, 1);
            RISC_POST_STATUS((0xb << 12) | (per_core_block_dim << 8) | (per_core_block_cnt << 4) | ct);
            copy_tile(tt::CB::c_in0, 0, 0);
            RISC_POST_STATUS((0xc << 12) | (per_core_block_dim << 8) | (per_core_block_cnt << 4) | ct);
            #ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
            #endif
            RISC_POST_STATUS((0xd << 12) | (tt::CB::c_out0 << 8) | (tt::CB::c_out0 << 4) | ct);
            pack_tile(0, tt::CB::c_out0);

            RISC_POST_STATUS((0xe << 12) | (tt::CB::c_out0 << 8) | (tt::CB::c_out0 << 4) | ct);
            cb_pop_front(tt::CB::c_in0, 1);

            release_dst(tt::DstMode::Half);
            ct++;
        }
        cb_push_back(tt::CB::c_out0, per_core_block_dim);
    }
    RISC_POST_STATUS((0xf << 12) | (per_core_block_dim << 8) | (per_core_block_cnt << 4) | ct);
}
}
