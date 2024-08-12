// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "dprint.h"


inline void RISC_POST_STATUS(uint32_t status, uint32_t addr = 0xFFB2010C) {
    volatile uint32_t *ptr = (volatile uint32_t *)(addr);
    ptr[0] = status;
}

UNPACK(uint32_t addr = PRINT_BUFFER_START + 12;)
MATH(uint32_t addr = PRINT_BUFFER_START + 16;)
PACK(uint32_t addr = PRINT_BUFFER_START + 20;)

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    init_sfpu(tt::CB::c_in0);
    uint32_t ct = 0;
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        // RISC_POST_STATUS((0xff << 24) | (per_core_block_dim << 16) | (per_core_block_cnt << 8) | ct, PRINT_BUFFER_START);
        cb_reserve_back(tt::CB::c_out0, per_core_block_dim);
        for(uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            ct++;
            RISC_POST_STATUS((0xee << 24) | (per_core_block_dim << 16) | (per_core_block_cnt << 8) | ct, addr);
            acquire_dst(tt::DstMode::Half);
            RISC_POST_STATUS((0xff << 24) | (per_core_block_dim << 16) | (per_core_block_cnt << 8) | ct, addr);
            // if (ct > 4) {
            //     RISC_POST_STATUS(ct, addr);
            //     break;
            // }
            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CB::c_in0, 1);
            RISC_POST_STATUS((0x10 << 24) | (per_core_block_dim << 16) | (per_core_block_cnt << 8) | ct, addr);
            RISC_POST_STATUS((0x13 << 24) | (per_core_block_dim << 16) | (per_core_block_cnt << 8) | ct, addr);
            copy_tile(tt::CB::c_in0, 0, 0);
            #ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
            #endif
            RISC_POST_STATUS((0xa << 24) | (per_core_block_dim << 16) | (per_core_block_cnt << 8) | ct, addr);
            pack_tile(0, tt::CB::c_out0);
            RISC_POST_STATUS((0xb << 24) | (per_core_block_dim << 16) | (per_core_block_cnt << 8) | ct, addr);
            cb_pop_front(tt::CB::c_in0, 1);
            RISC_POST_STATUS((0x11 << 24) | (per_core_block_dim << 16) | (per_core_block_cnt << 8) | ct, addr);

            release_dst(tt::DstMode::Half);
            RISC_POST_STATUS((0x12 << 24) | (per_core_block_dim << 16) | (per_core_block_cnt << 8) | ct, addr);
        }
        PACK(RISC_POST_STATUS((0xc << 24) | (per_core_block_dim << 16) | (per_core_block_cnt << 8) | ct, addr));
        cb_push_back(tt::CB::c_out0, per_core_block_dim);
        RISC_POST_STATUS((0xd << 24) | (per_core_block_dim << 16) | (per_core_block_cnt << 8) | ct, addr);
    }
    RISC_POST_STATUS(0xdddd0000 | ct, addr);
}
}
