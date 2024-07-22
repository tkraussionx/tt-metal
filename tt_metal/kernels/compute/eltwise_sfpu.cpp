// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "debug/dprint.h"

#define UINT8_PACKING_ISSUE 1

namespace NAMESPACE {

void print8Bits(uint8_t n) {
    for (int i = 7; i >= 0; --i) {
        DPRINT << ((n >> i) & 1);
        if (i % 4 == 0) DPRINT <<  ' ';
    }
    DPRINT << ENDL();
}

FORCE_INLINE void print_8bits() {
    uint32_t addr = 124256;
    PACK(volatile tt_l1_ptr float* test_ptrf = reinterpret_cast<volatile tt_l1_ptr float*>(addr);
         volatile tt_l1_ptr uint8_t* test_ptru = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(addr);
         DPRINT << "Print the value of the first four elements of the uint8 output CB in binary format." << ENDL();
         for (int i = 0; i < 4; ++i) { print8Bits(test_ptru[i]); });
}
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

#if UINT8_PACKING_ISSUE
    unary_op_init_common(tt::CB::c_in0, tt::CB::c_intermed0);
#else
    unary_op_init_common(tt::CB::c_in0, tt::CB::c_out0);
#endif
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CB::c_out0, per_core_block_dim);
        for(uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            acquire_dst(tt::DstMode::Half);

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CB::c_in0, 1);
            copy_tile(tt::CB::c_in0, 0, 0);

            // Current typecast doesn't support from UINT32 to UINT8.
            // #ifdef SFPU_OP_CHAIN_0
            // SFPU_OP_CHAIN_0
            // #endif

            pack_reconfig_data_format(tt::CB::c_out0);
            pack_tile(0, tt::CB::c_out0);
            print_8bits();

            cb_pop_front(tt::CB::c_in0, 1);
            release_dst(tt::DstMode::Half);
        }
        cb_push_back(tt::CB::c_out0, per_core_block_dim);
    }

}
}
