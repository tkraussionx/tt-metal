// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/negative.h"

// #include "debug/dprint.h"

// #include "dataflow_api.h"
// #include "hostdevcommon/kernel_structs.h"
// #include "include/compute_kernel_api/eltwise_binary.h"

namespace NAMESPACE {
void MAIN {
    uint32_t mblock = get_arg_val<uint32_t>(0);
    uint32_t ublock = get_arg_val<uint32_t>(1);

    // DPRINT << "MBLOCK: " << mblock << ENDL();
    // DPRINT << "UBLOCK: " << ublock << ENDL();

    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_in1 = tt::CB::c_in1;
    constexpr auto cb_out0 = tt::CB::c_out0;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    binary_op_specific_init<false>(0);  // add, (only math init)
    for (uint32_t block = 0; block < mblock; block++) {
        // DPRINT << block << ENDL();
        cb_wait_front(cb_in0, ublock);
        cb_wait_front(cb_in1, ublock);
        // DPRINT << "Block waits done: " << block << ENDL();
        cb_reserve_back(cb_out0, ublock);
        // DPRINT << "Block reserves done: " << block << ENDL();

        tile_regs_acquire();
        for (uint32_t i = 0; i < ublock; i++) {
            add_tiles(cb_in0, cb_in1, i, i, i);
        }

        negative_tile_init();
        for (uint32_t i = 0; i < ublock; i++) {
            negative_tile(i);
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < ublock; i++) {
            pack_tile(i, cb_out0);
        }
        tile_regs_release();

        // DPRINT << "Pop front begin i: " << block << " ublock: " << ublock << ENDL();
        cb_pop_front(cb_in0, ublock);
        cb_pop_front(cb_in1, ublock);
        // DPRINT << "Pop front done i: " << block << " ublock: " << ublock << ENDL();
        cb_push_back(cb_out0, ublock);
    }
}
}  // namespace NAMESPACE
