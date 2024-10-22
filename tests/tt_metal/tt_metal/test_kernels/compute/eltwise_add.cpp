// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t onetile = 1;
    constexpr uint32_t cb_input = tt::CB::c_in0;
    constexpr uint32_t cb_other = tt::CB::c_in1;
    constexpr uint32_t cb_output = tt::CB::c_out0;

    binary_op_init_common(cb_input, cb_other, cb_output);

    // push dummy tile in each CB
    cb_reserve_back(cb_input, onetile);
    cb_reserve_back(cb_other, onetile);
    cb_push_back(cb_input, onetile);
    cb_push_back(cb_other, onetile);
    cb_wait_front(cb_input, onetile);
    cb_wait_front(cb_other, onetile);
    cb_reserve_back(cb_output, onetile);

    for (uint32_t i = 0; i < num_tiles; i++) {
        // NO PACK instruction
        tile_regs_acquire();
        
        // NO PACK instruction
        binary_op_specific_init<false, EltwiseBinaryType::ELWADD>();

        // NO PACK instruction
        add_tiles(cb_input, cb_other, 0, 0, 0);

        // NO PACK instruction
        tile_regs_commit();

        // Only PACK
        tile_regs_wait();

        // Only PACK
        pack_tile(0, cb_output);

        // Only PACK
        tile_regs_release();
    }

    cb_push_back(cb_output, onetile);
    cb_pop_front(cb_input, onetile);
    cb_pop_front(cb_other, onetile);
}
}  // namespace NAMESPACE
