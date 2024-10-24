// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpshft2_test.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);

    constexpr uint32_t cb_input = tt::CB::c_in0;
    constexpr uint32_t cb_output = tt::CB::c_out0;

    unary_op_init_common(cb_input, cb_output);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_input, 1);
        cb_reserve_back(cb_output, 1);

        tile_regs_acquire();
        copy_tile_to_dst_init_short(cb_input);
        copy_tile(cb_input, 0, 0);
        sfpshft2_test_init();
        sfpshft2_test(0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_output);
        tile_regs_release();

        cb_push_back(cb_output, 1);
        cb_pop_front(cb_input, 1);
    }
}
}  // namespace NAMESPACE
