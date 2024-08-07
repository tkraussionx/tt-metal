// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"

using std::uint32_t;

namespace NAMESPACE {
void MAIN {

    constexpr int onetile = 1;

    uint32_t Mt = get_compile_time_arg_val(0);
    uint32_t Kt = get_compile_time_arg_val(1);
    uint32_t Nt = get_compile_time_arg_val(2);
    bool transpose_b = (get_compile_time_arg_val(3) == 1);

    uint32_t cb_in0 = tt::CB::c_in0;
    uint32_t cb_in1 = tt::CB::c_in1;
    uint32_t cb_out0 = tt::CB::c_out0;

    // TODO: init matmul API
    mm_init(cb_in0, cb_in1, cb_out0, transpose_b);

    for (uint32_t mt_C = 0; mt_C < Mt; ++mt_C) // output tile of C
    for (uint32_t nt_C = 0; nt_C < Nt; ++nt_C) // output tile index of C
    {
        tile_regs_acquire();
        for (uint32_t kt = 0; kt < Kt; kt++) {
            // TODO: call matmul_tiles API
            cb_wait_front(cb_in0, onetile);
            cb_wait_front(cb_in1, onetile);

            matmul_tiles(cb_in0, cb_in1, 0, 0, 0, transpose_b);

            cb_pop_front(cb_in0, onetile);
            cb_pop_front(cb_in1, onetile);
        }
        tile_regs_commit();

        // TODO: pack output tile
        tile_regs_wait();
        cb_reserve_back(cb_out0, onetile);
        pack_tile(0, cb_out0);
        cb_push_back(cb_out0, onetile);
        tile_regs_release();
    }


}
} // NAMESPACE
