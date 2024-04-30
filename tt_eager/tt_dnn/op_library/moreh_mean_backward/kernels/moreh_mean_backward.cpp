// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"

#define ACQ_REL 0
#define TILE_REGS 1


ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

namespace NAMESPACE {
void MAIN {
    const auto num_output_tiles = get_arg_val<uint32_t>(0);
    const auto n_need_bcast = get_arg_val<uint32_t>(1);
    const auto c_need_bcast = get_arg_val<uint32_t>(2);
    const auto ht_need_bcast = get_arg_val<uint32_t>(3);
    const auto wt_need_bcast = get_arg_val<uint32_t>(4);

    constexpr auto cb_in0 = tt::CB::c_in0;  // input
    constexpr auto cb_in1 = tt::CB::c_in1;  // zero tile
    constexpr auto cb_scalar = tt::CB::c_in2;
    constexpr auto cb_out0 = tt::CB::c_out0;
    constexpr auto cb_intermed0 = tt::CB::c_intermed0;
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in1);
    cb_wait_front(cb_in1, onetile);
    for (uint32_t i = 0; i < num_output_tiles; i++) {
        #if ACQ_REL
        ACQ();
        #endif
        cb_wait_front(cb_in0, onetile);
        #if TILE_REGS
        tile_regs_acquire();
        #endif
        if (ht_need_bcast && wt_need_bcast) {
            add_bcast_scalar_init_short();
            add_tiles_bcast_scalar(cb_in1, cb_in0, 0, 0, dst0);
        } else if (ht_need_bcast) {
            add_bcast_rows_init_short();
            add_tiles_bcast_rows(cb_in1, cb_in0, 0, 0, dst0);
        } else if (wt_need_bcast) {
            add_bcast_cols_init_short();
            add_tiles_bcast_cols(cb_in1, cb_in0, 0, 0, dst0);
        } else {
            copy_tile_init();
            copy_tile(cb_in0, 0, dst0);
        }
        #if TILE_REGS
        tile_regs_commit();
        #endif

        cb_reserve_back(cb_intermed0, onetile);
        #if TILE_REGS
        tile_regs_wait();
        #endif
        pack_tile(dst0, cb_intermed0);
        #if TILE_REGS
        tile_regs_release();
        #endif
        cb_push_back(cb_intermed0, onetile);
        cb_pop_front(cb_in0, onetile);
        #if ACQ_REL
        REL();
        #endif

        // output * (1 / number_of_elements)
        #if ACQ_REL
        ACQ();
        #endif
        cb_wait_front(cb_intermed0, onetile);
        #if TILE_REGS
        tile_regs_acquire();
        #endif
        mul_tiles_bcast_scalar_init_short();
        mul_tiles_bcast<BroadcastType::SCALAR>(cb_intermed0, cb_scalar, 0, 0, 0);
        #if TILE_REGS
        tile_regs_commit();
        #endif
        cb_reserve_back(cb_out0, onetile);
        #if TILE_REGS
        tile_regs_wait();
        #endif
        pack_tile(dst0, cb_out0);
        #if TILE_REGS
        tile_regs_release();
        #endif
        cb_push_back(cb_out0, onetile);
        cb_pop_front(cb_intermed0, onetile);
        #if ACQ_REL
        REL();
        #endif

    }
    cb_pop_front(cb_in1, onetile);
}
}  // namespace NAMESPACE
