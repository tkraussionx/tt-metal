// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t cb_weights = get_compile_time_arg_val(0);
    constexpr uint32_t cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);

    unary_op_init_common(cb_weights);

    acquire_dst(tt::DstMode::Half);

    cb_wait_front(cb_weights, 1);
    cb_wait_front(cb_index, 1);

    cb_reserve_back(cb_out, 1);
    copy_tile(cb_weights, 0, 0);

    pack_tile(0, cb_out);

    cb_pop_front(cb_weights, 1);
    cb_pop_front(cb_index, 1);
    cb_push_back(cb_out, 1);

    release_dst(tt::DstMode::Half);
}
}  // namespace NAMESPACE
