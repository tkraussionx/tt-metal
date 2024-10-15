// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "dprint.h"

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {

void MAIN {
    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_out0 = tt::CB::c_out0;
    constexpr auto cb_exps = tt::CB::c_intermed0;
    constexpr auto cb_dataflow = tt::CB::dataflow0;

    binary_op_init_common(cb_in0, cb_out0);

    DPRINT << "THIS KERNEL EXECUTED \n";

    constexpr int dst0 = 0;
    constexpr int dst1 = 1;
    constexpr uint32_t onetile = 1;

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);

    auto cb_tmp = cb_dataflow; // incorrect result
    // auto cb_tmp = cb_exps; // correct result

    // copy to temp
    cb_reserve_back(cb_tmp, onetile);
    cb_wait_front(cb_in0, onetile);

    tile_regs_acquire();
    copy_tile_init_with_dt(cb_in0);
    copy_tile(cb_in0, 0, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_tmp);
    tile_regs_release();

    cb_pop_front(cb_in0, onetile);
    cb_push_back(cb_tmp, onetile);


    // copy to output
    cb_reserve_back(cb_out0, onetile);
    cb_wait_front(cb_tmp, onetile);

    tile_regs_acquire();
    copy_tile_init_with_dt(cb_tmp);
    copy_tile(cb_tmp, 0, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile_with_dt(dst0, cb_out0);
    tile_regs_release();

    cb_pop_front(cb_tmp, onetile);
    cb_push_back(cb_out0, onetile);
}
}  // namespace NAMESPACE
