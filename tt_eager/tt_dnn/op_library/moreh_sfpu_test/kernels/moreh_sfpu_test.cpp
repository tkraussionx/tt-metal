// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

// #include "tt_eager/tt_dnn/kernels/compute/moreh_common.hpp"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/sfpu_test.h"
#include "debug/dprint.h"

#if SFPU_OP_TEST_CASE_2
namespace NAMESPACE {
void MAIN {

    // uint32
    constexpr auto cb_in0{tt::CB::c_in0};
    constexpr auto cb_intermed0{tt::CB::c_intermed0};

    // bfloat16
    constexpr auto cb_in1{tt::CB::c_in1};
    constexpr auto cb_out0{tt::CB::c_out0};

    unary_op_init_common(cb_in1, cb_out0);
    // copy cb_in1 (bfloat16) to dst0
    unpack_reconfig_data_format_srca(cb_in1);
    tile_regs_acquire();
    cb_wait_front(cb_in1, 1);
    copy_tile_to_dst_init_short(cb_in1);
    copy_tile(cb_in1, 0, 0);
    tile_regs_commit();

    // pack dst0 to cb_out0 (bfloat16)
    PACK((  pack_reconfig_data_format(cb_out0) ));
    tile_regs_wait();
    cb_reserve_back(cb_out0, 1);
    pack_tile(0, cb_out0);
    cb_push_back(cb_out0, 1);
    tile_regs_release();

    PACK(DPRINT << "pack tile to cb_out0(bfloat16) is done" << ENDL());

}
}  // namespace NAMESPACE
#else
namespace NAMESPACE {
void MAIN {
    constexpr auto cb_in0{tt::CB::c_in0};
    constexpr auto cb_intermed0{tt::CB::c_intermed0};
    constexpr auto cb_out0{tt::CB::c_out0};

    init_sfpu(cb_in0);

    tile_regs_acquire();
    cb_wait_front(cb_in0, 1);
    copy_tile_init();
    copy_tile(cb_in0, 0, 0);
    sfpu_test_tile_init();
    sfpu_test_tile(0, 0);
    tile_regs_commit();

    #if SFPU_OP_TEST_CASE_1
    PACK((  pack_reconfig_data_format(cb_intermed0) ));
    #endif
    tile_regs_wait();
    cb_reserve_back(cb_intermed0, 1);
    pack_tile(0, cb_intermed0);
    cb_push_back(cb_intermed0, 1);
    tile_regs_release();

    tile_regs_acquire();
    cb_wait_front(cb_intermed0, 1);
    copy_tile_init();
    copy_tile(cb_intermed0, 0, 0);

    sfpu_test_tile_init();
    sfpu_test_tile(0, 1);
    tile_regs_commit();

    PACK((  pack_reconfig_data_format(cb_out0) ));
    tile_regs_wait();
    cb_reserve_back(cb_out0, 1);
    pack_tile(0, cb_out0);
    cb_push_back(cb_out0, 1);
    tile_regs_release();

}
}  // namespace NAMESPACE
#endif
