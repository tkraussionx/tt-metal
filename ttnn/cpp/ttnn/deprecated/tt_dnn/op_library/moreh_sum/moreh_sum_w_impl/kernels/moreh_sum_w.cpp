// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {
void MAIN {
    uint32_t Ht = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);
    uint32_t NC = get_compile_time_arg_val(2);
    constexpr uint32_t origin_W = get_compile_time_arg_val(3);


    auto cb_input = tt::CB::c_in0;
    constexpr auto cb_scaler = tt::CB::c_in2;
    constexpr auto cb_mask_w = tt::CB::c_in3;
    constexpr auto cb_intermed0 = tt::CB::c_intermed0;
    constexpr auto cb_masked_input = tt::CB::c_intermed1;
    constexpr auto cb_out = tt::CB::c_out0;
    constexpr uint32_t TILE_W = 32;
    constexpr bool do_mask_w = (origin_W % TILE_W) != 0;

    binary_op_init_common(cb_input, cb_scaler, cb_out);

    cb_wait_front(cb_scaler, 1);  // scaler tile from the reader

    constexpr int onetile = 1;
    int reduce_dst_idx = 0;
    const uint32_t mask_dst_idx = reduce_dst_idx + 1;

    for (uint32_t nc = 0; nc < NC; nc++) {
        for (uint32_t ht = 0; ht < Ht; ++ht) {
            cb_input = tt::CB::c_in0;
            tile_regs_acquire();
            for (uint32_t wt = 0; wt < Wt; ++wt) {
                cb_wait_front(cb_input, onetile);
#if defined FP32_DEST_ACC_EN
                unpack_reconfig_data_format(cb_input, cb_scaler);
#endif
                reduce_init_delta<false>();
                reduce_tile(cb_input, cb_scaler, 0, 0, reduce_dst_idx);
                reduce_revert_delta();
                cb_pop_front(cb_input, onetile);
            }
            tile_regs_commit();

            // pack fp32 CB to check the bit precision
            cb_reserve_back(cb_intermed0, onetile);
            tile_regs_wait();
            #if defined FP32_DEST_ACC_EN
                pack_reconfig_data_format(cb_intermed0);
            #endif
            pack_tile(reduce_dst_idx, cb_intermed0);
            print_bits("w-dim", 118112);
            tile_regs_release();
            cb_push_back(cb_intermed0, onetile);

            // unpack to DST
            tile_regs_acquire();
            cb_wait_front(cb_intermed0, onetile);
            #if defined FP32_DEST_ACC_EN
                unpack_reconfig_data_format_srca(cb_intermed0);
            #endif
            copy_tile_to_dst_init_short(cb_intermed0);
            copy_tile(cb_intermed0, 0, 0);
            tile_regs_commit();

            cb_reserve_back(cb_out, onetile);
            tile_regs_wait();
            #if defined FP32_DEST_ACC_EN
                pack_reconfig_data_format(cb_out);
            #endif
            pack_tile(reduce_dst_idx, cb_out);
            tile_regs_release();
            cb_push_back(cb_out, onetile);
        }
    }
}
}  // namespace NAMESPACE
