// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t onetile = 1;

    constexpr auto cb_y = tt::CB::c_in0;
    constexpr auto cb_dy = tt::CB::c_in1;
    constexpr auto cb_bcast_scaler = tt::CB::c_in2;
    constexpr auto cb_mask = tt::CB::c_in3;
    constexpr auto cb_dx = tt::CB::c_out0;

    constexpr auto cb_ydy = tt::CB::c_intermed0;  // y * dy
    constexpr auto cb_sum = tt::CB::c_intermed1;
    constexpr auto cb_inter2 = tt::CB::c_intermed2;
    constexpr auto cb_add = tt::CB::c_intermed3;

    constexpr int dst0 = 0;
    constexpr int dst1 = 1;
    constexpr int dst2 = 2;

    binary_op_init_common(cb_y, cb_bcast_scaler);

    uint32_t N = get_compile_time_arg_val(0);
    uint32_t Wt = get_compile_time_arg_val(1);

    cb_wait_front(cb_mask, onetile);

    for (uint32_t n = 0; n < N; ++n) {
        #ifdef LOG
            // sum(dy)
            for (uint32_t w = 0; w < Wt; ++w) {
                cb_wait_front(cb_dy, onetile);
                if (w > 0) {
                    cb_wait_front(cb_add, onetile);
                }

                tile_regs_acquire();
                copy_tile_init_with_dt(cb_dy);
                copy_tile(cb_dy, 0, dst0);
                if (w == Wt - 1) {
                    copy_tile_init_with_dt(cb_mask);
                    copy_tile(cb_mask, 0, dst1);
                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }
                if (w > 0) {
                    copy_tile_init_with_dt(cb_add);
                    copy_tile(cb_add, 0, dst2);
                }
                layernorm_acc_tile_init();
                layernorm_acc_tile(dst0, w == 0);
                tile_regs_commit();

                if (w > 0) {
                    cb_pop_front(cb_add, onetile);
                }
                cb_reserve_back(cb_add, onetile);

                tile_regs_wait();
                pack_tile_with_dt(dst2, cb_add);
                tile_regs_release();

                cb_push_back(cb_add, onetile);
                cb_pop_front(cb_dy, onetile);
            }

            cb_wait_front(cb_add, onetile);
            cb_reserve_back(cb_sum, onetile);

            tile_regs_acquire();
            copy_tile_init_with_dt(cb_add);
            copy_tile(cb_add, 0, dst0);
            layernorm_reduce_sum_w_tile_init();
            layernorm_reduce_sum_w_tile(dst0, 0x3F800000);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_sum);
            tile_regs_release();

            cb_push_back(cb_sum, onetile);
            cb_pop_front(cb_add, onetile);

            // dy - sum * exp(y)
            cb_wait_front(cb_sum, onetile);

            for (uint32_t w = 0; w < Wt; ++w) {
                cb_wait_front(cb_dy, onetile);
                cb_wait_front(cb_y, onetile);
                cb_reserve_back(cb_dx, onetile);

                tile_regs_acquire();
                copy_tile_init_with_dt(cb_dy);
                copy_tile(cb_dy, 0, dst0);
                copy_tile_init_with_dt(cb_y);
                copy_tile(cb_y, 0, dst1);
                copy_tile_init_with_dt(cb_sum);
                copy_tile(cb_sum, 0, dst2);
                exp_tile_init();
                exp_tile(dst1);
                moreh_binary_op_init();
                moreh_binary_mul(dst1);
                moreh_binary_op_init();
                moreh_binary_sub(dst0);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_dx);
                tile_regs_release();

                cb_push_back(cb_dx, onetile);
                cb_pop_front(cb_dy, onetile);
                cb_pop_front(cb_y, onetile);
            }

            cb_pop_front(cb_sum, onetile);
        #else
            // sum(y * dy)
            for (uint32_t w = 0; w < Wt; ++w) {
                cb_wait_front(cb_y, onetile);
                cb_wait_front(cb_dy, onetile);
                if (w > 0) {
                    cb_wait_front(cb_add, onetile);
                }

                tile_regs_acquire();
                copy_tile_init_with_dt(cb_y);
                copy_tile(cb_y, 0, dst0);
                copy_tile_init_with_dt(cb_dy);
                copy_tile(cb_dy, 0, dst1);
                moreh_binary_op_init();
                moreh_binary_mul(dst0);
                if (w == Wt - 1) {
                    copy_tile_init_with_dt(cb_mask);
                    copy_tile(cb_mask, 0, dst1);
                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }
                if (w > 0) {
                    copy_tile_init_with_dt(cb_add);
                    copy_tile(cb_add, 0, dst2);
                }
                layernorm_acc_tile_init();
                layernorm_acc_tile(dst0, w == 0);
                tile_regs_commit();

                if (w > 0) {
                    cb_pop_front(cb_add, onetile);
                }
                cb_reserve_back(cb_add, onetile);

                tile_regs_wait();
                pack_tile_with_dt(dst2, cb_add);
                tile_regs_release();

                cb_push_back(cb_add, onetile);
                cb_pop_front(cb_y, onetile);
                cb_pop_front(cb_dy, onetile);
            }

            cb_wait_front(cb_add, onetile);
            cb_reserve_back(cb_sum, onetile);

            tile_regs_acquire();
            copy_tile_init_with_dt(cb_add);
            copy_tile(cb_add, 0, dst0);
            layernorm_reduce_sum_w_tile_init();
            layernorm_reduce_sum_w_tile(dst0, 0x3F800000);
            tile_regs_commit();

            tile_regs_wait();
            pack_tile_with_dt(dst0, cb_sum);
            tile_regs_release();

            cb_push_back(cb_sum, onetile);
            cb_pop_front(cb_add, onetile);

            cb_wait_front(cb_sum, onetile);

            // softmax: (dy - sum) * y
            // softmin: -(dy - sum) * y
            for (uint32_t w = 0; w < Wt; ++w) {
                cb_wait_front(cb_dy, onetile);
                cb_wait_front(cb_y, onetile);
                cb_reserve_back(cb_dx, onetile);

                tile_regs_acquire();
                copy_tile_init_with_dt(cb_dy);
                copy_tile(cb_dy, 0, dst0);
                copy_tile_init_with_dt(cb_sum);
                copy_tile(cb_sum, 0, dst1);
                moreh_binary_op_init();
                moreh_binary_sub(dst0);
                copy_tile_init_with_dt(cb_y);
                copy_tile(cb_y, 0, dst1);
                moreh_binary_op_init();
                moreh_binary_mul(dst0);
                #ifndef SOFTMAX
                    negative_tile_init();
                    negative_tile(dst0);
                #endif
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_dx);
                tile_regs_release();

                cb_push_back(cb_dx, onetile);
                cb_pop_front(cb_dy, onetile);
                cb_pop_front(cb_y, onetile);
            }

            cb_pop_front(cb_sum, onetile);
        #endif
    }

    cb_pop_front(cb_mask, onetile);
}
}  // namespace NAMESPACE
