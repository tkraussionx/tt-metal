// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/binop_with_scalar.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

float get_float_from_bits(uint32_t bits) {
    union {
        uint32_t bits;
        float f;
    } u;
    u.bits = bits;
    return u.f;
}

uint32_t get_bits(float f) {
    union {
        uint32_t bits;
        float f;
    } u;
    u.f = f;
    return u.bits;
}

namespace NAMESPACE {
void MAIN {
    uint32_t i = 0;
    uint32_t step = get_arg_val<uint32_t>(i++);

    const auto lr_bits = get_arg_val<uint32_t>(i++);
    const auto beta1_bits = get_arg_val<uint32_t>(i++);
    const auto beta2_bits = get_arg_val<uint32_t>(i++);
    const auto eps_bits = get_arg_val<uint32_t>(i++);
    const auto weight_decay_bits = get_arg_val<uint32_t>(i++);

    const auto lr = get_float_from_bits(lr_bits);
    const auto beta1 = get_float_from_bits(beta1_bits);
    const auto beta2 = get_float_from_bits(beta2_bits);
    const auto eps = get_float_from_bits(eps_bits);
    const auto weight_decay = get_float_from_bits(weight_decay_bits);

    const auto recip_bias_correction1_bits = get_arg_val<uint32_t>(i++);
    const auto recip_bias_correction2_bits = get_arg_val<uint32_t>(i++);

    // constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_tile_cnt = get_arg_val<uint32_t>(i++);

    constexpr auto cb_param_in = tt::CB::c_in0;
    constexpr auto cb_grad_in = tt::CB::c_in1;
    constexpr auto cb_exp_avg_in = tt::CB::c_in2;
    constexpr auto cb_exp_avg_sq_in = tt::CB::c_in3;
#ifdef AMSGRAD
    constexpr auto cb_max_exp_avg_sq_in = tt::CB::c_in4;
#endif
    // lr, beta1, beta2, eps, weight_decay
    constexpr auto cb_scalar_args = tt::CB::c_in5;
    constexpr auto cb_one = tt::CB::c_in6;
    constexpr auto cb_param_out = tt::CB::c_out0;
    constexpr auto cb_exp_avg_out = tt::CB::c_out1;
    constexpr auto cb_exp_avg_sq_out = tt::CB::c_out2;
#ifdef AMSGRAD
    constexpr auto cb_max_exp_avg_sq_out = tt::CB::c_out3;
#endif

    constexpr auto tmp_cb_param = tt::CB::c_intermed0;
    constexpr auto tmp_cb_exp_avg = tt::CB::c_intermed1;
    constexpr auto tmp_cb_exp_avg_sq = tt::CB::c_intermed2;
#ifdef AMSGRAD
    constexpr auto tmp_cb_max_exp_avg_sq = tt::CB::c_intermed3;
#endif

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t dst2 = 2;
    constexpr uint32_t dst3 = 3;

    constexpr uint32_t first_tile = 0;
    constexpr uint32_t onetile = 1;

    cb_wait_front(cb_scalar_args, 5);
    cb_wait_front(cb_one, onetile);

    binary_op_init_common(cb_param_in, cb_exp_avg_in, cb_param_out);

#pragma GCC nounroll
    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        cb_wait_front(cb_param_in, onetile);
        cb_wait_front(cb_grad_in, onetile);
        cb_wait_front(cb_exp_avg_in, onetile);
        cb_wait_front(cb_exp_avg_sq_in, onetile);

        // tmp_cb_param = (1-lr * weight_decay) * param_in;
        tile_regs_acquire();
        copy_tile_init_with_dt(cb_param_in);
        copy_tile(cb_param_in, first_tile, dst0);

        binop_with_scalar_tile_init();
        mul_unary_tile(dst0, get_bits(1 - lr * weight_decay));
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(tmp_cb_param, onetile);
        pack_tile_with_dt(dst0, tmp_cb_param);
        cb_push_back(tmp_cb_param, onetile);
        tile_regs_release();

        tile_regs_acquire();
        copy_tile_init_with_dt(cb_exp_avg_in);
        copy_tile(cb_exp_avg_in, first_tile, dst0);
        copy_tile_init_with_dt(cb_exp_avg_sq_in);
        copy_tile(cb_exp_avg_sq_in, first_tile, dst1);
        copy_tile_init_with_dt(cb_grad_in);
        copy_tile(cb_grad_in, first_tile, dst2);

        moreh_adamw_init();
        moreh_adamw(dst0, beta1_bits, beta2_bits, recip_bias_correction1_bits, recip_bias_correction2_bits);
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_exp_avg_out, onetile);
        cb_reserve_back(cb_exp_avg_sq_out, onetile);
        cb_reserve_back(tmp_cb_exp_avg, onetile);
        cb_reserve_back(tmp_cb_exp_avg_sq, onetile);

        pack_tile_with_dt(dst0, cb_exp_avg_out);
        pack_tile_with_dt(dst1, cb_exp_avg_sq_out);
        pack_tile_with_dt(dst2, tmp_cb_exp_avg);
        pack_tile_with_dt(dst3, tmp_cb_exp_avg_sq);

        cb_push_back(cb_exp_avg_out, onetile);
        cb_push_back(cb_exp_avg_sq_out, onetile);
        cb_push_back(tmp_cb_exp_avg, onetile);
        cb_push_back(tmp_cb_exp_avg_sq, onetile);
        tile_regs_release();

        // param_out = tmp_cb_param - lr* tmp_cb_exp_avg / (sqrt(tmp_cb_exp_avg_sq) + eps)
        tile_regs_acquire();
        cb_wait_front(tmp_cb_param, onetile);
        cb_wait_front(tmp_cb_exp_avg, onetile);
        cb_wait_front(tmp_cb_exp_avg_sq, onetile);

        copy_tile_init_with_dt(tmp_cb_param);
        copy_tile(tmp_cb_param, first_tile, dst0);
        copy_tile_init_with_dt(tmp_cb_exp_avg);
        copy_tile(tmp_cb_exp_avg, first_tile, dst1);
        copy_tile_init_with_dt(tmp_cb_exp_avg_sq);
        copy_tile(tmp_cb_exp_avg_sq, first_tile, dst2);

        binop_with_scalar_tile_init();
        mul_unary_tile(dst1, lr_bits);
        sqrt_tile_init();
        sqrt_tile(dst2);
        binop_with_scalar_tile_init();
        add_unary_tile(dst2, eps_bits);
        recip_tile_init();
        recip_tile(dst2);
        moreh_binary_op_init();
        moreh_binary_mul(dst1);
        moreh_binary_op_init();
        moreh_binary_sub(dst0);

        cb_pop_front(tmp_cb_param, onetile);
        cb_pop_front(tmp_cb_exp_avg, onetile);
        cb_pop_front(tmp_cb_exp_avg_sq, onetile);
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb_param_out, onetile);

        pack_tile_with_dt(dst0, cb_param_out);

        cb_push_back(cb_param_out, onetile);
        tile_regs_release();

        cb_pop_front(cb_param_in, onetile);
        cb_pop_front(cb_grad_in, onetile);
        cb_pop_front(cb_exp_avg_in, onetile);
        cb_pop_front(cb_exp_avg_sq_in, onetile);
#ifdef AMSGRAD
        cb_pop_front(cb_max_exp_avg_sq_in, onetile);
#endif
    }

    cb_pop_front(cb_scalar_args, 5);
    cb_pop_front(cb_one, onetile);
}  // void MAIN
}  // namespace NAMESPACE
