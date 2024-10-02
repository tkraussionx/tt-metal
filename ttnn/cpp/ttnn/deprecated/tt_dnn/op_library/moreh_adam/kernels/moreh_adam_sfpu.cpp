// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

#ifdef FP32_DEST_ACC_EN
#define WITH_FP32_DEST_ACC(x) x
#else
#define WITH_FP32_DEST_ACC(x)
#endif

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
    int i = 0;
    const auto step = get_arg_val<uint32_t>(i++);
    const auto lr_bits = get_arg_val<uint32_t>(i++);
    const auto beta1_bits = get_arg_val<uint32_t>(i++);
    const auto beta2_bits = get_arg_val<uint32_t>(i++);
    const auto eps_bits = get_arg_val<uint32_t>(i++);
    const auto weight_decay_bits = get_arg_val<uint32_t>(i++);
    const auto bias_correction1_bits = get_arg_val<uint32_t>(i++);
    const auto bias_correction2_bits = get_arg_val<uint32_t>(i++);

    const auto lr = get_float_from_bits(lr_bits);
    const auto beta1 = get_float_from_bits(beta1_bits);
    const auto beta2 = get_float_from_bits(beta2_bits);
    const auto eps = get_float_from_bits(eps_bits);
    const auto weight_decay = get_float_from_bits(weight_decay_bits);
    const auto bias_correction1 = get_float_from_bits(bias_correction1_bits);
    const auto bias_correction2 = get_float_from_bits(bias_correction2_bits);

    constexpr uint32_t per_core_tile_cnt = get_compile_time_arg_val(0);

    constexpr auto cb_param_in = tt::CB::c_in0;
    constexpr auto cb_grad_in = tt::CB::c_in1;
    constexpr auto cb_exp_avg_in = tt::CB::c_in2;
    constexpr auto cb_exp_avg_sq_in = tt::CB::c_in3;
#ifdef AMSGRAD
    constexpr auto cb_max_exp_avg_sq_in = tt::CB::c_in4;
#endif
    constexpr auto cb_param_out = tt::CB::c_out0;
    constexpr auto cb_exp_avg_out = tt::CB::c_out1;
    constexpr auto cb_exp_avg_sq_out = tt::CB::c_out2;
#ifdef AMSGRAD
    constexpr auto cb_max_exp_avg_sq_out = tt::CB::c_out3;
#endif

    constexpr auto tmp_cb_grad = tt::CB::c_intermed0;
    constexpr auto tmp_cb_exp_avg = tt::CB::c_intermed1;
    constexpr auto tmp_cb_exp_avg_sq = tt::CB::c_intermed2;
#ifdef AMSGRAD
    constexpr auto tmp_cb_max_exp_avg_sq = tt::CB::c_intermed3;
#endif
    constexpr auto cb_tmp1 = tt::CB::c_intermed6;
    constexpr auto cb_tmp2 = tt::CB::c_intermed7;

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t dst2 = 2;
    constexpr uint32_t dst3 = 3;

    constexpr uint32_t first_tile = 0;
    constexpr uint32_t onetile = 1;

    binary_op_init_common(cb_param_in, tmp_cb_grad);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        cb_wait_front(cb_param_in, onetile);
        cb_wait_front(cb_grad_in, onetile);
        cb_wait_front(cb_exp_avg_in, onetile);
        cb_wait_front(cb_exp_avg_sq_in, onetile);
#ifdef AMSGRAD
        cb_wait_front(cb_max_exp_avg_sq_in, onetile);
#endif

        // grad += grad + param * weight_decay;
        // tmp_cb_grad : cb_grad_in + cb_param_in * weight_decay;
        tile_regs_acquire();
        cb_reserve_back(tmp_cb_grad, onetile);
        copy_tile_init_with_dt(cb_grad_in);
        copy_tile(cb_grad_in, first_tile, dst0);
        copy_tile_init_with_dt(cb_param_in);
        copy_tile(cb_param_in, first_tile, dst1);
        binop_with_scalar_tile_init();
        mul_unary_tile(dst1, weight_decay_bits);
        moreh_binary_op_init();
        moreh_binary_add(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, tmp_cb_grad);
        cb_push_back(tmp_cb_grad, onetile);
        tile_regs_release();

        ////////////////////////////////////////////////////////////////////////
        // exp_avg = exp_avg * beta1 + grad * (1 - beta1);
        // tmp_cb_exp_avg = cb_exp_avg_out = cb_exp_avg_in * beta1 + tmp_cb_grad * (1 - beta1);
        tile_regs_acquire();
        cb_wait_front(tmp_cb_grad, onetile);
        cb_reserve_back(tmp_cb_exp_avg, onetile);
        cb_reserve_back(cb_exp_avg_out, onetile);
        copy_tile_init_with_dt(cb_exp_avg_in);
        copy_tile(cb_exp_avg_in, first_tile, dst0);
        copy_tile_init_with_dt(tmp_cb_grad);
        copy_tile(tmp_cb_grad, first_tile, dst1);
        binop_with_scalar_tile_init();
        mul_unary_tile(dst0, beta1_bits);
        binop_with_scalar_tile_init();
        mul_unary_tile(dst1, get_bits(1 - beta1));
        moreh_binary_op_init();
        moreh_binary_add(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, tmp_cb_exp_avg);
        pack_tile_with_dt(dst0, cb_exp_avg_out);
        cb_push_back(tmp_cb_exp_avg, onetile);
        cb_push_back(cb_exp_avg_out, onetile);
        tile_regs_release();
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // exp_avg_sq = exp_avg_sq * beta2 + grad * grad * (1 - beta2);
        // tmp_cb_exp_avg_sq = cb_exp_avg_sq_out = cb_exp_avg_sq_in * beta2 + tmp_cb_grad * tmp_cb_grad * (1 - beta2);
        tile_regs_acquire();
        cb_reserve_back(tmp_cb_exp_avg_sq, onetile);
        cb_reserve_back(cb_exp_avg_sq_out, onetile);
        copy_tile_init_with_dt(cb_exp_avg_sq_in, onetile);
        copy_tile(cb_exp_avg_sq_in, first_tile, dst0);
        copy_tile_init_with_dt(tmp_cb_grad, onetile);
        copy_tile(tmp_cb_grad, first_tile, dst1);
        copy_tile_init_with_dt(tmp_cb_grad, onetile);
        copy_tile(tmp_cb_grad, first_tile, dst2);
        binop_with_scalar_tile_init();
        mul_unary_tile(dst0, beta2_bits);
        moreh_binary_op_init();
        moreh_binary_mul(dst1);
        binop_with_scalar_tile_init();
        mul_unary_tile(dst1, get_bits(1.0f - beta2));
        moreh_binary_op_init();
        moreh_binary_add(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, tmp_cb_exp_avg_sq);
        pack_tile_with_dt(dst0, cb_exp_avg_sq_out);
        cb_push_back(tmp_cb_exp_avg_sq, onetile);
        cb_push_back(cb_exp_avg_sq_out, onetile);
        cb_pop_front(tmp_cb_grad, onetile);
        cb_wait_front(tmp_cb_exp_avg_sq, onetile);
        tile_regs_release();
        //////////////////////////////////////////////////////////////////////

        ////////////////////////////////////////////////////////////////////////
        // denom = sqrt(max_exp_avg_sq) / sqrt(bias_correction2) + eps;
        // denom = sqrt(exp_avg_sq) / sqrt(bias_correction2) + eps;
        // bias_correction2 = 1 - pow(beta2, step);
        const float recip_bias_correction2 = 1.0f / bias_correction2;

#ifdef AMSGRAD
        // tmp_cb_max_exp_avg_sq = max(cb_max_exp_avg_sq_in, tmp_cb_exp_avg_sq);
        tile_regs_acquire();
        cb_reserve_back(tmp_cb_max_exp_avg_sq, onetile);
        cb_reserve_back(cb_max_exp_avg_sq_out, onetile);
        copy_tile_init_with_dt(cb_max_exp_avg_sq_in);
        copy_tile(cb_max_exp_avg_sq_in, first_tile, dst0);
        copy_tile_init_with_dt(tmp_cb_exp_avg_sq);
        copy_tile(tmp_cb_exp_avg_sq, first_tile, dst1);
        max_tile_init();
        max_tile(dst0, dst1);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, tmp_cb_max_exp_avg_sq);
        pack_tile_with_dt(dst0, cb_max_exp_avg_sq_out);
        cb_push_back(tmp_cb_max_exp_avg_sq, onetile);
        cb_push_back(cb_max_exp_avg_sq_out, onetile);
        cb_wait_front(tmp_cb_max_exp_avg_sq, onetile);
        tile_regs_release();
#endif

        // cb_tmp1 = 1 / (sqrt(exp_avg_sq * recip_bias_correction2) + eps);
        // cb_tmp1 = 1 / (sqrt(max_exp_avg_sq * recip_bias_correction2) + eps);
        tile_regs_acquire();
        cb_reserve_back(cb_tmp1, onetile);

#ifdef AMSGRAD
        copy_tile_init_with_dt(tmp_cb_max_exp_avg_sq);
        copy_tile(tmp_cb_max_exp_avg_sq, first_tile, dst0);
        binop_with_scalar_tile_init();
        mul_unary_tile(dst0, get_bits(recip_bias_correction2));
#else
        copy_tile_init_with_dt(tmp_cb_exp_avg_sq);
        copy_tile(tmp_cb_exp_avg_sq, first_tile, dst0);
        binop_with_scalar_tile_init();
        mul_unary_tile(dst0, get_bits(recip_bias_correction2));
#endif
        sqrt_tile_init();
        sqrt_tile(dst0);
        binop_with_scalar_tile_init();
        add_unary_tile(dst0, eps_bits);
        recip_tile_init();
        recip_tile(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_tmp1);
        cb_push_back(cb_tmp1, onetile);
#ifdef AMSGRAD
        cb_pop_front(tmp_cb_max_exp_avg_sq, onetile);
#endif
        cb_pop_front(tmp_cb_exp_avg_sq, onetile);
        tile_regs_release();

        // bias_correction1 = 1 - pow(beta1, step);
        const float recip_bias_correction1 = 1.0f / bias_correction1;

        // param = param - cb_tmp1 * lr * recip_bias_correction1 * tmp_cb_exp_avg;
        tile_regs_acquire();
        cb_wait_front(cb_tmp1, onetile);
        cb_wait_front(tmp_cb_exp_avg, onetile);
        cb_reserve_back(cb_param_out, onetile);
        copy_tile_init_with_dt(cb_param_in);
        copy_tile(cb_param_in, first_tile, dst0);
        copy_tile_init_with_dt(cb_tmp1);
        copy_tile(cb_tmp1, first_tile, dst1);
        copy_tile_init_with_dt(tmp_cb_exp_avg);
        copy_tile(tmp_cb_exp_avg, first_tile, dst2);
        moreh_binary_op_init();
        moreh_binary_mul(dst1);
        binop_with_scalar_tile_init();
        mul_unary_tile(dst1, get_bits(lr * recip_bias_correction1));
        moreh_binary_op_init();
        moreh_binary_sub(dst0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_param_out);
        cb_push_back(cb_param_out, onetile);
        cb_pop_front(cb_tmp1, onetile);
        cb_pop_front(tmp_cb_exp_avg, onetile);
        tile_regs_release();

        cb_pop_front(cb_param_in, onetile);
        cb_pop_front(cb_grad_in, onetile);
        cb_pop_front(cb_exp_avg_in, onetile);
        cb_pop_front(cb_exp_avg_sq_in, onetile);
#ifdef AMSGRAD
        cb_pop_front(cb_max_exp_avg_sq_in, onetile);
#endif
    }
}  // void MAIN
}  // namespace NAMESPACE
