// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

enum class MorehBinary {
    ADD = 0,
    SUB = 1,
    MUL = 2,
};  // BINOP_MODE

template <bool APPROXIMATION_MODE, MorehBinary op, int ITERATIONS = 8>
inline void moreh_binary() {
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat lhs = dst_reg[0];
        vFloat rhs = dst_reg[32];
        vFloat res = 0.0f;

        if constexpr (op == MorehBinary::ADD) {
            res = lhs + rhs;
        } else if (op == MorehBinary::SUB) {
            res = lhs - rhs;
        } else if (op == MorehBinary::MUL) {
            res = lhs * rhs;
        }

        vFloat bf16 = reinterpret<vFloat>(float_to_fp16b(res, 0));
        dst_reg[0] = bf16;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void moreh_binary_add() {
    moreh_binary<APPROXIMATION_MODE, MorehBinary::ADD, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void moreh_binary_sub() {
    moreh_binary<APPROXIMATION_MODE, MorehBinary::SUB, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void moreh_binary_mul() {
    moreh_binary<APPROXIMATION_MODE, MorehBinary::MUL, ITERATIONS>();
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void moreh_adamw(
    uint32_t beta1_bits,
    uint32_t beta2_bits,
    uint32_t recip_bias_correction1_bits,
    uint32_t recip_bias_correction2_bits) {
    constexpr auto exp_avg_in = 0;
    constexpr auto exp_avg_sq_in = 32;
    constexpr auto grad_in = 64;

    constexpr auto exp_avg_out = 0;
    constexpr auto exp_avg_sq_out = 32;
    constexpr auto exp_avg_hat_out = 64;
    constexpr auto exp_avg_sq_hat_out = 96;

    float beta1 = Converter::to_float(beta1_bits);
    float beta2 = Converter::to_float(beta2_bits);
    float recip_bias_correction1 = Converter::to_float(recip_bias_correction1_bits);
    float recip_bias_correction2 = Converter::to_float(recip_bias_correction2_bits);

    for (int d = 0; d < ITERATIONS; d++) {
        vFloat grad = dst_reg[grad_in];
        vFloat exp_avg = dst_reg[exp_avg_in];
        vFloat exp_avg_sq = dst_reg[exp_avg_sq_in];
        // {
            vFloat beta1_v = beta1;
            vFloat one_minus_beta1_v = 1 - beta1;
            exp_avg = beta1_v * exp_avg + one_minus_beta1_v * grad;
            dst_reg[exp_avg_out] = exp_avg;

            vFloat recip_bias_correction1_v = recip_bias_correction1;
            dst_reg[exp_avg_hat_out] = exp_avg * recip_bias_correction1_v;
        // }
        // {
            vFloat beta2_v = beta2;
            vFloat one_minus_beta2_v = 1 - beta2;
            exp_avg_sq = beta2_v * exp_avg_sq + one_minus_beta2_v * (grad * grad);
            dst_reg[exp_avg_sq_out] = exp_avg_sq;

            vFloat recip_bias_correction2_v = recip_bias_correction2;
            dst_reg[exp_avg_sq_hat_out] = exp_avg_sq * recip_bias_correction2_v;
        // }
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
