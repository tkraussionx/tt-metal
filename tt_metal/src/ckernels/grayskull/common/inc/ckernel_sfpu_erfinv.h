/*
 * SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_erfinv_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::erfinv, APPROXIMATE>();
}

template <bool HAS_BASE_SCALING>
sfpi_inline void calculate_log_body_custom(const int log_base_scale_factor)
{
    sfpi::vFloat in = sfpi::dst_reg[0];
    sfpi::vFloat x = setexp(in, 127);
    sfpi::vFloat a = sfpi::s2vFloat16a(0.1058F);
    sfpi::vFloat series_result = x * (x * (x * a + sfpi::s2vFloat16a(-0.7122f)) + sfpi::s2vFloat16a(2.0869)) + sfpi::s2vFloat16a(-1.4753f);
    sfpi::vInt exp = 0;
    v_if (in != 0.0F) {
        exp = exexp(in);
        v_if (exp < 0) {
            exp = sfpi::abs(exp);
            in = setsgn(in, 1);
        }
        v_endif;
    }
    v_endif;
    sfpi::vInt new_exp = 0;
    v_if (exp != 0) {
        new_exp = lz(exp);
        new_exp = ~new_exp;
        new_exp += 19;
        v_if (new_exp >= 0) {
            new_exp += 127;
        }
        v_endif;
    }
    v_endif;
    sfpi::vFloat result = setexp(in, new_exp);
    sfpi::vInt shift = lz(exp) + 1;
    result = setman(result, shft(sfpi::reinterpret<sfpi::vUInt>(exp), shift));
    result = result * sfpi::vConst0p6929 + series_result;
    if constexpr (HAS_BASE_SCALING) {
        result *= sfpi::s2vFloat16a(log_base_scale_factor);
    }
    v_if (sfpi::dst_reg[0] == 0.0F) {
        result = -std::numeric_limits<float>::infinity();
    }
    v_endif;
    sfpi::dst_reg[0] = result;
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_sqrt_custom(sfpi::vFloat in)
{
    sfpi::vFloat val = in;
    sfpi::vFloat out;
    sfpi::vUInt magic = sfpi::reinterpret<sfpi::vUInt>(sfpi::vFloat(sfpi::s2vFloat16b(0x5f37)));
    sfpi::vFloat approx = sfpi::reinterpret<sfpi::vFloat>(magic - (sfpi::reinterpret<sfpi::vUInt>(val) >> 1));
    val = sfpi::dst_reg[0];
    for (int r = 0; r < 2; r++)
    {
        approx = (approx * approx * val * sfpi::vConstNeg0p5 + sfpi::vConst1 + 0.5F) * approx;
    }
    out = approx * val;

    return out;
}

template <bool APPROXIMATION_MODE>
sfpi_inline sfpi::vFloat calculate_erfinv_body(sfpi::vFloat in)
{
    sfpi::vFloat log_value = in * in;
    log_value = 1 - log_value;
    sfpi::dst_reg[0] = log_value;
    calculate_log_body_custom<false>(0);
    log_value = sfpi::dst_reg[0];
    sfpi::vFloat temp = sfpi::dst_reg[0] * 0.5;
    temp = 4.5469 + temp;
    temp = -temp;
    sfpi::vFloat calculated_value = (temp * temp) - (log_value * 7.1427);
    sfpi::dst_reg[0] = calculated_value;
    sfpi::vFloat intermediate_result = calculate_sqrt_custom<false>(sfpi::dst_reg[0]);
    calculated_value = temp + intermediate_result;
    sfpi::dst_reg[0] = calculated_value;
    log_value = calculate_sqrt_custom<false>(sfpi::dst_reg[0]);
    sfpi::dst_reg[0] = log_value;
    return log_value;
}

template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void calculate_erfinv()
{
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        v_if (sfpi::dst_reg[0] == 1.0f) {
            sfpi::dst_reg[0] = std::numeric_limits<float>::infinity();
        }v_elseif (sfpi::dst_reg[0] == -1.0f) {
            sfpi::dst_reg[0] = -std::numeric_limits<float>::infinity();
        }v_elseif ((sfpi::dst_reg[0] < -1.0f)||(sfpi::dst_reg[0] > 1.0f)) {
            sfpi::dst_reg[0] = std::numeric_limits<float>::quiet_NaN();
        }v_elseif (sfpi::dst_reg[0] < 0.0f) {
            calculate_erfinv_body<true>(sfpi::dst_reg[0]);
            sfpi::dst_reg[0] = -sfpi::dst_reg[0];
        }v_else {
            calculate_erfinv_body<true>(sfpi::dst_reg[0]);
        }
        v_endif;
        sfpi::dst_reg++;
    }
}

template <bool APPROXIMATE>
inline void erfinv_init() {
    ;
}

}  // namespace sfpu
}  // namespace ckernel
