// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_converter.h"
#include "noc_nonblocking_api.h"
#include "ckernel_sfpu_recip.h"

#include "sfpi.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{


template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_layernorm_acc(uint32_t first)
{
    if (first) {
        #pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            vFloat v = dst_reg[0];
            dst_reg[64] = v;
            dst_reg++;
        }
    } else {
        #pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            vFloat u = dst_reg[64];
            vFloat v = dst_reg[0];
            dst_reg[64] = u + v;
            dst_reg++;
        }
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_layernorm_reduce_sum_w(uint32_t scaler)
{
    #pragma GCC unroll 2
    for (int i = 0; i < 2; i++) {
        #pragma GCC unroll 4
        for (int j = 0; j < 4; j++) {
            TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, 32 * i + 4 * j);
            TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, 32 * i + 4 * j + 2);
            TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
            TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, 32 * i + 4 * j + 16);
            TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
            TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, 32 * i + 4 * j + 18);
            TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
            TTI_SFPNOP;

            TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
            #pragma GCC unroll 8
            for (int k = 0; k < 8; k++) {
                TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG0, 3);
                TTI_SFPNOP;
                TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG1, 0);
            }
            TTI_SFPNOP;

            _sfpu_load_imm32_(p_sfpu::LREG0, scaler);
            TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG1, 0);
            TTI_SFPNOP;

            TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, 32 * i + 4 * j);
            TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, 32 * i + 4 * j + 2);
            TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, 32 * i + 4 * j + 16);
            TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_3, 32 * i + 4 * j + 18);
        }
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_layernorm_reduce_sum_h(uint32_t scaler)
{
    #pragma GCC unroll 2
    for (int i = 0; i < 2; i++) {
        #pragma GCC unroll 2
        for (int j = 0; j < 2; j++) {
            TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, 16 * i + 2 * j);
            #pragma GCC unroll 3
            for (int k = 1; k < 4; k++) {
                TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, 16 * i + 2 * j + 4 * k);
                TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
            }
            #pragma GCC unroll 4
            for (int k = 0; k < 4; k++) {
                TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_3, 16 * i + 2 * j + 4 * k + 32);
                TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG0, 0);
            }
            TTI_SFPNOP;

            TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG1, 0);
            TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
            TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);

            TTI_SFPTRANSP(0, 0, 0, 0);
            TTI_SFPADD(p_sfpu::LREG0, p_sfpu::LCONST_1, p_sfpu::LREG1, p_sfpu::LREG1, 0);
            TTI_SFPADD(p_sfpu::LREG2, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG3, 0);
            TTI_SFPNOP;
            TTI_SFPADD(p_sfpu::LREG1, p_sfpu::LCONST_1, p_sfpu::LREG3, p_sfpu::LREG3, 0);
            TTI_SFPNOP;

            _sfpu_load_imm32_(p_sfpu::LREG0, scaler);
            TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG3, 0);
            TTI_SFPNOP;

            #pragma GCC unroll 4
            for (int k = 0; k < 4; k++) {
                TTI_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_3, 16 * i + 2 * j + 4 * k);
            }
            for (int k = 0; k < 4; k++) {
                TTI_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_3, 16 * i + 2 * j + 4 * k + 32);
            }
        }
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_layernorm_sub()
{
    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat u = dst_reg[0];
        vFloat v = dst_reg[32];
        dst_reg[0] = u - v;
        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_layernorm_sq_acc(uint32_t first)
{
    if (first) {
        #pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            vFloat v = dst_reg[0];
            dst_reg[32] = v * v;
            dst_reg++;
        }
    } else {
        #pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            vFloat u = dst_reg[32];
            vFloat v = dst_reg[0];
            dst_reg[32] = u + v * v;
            dst_reg++;
        }
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_layernorm_rsqrt(uint32_t eps)
{
    vFloat e = Converter::to_float(eps);

    #pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat in = dst_reg[0];
        in += e;

        vFloat result = 1.0f;
        v_if (in > 1.0f){
            result = sfpu_reciprocal(in);
        }
        v_endif;

        for (int r = 0; r < 25; r++) {
            // y = y * (1.5 - 0.5 * x * y * y) Newton's method iteration.
            result = result * (1.5F - 0.5F  * in * result * result);
        }
        dst_reg[0] = result;

        dst_reg++;
    }
}


}  // namespace sfpu
}  // namespace ckernel
