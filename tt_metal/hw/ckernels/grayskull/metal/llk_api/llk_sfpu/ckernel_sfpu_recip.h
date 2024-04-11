// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_defs.h"
#include "ckernel.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"

using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

template <bool save_reg, int max_iter = 3>
sfpi_inline vFloat sfpu_reciprocal(const vFloat in)
{
    vInt orig_exp;

    if constexpr (max_iter == 1) {
        // If we are only doing one iteration of the MAD loop, then we only need to use one LREG for the MAD instructions because we have our "first guess" in a hard-coded register
        // This allows us to avoid having to load back in the original value later on
        orig_exp = exexp(in);
    }

    // Force sign to 1 (make number negative)
    vFloat val = setsgn(in, 1);

    val = setexp(val, 126); // Set exponent to 126 to make the number in 0.5-1
    // Use 1.44 as first guess at x, ideal value would be 1.33, but we happen to have 1.44 available, so use that to avoid a load
    vFloat two;
    if (!save_reg) {
        two = 2.0f;
    }
    vFloat result = vConst1p4424 * (val * vConst1p4424 + (save_reg ? 2.0f : two));

    for (int s_iter = 0; s_iter < (max_iter-1); s_iter++) {
        result = result * (val * result + (save_reg ? 2.0f : two));
    }

    vInt new_exp = exexp(result);
    if constexpr (max_iter != 1) {
        orig_exp = exexp(dst_reg[0]);
    }

    // "Subtract" exponents, and re-bias.
    // Execute: -1 - exp, then exp += 127
    new_exp -= orig_exp;
    new_exp += 126;

    v_if (new_exp < 0) {
        // If rebiased exponent is negative, we need to saturate at 0.
        // This means the initial number was too big so reciprocal result should be 0
        result = 0.0F;
        new_exp = 0;
    }
    v_endif;

    // Set newly denormalized exponent to result exponent field
    return setexp(result, new_exp);
}

template <bool APPROXIMATION_MODE, int ITERATIONS=4>
inline void calculate_reciprocal()
{
    #pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        vFloat orig = dst_reg[0];

        vFloat res=0;
        if (val>0){
            while (val>1){
                val-=1;
            }
            res = orig-val-1;
            if (orig-res==1){
                res+=1;
            }
        }
        else if(val<0){
            val*=-1;
            while (val>1){
                val-=1;
            }
            res = -orig-val+2;
            res*=-1;
        }
        dst_reg[0] = res;
        dst_reg++;
    }
}

} // namespace sfpu
} // namespace ckernel
