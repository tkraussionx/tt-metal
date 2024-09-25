// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_converter.h"
#include "noc_nonblocking_api.h"
#include "sfpi.h"
using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_simple() {
    constexpr int cond_val_idx = 64;
    constexpr int other_val_idx = 128;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        // vUInt cond = dst_reg[cond_val_idx];
        // vInt flag = cond == 0;

        // vFloat other = dst_reg[other_val_idx];
        // v_if (flag) {
        //     dst_reg[0] = other;
        // }
        // v_endif;

        TTI_SFPLOAD(0,6,3,cond_val_idx);
        TTI_SFPSETCC(0,0,0,6);
        TTI_SFPLOAD(0,0,3,other_val_idx);
        TTI_SFPSTORE(0,0,3,0);
        TTI_SFPENCC(0,0,0,0);

        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
