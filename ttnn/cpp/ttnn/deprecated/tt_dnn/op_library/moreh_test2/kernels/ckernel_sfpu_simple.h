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

template <bool DATA_FLOAT, bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_simple() {
    constexpr int cond_val_idx = 32;
    constexpr int other_val_idx = 64;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vUInt cond = dst_reg[cond_val_idx];
        vInt flag = cond == 0;

        if constexpr (DATA_FLOAT) {
            vFloat other = dst_reg[other_val_idx];
            v_if(flag) { dst_reg[0] = other; }
            v_endif;
        } else {
            vInt other = dst_reg[other_val_idx];
            v_if(flag) { dst_reg[0] = other; }
            v_endif;
        }
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
