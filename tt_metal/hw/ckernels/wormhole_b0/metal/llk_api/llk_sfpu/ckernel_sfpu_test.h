// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"

#include "ckernel_sfpu_converter.h"

#include "sfpi.h"
using namespace sfpi;

namespace ckernel
{
namespace sfpu
{

template <bool APPROXIMATION_MODE>
void test_init() {
    ;
}

template <bool APPROXIMATION_MODE, int ITERATIONS=8>
inline void calculate_test(uint option) {
    if (option == 0) {
#pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            vInt value = dst_reg[0];
            dst_reg[0] = value;
            dst_reg++;
        }
    } else if (option == 1) {
#pragma GCC unroll 0
        for (int d = 0; d < ITERATIONS; d++) {
            vInt value = dst_reg[0];
            vFloat value_f = int32_to_float(value);
            vInt tmp = reinterpret<vInt>(value_f);
            dst_reg[0] = tmp;
            dst_reg++;
        }
    }

}

    }  // namespace sfpu
}  // namespace ckernel
