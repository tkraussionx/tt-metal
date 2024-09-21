// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_simple.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

ALWI void simple_tile_init() { MATH((llk_math_eltwise_unary_sfpu_simple_tile_init<true>())); }

ALWI void simple_tile(uint32_t idst_data, uint32_t idst2_cond, uint32_t idst3_other) {
    MATH((llk_math_eltwise_unary_sfpu_simple_tile<APPROX>(idst_data)));
}

}  // namespace ckernel
