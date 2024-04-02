// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_test.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif



namespace ckernel {

ALWI void sfpu_test_tile(uint32_t idst, uint32_t param0 = 0) {
    MATH(( llk_math_eltwise_unary_sfpu_test<APPROX>(idst, param0) ));
}

/**
 * Please refer to documentation for any_init.
 */
ALWI void sfpu_test_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_test_init<APPROX>() ));
}

} // namespace ckerne
