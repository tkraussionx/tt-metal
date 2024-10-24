// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_sfpshft2_test.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif



namespace ckernel {

ALWI void sfpshft2_test_init() {
    MATH(( llk_math_eltwise_unary_sfpu_sfpshft2_test_init<APPROX>() ));
}

ALWI void sfpshft2_test(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_sfpshft2_test<APPROX>(idst)));
}

} // namespace ckerne
