// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_moreh_binary.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

template <bool fast_and_approx = false>
ALWI void moreh_binary_add(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_moreh_binary_add<fast_and_approx>(idst)));
}

template <bool fast_and_approx = false>
ALWI void moreh_binary_sub(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_moreh_binary_sub<fast_and_approx>(idst)));
}

template <bool fast_and_approx = false>
ALWI void moreh_binary_mul(uint32_t idst) {
    MATH((llk_math_eltwise_unary_sfpu_moreh_binary_mul<fast_and_approx>(idst)));
}

template <bool fast_and_approx = false>
ALWI void moreh_adamw(
    uint32_t idst,
    uint32_t beta1_bits,
    uint32_t beta2_bits,
    uint32_t recip_bias_correction1_bits,
    uint32_t recip_bias_correction2_bits) {
    MATH((llk_math_eltwise_unary_sfpu_moreh_adamw<fast_and_approx>(
        idst, beta1_bits, beta2_bits, recip_bias_correction1_bits, recip_bias_correction2_bits)));
}

ALWI void moreh_binary_op_init() { MATH((llk_math_eltwise_unary_sfpu_moreh_binary_init<APPROX>())); }
ALWI void moreh_adamw_init() { MATH((llk_math_eltwise_unary_sfpu_moreh_adamw_init<APPROX>())); }

}  // namespace ckernel
