// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_moreh_fusion_answer.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

ALWI void moreh_fusion_answer_init() { MATH((llk_math_eltwise_unary_sfpu_moreh_fusion_answer_init<APPROX>())); }

ALWI void moreh_fusion_answer(uint32_t idst, uint32_t slope0_bits, uint32_t slope1_bits) {
    MATH((llk_math_eltwise_unary_sfpu_moreh_fusion_answer<APPROX>(idst, slope0_bits, slope1_bits)));
}

}  // namespace ckernel
