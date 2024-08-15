// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_moreh_reduce_h_answer.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

ALWI void moreh_reduce_h_answer_init() {
  MATH((llk_math_eltwise_unary_sfpu_moreh_reduce_h_answer_init<APPROX>()));
}

ALWI void moreh_reduce_h_answer(uint32_t idst) {
  MATH((llk_math_eltwise_unary_sfpu_moreh_reduce_h_answer<APPROX>(idst)));
}


}  // namespace ckernel
