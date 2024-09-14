// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once


#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_moreh_sub.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif



namespace ckernel {

/**
 * Please refer to documentation for any_init.
 */
// template <bool fast_and_approx = false>
// ALWI void moreh_sub_init() {
//     MATH(( llk_math_eltwise_unary_sfpu_moreh_sub_init<fast_and_approx>() ));
// }

/**
 * Performs element-wise computation of exponential on each element of a tile
 * in DST register at index tile_index. The DST register buffer must be in
 * acquired state via *acquire_dst* call. This call is blocking and is only
 * available on the compute engine.
 *
 * Return value: None
 *
 * | Argument        | Description                                                                | Type     | Valid Range                                           | Required |
 * |-----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index      | The index of the tile in DST register buffer to perform the computation on | uint32_t | Must be less than the size of the DST register buffer | True     |
 * | fast_and_approx | Computation to be done faster and approximate                              | bool     |                                                       | False    |
 */
template <bool fast_and_approx = false>
ALWI void moreh_sub(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_moreh_sub<fast_and_approx>(idst) ));
}

} // namespace ckernel
