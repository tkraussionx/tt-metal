// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_rand_uint.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

/**
 * Performs element-wise rand_uint on each element of a of a tile in DST register at index tile_index.
 * That is each element is overwritten with a randomly generated uint32.
 * The DST register buffer must be in acquired state via *acquire_dst* call.
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                                | Type     | Valid
 * Range                                           | Required |
 * |----------------|----------------------------------------------------------------------------|----------|-------------------------------------------------------|----------|
 * | tile_index     | The index of the tile in DST register buffer to perform typecast operation | uint32_t | Must be
 * less than the size of the DST register buffer | True     |
 */
ALWI void rand_uint_tile(uint32_t idst) { MATH((llk_math_eltwise_unary_sfpu_rand_uint<APPROX>(idst))); }

/**
 * Please refer to documentation for any_init.
 */
ALWI void rand_uint_tile_init(uint32_t seed) { MATH((llk_math_eltwise_unary_sfpu_rand_uint_init<APPROX>(seed))); }

}  // namespace ckernel
