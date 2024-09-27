// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_unary_sfpu_layernorm.h"
#define MAIN math_main()
#define MATH(x) x
#else
#define MATH(x)
#endif

namespace ckernel {

ALWI void layernorm_acc_tile(uint32_t idst, bool first) {
    MATH(( llk_math_eltwise_unary_sfpu_layernorm_acc<APPROX>(idst, first) ));
}

ALWI void layernorm_acc_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_layernorm_acc_init<APPROX>() ));
}

ALWI void layernorm_reduce_sum_w_tile(uint32_t idst, uint32_t scaler) {
    MATH(( llk_math_eltwise_unary_sfpu_layernorm_reduce_sum_w<APPROX>(idst, scaler) ));
}

ALWI void layernorm_reduce_sum_w_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_layernorm_reduce_sum_w_init<APPROX>() ));
}

ALWI void layernorm_reduce_sum_h_tile(uint32_t idst, uint32_t scaler) {
    MATH(( llk_math_eltwise_unary_sfpu_layernorm_reduce_sum_h<APPROX>(idst, scaler) ));
}

ALWI void layernorm_reduce_sum_h_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_layernorm_reduce_sum_h_init<APPROX>() ));
}

ALWI void layernorm_sub_tiles(uint32_t idst) {
    MATH(( llk_math_eltwise_unary_sfpu_layernorm_sub<APPROX>(idst) ));
}

ALWI void layernorm_sub_tiles_init() {
    MATH(( llk_math_eltwise_unary_sfpu_layernorm_sub_init<APPROX>() ));
}

ALWI void layernorm_sq_acc_tile(uint32_t idst, bool first) {
    MATH(( llk_math_eltwise_unary_sfpu_layernorm_sq_acc<APPROX>(idst, first) ));
}

ALWI void layernorm_sq_acc_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_layernorm_sq_acc_init<APPROX>() ));
}

ALWI void layernorm_rsqrt_tile(uint32_t idst, bool eps) {
    MATH(( llk_math_eltwise_unary_sfpu_layernorm_rsqrt<APPROX>(idst, eps) ));
}

ALWI void layernorm_rsqrt_tile_init() {
    MATH(( llk_math_eltwise_unary_sfpu_layernorm_rsqrt_init<APPROX>() ));
}

} // namespace ckernel
