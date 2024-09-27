// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_layernorm.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_layernorm_acc(uint dst_index, uint first, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_layernorm_acc<APPROXIMATE>,
        dst_index,
        vector_mode,
        first);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_layernorm_acc_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_layernorm_reduce_sum_w(uint dst_index, uint scaler, int vector_mode = (int)VectorMode::RC_custom) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_layernorm_reduce_sum_w<APPROXIMATE>,
        dst_index,
        vector_mode,
        scaler);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_layernorm_reduce_sum_w_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_layernorm_reduce_sum_h(uint dst_index, uint scaler, int vector_mode = (int)VectorMode::RC_custom) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_layernorm_reduce_sum_h<APPROXIMATE>,
        dst_index,
        vector_mode,
        scaler);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_layernorm_reduce_sum_h_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_layernorm_sub(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_layernorm_sub<APPROXIMATE>,
        dst_index,
        vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_layernorm_sub_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_layernorm_sq_acc(uint dst_index, uint first, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_layernorm_sq_acc<APPROXIMATE>,
        dst_index,
        vector_mode,
        first);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_layernorm_sq_acc_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_layernorm_rsqrt(uint dst_index, uint eps, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::calculate_layernorm_rsqrt<APPROXIMATE>,
        dst_index,
        vector_mode,
        eps);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_layernorm_rsqrt_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>();
}

}
