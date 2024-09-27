// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_moreh_binary.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"

namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_moreh_binary_add(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::moreh_binary_add<APPROXIMATE>, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_moreh_binary_sub(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::moreh_binary_sub<APPROXIMATE>, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_moreh_binary_mul(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::moreh_binary_mul<APPROXIMATE>, dst_index, vector_mode);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_moreh_adamw(
    uint dst_index, uint param1, uint param2, uint param3, uint param4, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::moreh_adamw<APPROXIMATE>, dst_index, vector_mode, param1, param2, param3, param4);
}

}  // namespace ckernel
