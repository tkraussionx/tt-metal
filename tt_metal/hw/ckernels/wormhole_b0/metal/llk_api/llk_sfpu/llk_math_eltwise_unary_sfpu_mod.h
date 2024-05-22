// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_sfpu_mod.h"
#include "llk_math_eltwise_unary_sfpu_1_param.h"
#include "llk_math_eltwise_unary_sfpu_init.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_mod_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::mod, APPROXIMATE>();
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_mod(uint dst_index, uint param0, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_1_param<APPROXIMATE>(
        ckernel::sfpu::calculate_mod<APPROXIMATE>,
        ckernel::sfpu::calculate_mod<APPROXIMATE>,
        dst_index,
        vector_mode,
        param0);
}

}  // namespace ckernel
