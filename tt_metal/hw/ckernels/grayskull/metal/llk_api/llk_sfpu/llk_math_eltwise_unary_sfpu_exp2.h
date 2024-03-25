// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_0_param.h"
#include "ckernel_sfpu_exp2.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_exp2_init() {
    llk_math_eltwise_unary_sfpu_init<APPROXIMATE>(sfpu::exp2_init<APPROXIMATE>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_exp2(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_0_param<APPROXIMATE>
                                (ckernel::sfpu::calculate_exp2<APPROXIMATE>,
                                ckernel::sfpu::calculate_exp2<APPROXIMATE>,
                                dst_index, vector_mode);
}

}
