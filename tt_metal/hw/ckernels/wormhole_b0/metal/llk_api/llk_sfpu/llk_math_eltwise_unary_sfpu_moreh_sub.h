// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_moreh_sub.h"

namespace ckernel {

// New LLK SFPU APIs

// template <bool APPROXIMATE>
// inline void llk_math_eltwise_unary_sfpu_moreh_sub_init() {
//     llk_math_eltwise_unary_sfpu_init<SfpuType::abs, APPROXIMATE>();
// }

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_moreh_sub(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        ckernel::sfpu::moreh_sub<APPROXIMATE>,
        dst_index,
        vector_mode);
}

}
