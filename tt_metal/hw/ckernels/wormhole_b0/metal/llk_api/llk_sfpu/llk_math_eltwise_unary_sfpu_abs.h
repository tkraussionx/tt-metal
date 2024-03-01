// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_0_param.h"
#include "ckernel_sfpu_abs.h"

namespace ckernel {

// New LLK SFPU APIs

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_abs_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::abs, APPROXIMATE>(sfpu::abs_init<APPROXIMATE>);
}

template <bool APPROXIMATE, DstSync Dst = DstSync::SyncFull>
inline void llk_math_eltwise_unary_sfpu_abs(uint dst_index, int vector_mode = (int)VectorMode::RC) {
    llk_math_eltwise_unary_sfpu_0_param<APPROXIMATE, Dst>
                                (ckernel::sfpu::calculate_abs<APPROXIMATE>,
                                ckernel::sfpu::calculate_abs<APPROXIMATE>,
                                dst_index, vector_mode);
}

}
