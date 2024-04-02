// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_1_param.h"
#include "ckernel_sfpu_test.h"


namespace ckernel {

    /************** test ************/

    template <bool APPROXIMATE>
    inline void llk_math_eltwise_unary_sfpu_test_init() {
        llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>(sfpu::test_init<APPROXIMATE>);
    }

    template <bool APPROXIMATE>
    inline void llk_math_eltwise_unary_sfpu_test(uint dst_index, uint param0 = 0) {
        llk_math_eltwise_unary_sfpu_1_param<APPROXIMATE>
                                    (ckernel::sfpu::calculate_test<APPROXIMATE,8>,
                                    ckernel::sfpu::calculate_test<APPROXIMATE,8>,
                                    dst_index, (int)VectorMode::RC, param0);
    }

}
