#pragma once

#include "ckernel_sfpu_moreh_fusion.h"
#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_moreh_fusion_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>(sfpu::moreh_fusion_init<APPROXIMATE>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_moreh_fusion(uint dst_index, uint slope0_bits = 0, uint slope1_bits = 0) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        sfpu::moreh_fusion<APPROXIMATE>, dst_index, VectorMode::RC, slope0_bits, slope1_bits);
}

}  // namespace ckernel
