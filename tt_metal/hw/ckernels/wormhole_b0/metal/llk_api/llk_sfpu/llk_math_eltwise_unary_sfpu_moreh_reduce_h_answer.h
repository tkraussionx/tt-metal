
#pragma once

#include "llk_math_eltwise_unary_sfpu_init.h"
#include "llk_math_eltwise_unary_sfpu_params.h"
#include "ckernel_sfpu_moreh_reduce_h_answer.h"
namespace ckernel {

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_moreh_reduce_h_answer_init() {
    llk_math_eltwise_unary_sfpu_init<SfpuType::unused, APPROXIMATE>(sfpu::moreh_reduce_h_answer_init<APPROXIMATE>);
}

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_moreh_reduce_h_answer(uint dst_index) {
    llk_math_eltwise_unary_sfpu_params<APPROXIMATE>(
        sfpu::moreh_reduce_h_answer<APPROXIMATE>, dst_index, VectorMode::R);
}

}  // namespace ckernel
