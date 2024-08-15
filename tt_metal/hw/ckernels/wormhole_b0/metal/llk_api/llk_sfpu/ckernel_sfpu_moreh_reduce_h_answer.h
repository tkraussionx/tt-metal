#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_converter.h"
#include "noc_nonblocking_api.h"

using namespace sfpi;
namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE>
inline void moreh_reduce_h_answer_init() {
    // nothing to do.
}

template <bool APPROXIMATION_MODE>
inline void moreh_reduce_h_answer() {
    vFloat odd_max = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < 4; i++) {
        vFloat v = dst_reg[i * 2];
        v_if(odd_max < v) { odd_max = v; }
        v_endif;
    }

    for (int i = 0; i < 4; i++) {
        vFloat v = dst_reg[16 + i * 2];
        v_if(odd_max < v) { odd_max = v; }
        v_endif;
    }


    vFloat even_max = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < 4; i++) {
        vFloat v = dst_reg[1 + i * 2];
        v_if(even_max < v) { even_max = v; }
        v_endif;
    }

    for (int i = 0; i < 4; i++) {
        vFloat v = dst_reg[17 + i * 2];
        v_if(even_max < v) { even_max = v; }
        v_endif;
    }

    vFloat r0 = odd_max;
    vFloat r1 = even_max;
    vFloat r2 = 0.0f;
    vFloat r3 = 0.0f;
    subvec_transp(r0, r1, r2, r3);

    v_if(r0 < r1) { r0 = r1; }
    v_endif;

    v_if(r0 < r2) { r0 = r2; }
    v_endif;

    v_if(r0 < r3) { r0 = r3; }
    v_endif;

    subvec_transp(r0, r1, r2, r3);

    odd_max = r0;
    even_max = r1;

    dst_reg[0] = r0;
    dst_reg[1] = r1;
}

}  // namespace sfpu
}  // namespace ckernel
