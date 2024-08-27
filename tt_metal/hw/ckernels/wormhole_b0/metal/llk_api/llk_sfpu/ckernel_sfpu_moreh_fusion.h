#pragma once
#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_sfpu_converter.h"
#include "debug/dprint.h"
#include "noc_nonblocking_api.h"
using namespace sfpi;
namespace ckernel {
namespace sfpu {
template <bool APPROXIMATION_MODE>
inline void moreh_fusion_init() {
    // nothing to do.
}
template <bool APPROXIMATION_MODE>
inline void moreh_fusion(uint slope0_bits, uint slope1_bits) {
    vFloat slope0 = Converter::to_float(slope0_bits);
    vFloat slope1 = Converter::to_float(slope1_bits);

#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        vFloat input0 = dst_reg[0];
        vFloat input1 = dst_reg[32];
        v_if(input0 < 0) { input0 = input0 * slope0; }
        v_endif;

        v_if(input1 < 0) { input1 = input1 * slope1; }
        v_endif;

        dst_reg[0] = input0 + input1;

        dst_reg++;
    }
}
}  // namespace sfpu
}  // namespace ckernel
