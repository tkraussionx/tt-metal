// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 4>
inline void calculate_add1()
{
    for (int d = 0; d < ITERATIONS; d++)
    {
        vFloat val = dst_reg[0];
        vFloat orig = dst_reg[0];

        vFloat res=0;
        if (val>0){
            while (val>1){
                val-=1;
            }
            res = orig-val-1;
            if (orig-res==1){
                res+=1;
            }
        }
        else if(val<0){
            val*=-1;
            while (val>1){
                val-=1;
            }
            res = -orig-val+2;
            res*=-1;
        }
        dst_reg[0] = res;
        dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
