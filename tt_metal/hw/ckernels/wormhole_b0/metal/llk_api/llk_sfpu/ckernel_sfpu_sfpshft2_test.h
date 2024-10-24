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

template <bool APPROXIMATION_MODE>
inline void calculate_sfpshft2_test()
{
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);

    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG1, 4);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG0, 3);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG1, 4);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG0, 3);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, p_sfpu::LREG0, p_sfpu::LREG1, 4);
    TTI_SFPNOP;
    TTI_SFPSHFT2(0, p_sfpu::LREG1, p_sfpu::LREG0, 3);
    TTI_SFPNOP;

    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_3, 0);
}

}  // namespace sfpu
}  // namespace ckernel
