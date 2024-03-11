// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/kernels/packet_demux.hpp"

void kernel_main() {

    // TODO - implement demux logic

    constexpr uint32_t fan_out = get_compile_time_arg_val(0);

    volatile uint32_t* reg_ptr = reinterpret_cast<volatile uint32_t*>(0xFFB40000);
    reg_ptr[0] = 0xb00 + fan_out;
}
