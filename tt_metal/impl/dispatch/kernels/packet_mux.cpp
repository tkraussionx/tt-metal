// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/impl/dispatch/kernels/packet_mux.hpp"

void kernel_main() {

    constexpr uint32_t fan_in = get_compile_time_arg_val(0);

    // TODO: implement mux logic

    // test argument passing and pointer update

}
