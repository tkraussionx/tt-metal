// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"

using std::uint32_t;

namespace NAMESPACE {
void MAIN {

    constexpr int onetile = 1;

    uint32_t Kt = get_compile_time_arg_val(0);
    uint32_t num_output_tiles = get_compile_time_arg_val(1);
    bool transpose_b = (get_compile_time_arg_val(2) == 1);

    uint32_t cb_in0 = tt::CB::c_in0;
    uint32_t cb_in1 = tt::CB::c_in1;
    uint32_t cb_out0 = tt::CB::c_out0;

    // TODO: init matmul API

    // TODO: implement matmul and pack
    for (uint32_t n = 0; n < num_output_tiles; n++) {
    }

}
} // NAMESPACE
