// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_eager/tt_dnn/op_library/moreh_norm/kernel_utils/common_ckernels.hpp"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);

    std::uint8_t input_id{tt::CB::c_in0};
    const auto cb_x = input_id++;

    std::uint8_t output_id{tt::CB::c_out0};
    const auto cb_y = output_id++;

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in0);

    for (uint32_t idx = 0; idx < num_tiles; ++idx) {
        ACQ();
        cb_wait_front(cb_x, onetile);
        cb_reserve_back(cb_y, onetile);

        copy_tile_init();
        copy_tile(cb_x, 0, dst0);

        pack_tile(dst0, cb_y);

        cb_pop_front(cb_x, onetile);
        cb_push_back(cb_y, onetile);
        REL();
    }

}  // void MAIN
}  // namespace NAMESPACE
