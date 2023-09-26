// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/tilize.h"

//#include "debug_print.h"

inline void sleep_loop(uint32_t loop_count = 100000) {
    for (volatile uint32_t i = 0; i < loop_count; i++);
}

namespace NAMESPACE {
void MAIN {

    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    //UNPACK(( DPRINT << "Block count=" << U32(per_core_block_cnt) << " tile count=" << per_core_block_tile_cnt << ENDL() ));
    tilize_init(tt::CB::c_in0, per_core_block_tile_cnt, tt::CB::c_out0);

    for(uint32_t b=0;b<per_core_block_cnt;++b)
    {
        cb_wait_front(tt::CB::c_in0, per_core_block_tile_cnt);
        cb_reserve_back(tt::CB::c_out0, per_core_block_tile_cnt);

        tilize_block(tt::CB::c_in0, per_core_block_tile_cnt, tt::CB::c_out0);

        cb_push_back(tt::CB::c_out0, per_core_block_tile_cnt);
        sleep_loop(2500);
        cb_pop_front(tt::CB::c_in0, per_core_block_tile_cnt);
    }
}
}
