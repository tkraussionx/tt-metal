// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"

using std::uint32_t;

#define P(s) \
    PACK( DPRINT << "P" << s << ENDL()); \
    UNPACK( DPRINT << "U" << s << ENDL()); \
    MATH( DPRINT << "M" << s << ENDL());

// matmul C=A*B using dims MK*KN = MN (row major order)
//
namespace NAMESPACE {
void MAIN {
    constexpr int onetile = 1;

    int dst_tile_index = 0;
    int in0_block_tile_index = 0;

    uint32_t batch = get_compile_time_arg_val(0);
    uint32_t Mt = get_compile_time_arg_val(1);
    uint32_t Kt = get_compile_time_arg_val(2);
    uint32_t Nt = get_compile_time_arg_val(3);

    P("bmm1")
    mm_init();

    // the simplest possible version of outer product blocked matmul
    // the reader is expected to read the A's and B's tile rows and tile columns for each output tile
    for (uint32_t nb = 0; nb < batch; nb++)
        for (uint32_t mt_C = 0; mt_C < Mt; ++mt_C)      // output tile of C
            for (uint32_t nt_C = 0; nt_C < Nt; ++nt_C)  // output tile index of C
            {
		P("bmm2")
                acquire_dst(tt::DstMode::Full);
		P("bmm3")
                for (uint32_t kt = 0; kt < Kt; kt++) {
		    P("bmm4")
                    cb_wait_front(tt::CB::c_in0, onetile);
		    P("bmm5")
                    cb_wait_front(tt::CB::c_in1, onetile);
		    P("bmm6")

                    matmul_tiles(tt::CB::c_in0, tt::CB::c_in1, 0, 0, 0, false);
		    P("bmm7")

                    cb_pop_front(tt::CB::c_in0, onetile);
		    P("bmm8")
                    cb_pop_front(tt::CB::c_in1, onetile);
		    P("bmm9")
                }
		P("bmm10")

                cb_reserve_back(tt::CB::c_out0, onetile);
		P("bmm11")
                pack_tile(0, tt::CB::c_out0);
		P("bmm12")
		P("BEFORE_PUSH_BACK_C_OUT_0")
                cb_push_back(tt::CB::c_out0, onetile);
		P("AFTER_PUSH_BACK_C_OUT_0")
		P("bmm13")

                release_dst(tt::DstMode::Full);
		P("bmm14")
            }
    P("bmm15")
}
}  // namespace NAMESPACE
