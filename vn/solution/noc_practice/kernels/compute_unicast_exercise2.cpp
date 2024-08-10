#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    const auto num_tiles = get_arg_val<uint32_t>(0);

    const auto cb0_id = tt::CB::c_in0;
    const auto cb1_id = tt::CB::c_out0;
    constexpr auto dst0 = 0;
    constexpr auto first = 0;
    constexpr auto onetile = 1;

    unary_op_init_common(cb0_id, cb1_id);

    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        cb_wait_front(cb0_id, onetile);

        // This section is reserved for kernel debug print practice session.
        #if 0
        if (i == 0) {
            UNPACK(DPRINT << TSLICE(cb0_id, 0, SliceRange::hw0_32_16()) << ENDL());
        }
        #endif

        // This section is reserved for kernel debug print practice session.
        #if 0
        if (i == 0) {
            for (int32_t r = 0; r < 32; ++r) {
                SliceRange sr = SliceRange{
                    .h0 = static_cast<uint16_t>(r),
                    .h1 = static_cast<uint16_t>(r + 1),
                    .hs = 1,
                    .w0 = 0,
                    .w1 = 32,
                    .ws = 1};
                UNPACK(DPRINT << (uint)r << " --READ--cin0-- " << TileSlice(0, 0, sr, true, false) << ENDL());
            }
        }
        #endif



        copy_tile_init();
        copy_tile(cb0_id, first, dst0);
        cb_pop_front(cb0_id, onetile);
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(cb1_id, onetile);
        pack_tile(dst0, cb1_id, first);
        cb_push_back(cb1_id, onetile);
        tile_regs_release();
    }
}
}  // namespace NAMESPACE
