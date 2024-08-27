#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/moreh_fusion.h"
#include "compute_kernel_api/moreh_reduce_h_answer.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    uint32_t arg = 0;
    const auto input0_cb = get_arg_val<uint32_t>(arg++);
    const auto input1_cb = get_arg_val<uint32_t>(arg++);
    const auto output_cb = get_arg_val<uint32_t>(arg++);

    const auto slope0 = get_arg_val<uint32_t>(arg++);
    const auto slope1 = get_arg_val<uint32_t>(arg++);

    const auto num_tiles = get_arg_val<uint32_t>(arg++);

    constexpr auto dst0 = 0;
    constexpr auto dst1 = 1;
    constexpr auto first = 0;
    constexpr auto onetile = 1;

    init_sfpu(input0_cb);
    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        cb_wait_front(input0_cb, onetile);
        cb_wait_front(input1_cb, onetile);

        copy_tile_to_dst_init_short(input0_cb);
        copy_tile(input0_cb, first, dst0);
        copy_tile_to_dst_init_short(input1_cb);
        copy_tile(input1_cb, first, dst1);

        // Answer of practice 1
        // moreh_fusion_answer_init();
        // moreh_fusion_answer(dst0, slope0, slope1);
        moreh_fusion_init();
        moreh_fusion(dst0, slope0, slope1);

        // Answer of practice 2
        // moreh_reduce_h_answer_init();
        // moreh_reduce_h_answer(dst0);

        cb_pop_front(input0_cb, onetile);
        cb_pop_front(input1_cb, onetile);
        tile_regs_commit();

        tile_regs_wait();
        cb_reserve_back(output_cb, onetile);
        pack_tile(dst0, output_cb);
        cb_push_back(output_cb, onetile);

        tile_regs_release();
    }

    UNPACK(DPRINT << "UNPACK END" << ENDL());
    MATH(DPRINT << "MATH END" << ENDL());
    PACK(DPRINT << "PACK END" << ENDL());
}
}  // namespace NAMESPACE
