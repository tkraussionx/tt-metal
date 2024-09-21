#include <cstdint>

#include "simple.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

constexpr uint32_t onetile = 1;

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t num_tiles = get_compile_time_arg_val(0);

    constexpr uint32_t cb_input1 = tt::CB::c_in0;
    constexpr uint32_t cb_cond = tt::CB::c_in1;
    constexpr uint32_t cb_input2 = tt::CB::c_in2;
    constexpr uint32_t cb_output = tt::CB::c_out0;

    constexpr uint32_t input1_dst_reg = 0;
    constexpr uint32_t cond_dst_reg = input1_dst_reg + 1;
    constexpr uint32_t input2_dst_reg = input1_dst_reg + 2;

    unary_op_init_common(cb_input1, cb_output);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_input1, onetile);
        cb_wait_front(cb_cond, onetile);
        cb_wait_front(cb_input2, onetile);
        cb_reserve_back(cb_output, onetile);

        // since input cbs may have different data formats, source register must be reconfigured each time

        tile_regs_acquire();
        unpack_reconfig_data_format_srca(cb_input1);
        math_reconfig_data_format_srca(cb_input1);
        copy_tile_to_dst_init_short(cb_input1);
        copy_tile(cb_input1, 0, input1_dst_reg);

        unpack_reconfig_data_format_srca(cb_cond);
        math_reconfig_data_format_srca(cb_cond);
        copy_tile_to_dst_init_short(cb_cond);
        copy_tile(cb_cond, 0, cond_dst_reg);

        unpack_reconfig_data_format_srca(cb_input2);
        math_reconfig_data_format_srca(cb_input2);
        copy_tile_to_dst_init_short(cb_input2);
        copy_tile(cb_input2, 0, input2_dst_reg);

        simple_tile_init();
        simple_tile(input1_dst_reg, cond_dst_reg, input2_dst_reg);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(input1_dst_reg, cb_output);
        tile_regs_release();

        cb_push_back(cb_output, onetile);
        cb_pop_front(cb_input1, onetile);
        cb_pop_front(cb_cond, onetile);
        cb_pop_front(cb_input2, onetile);
    }
}
}  // namespace NAMESPACE
