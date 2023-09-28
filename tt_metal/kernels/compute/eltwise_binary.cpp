// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api.h"

#if SFPU_OP_ERF_ERFC_INCLUDE
#include "compute_kernel_api/eltwise_unary/erf_erfc.h"
#endif

#if SFPU_OP_LOGICAL_NOT_NOTI_INCLUDE
#include "compute_kernel_api/eltwise_unary/logical_not_noti.h"
#endif

#if SFPU_OP_EXP_INCLUDE
#include "compute_kernel_api/eltwise_unary/exp.h"
#endif

#if SFPU_OP_GELU_INCLUDE
#include "compute_kernel_api/eltwise_unary/gelu.h"
#endif

#if SFPU_OP_SQRT_INCLUDE
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#endif

#if SFPU_OP_RECIP_INCLUDE
#include "compute_kernel_api/eltwise_unary/recip.h"
#endif

#if SFPU_OP_RELU_FAMILY_INCLUDE
#include "compute_kernel_api/eltwise_unary/relu.h"
#endif

#if SFPU_OP_ELU_INCLUDE
#include "compute_kernel_api/eltwise_unary/elu.h"
#endif

//#include "debug_print.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api.h"
//#include "debug_print.h"

ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

ALWI void prescale(tt::CB input, tt::CB output, uint32_t per_core_block_size) {

}
namespace NAMESPACE {
void MAIN {

    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_size = get_compile_time_arg_val(1);

    #ifdef SFPU_OP_INIT_PRE_IN0_0
        constexpr auto cb_inp0 = tt::CB::c_intermed0;
    #else
        constexpr auto cb_inp0 = tt::CB::c_in0;
    #endif

    #ifdef SFPU_OP_INIT_PRE_IN1_0
        constexpr auto cb_inp1 = tt::CB::c_intermed1;
    #else
        constexpr auto cb_inp1 = tt::CB::c_in1;
    #endif

    binary_op_init_common(cb_inp0, cb_inp1);

    for(uint32_t block = 0; block < per_core_block_cnt; ++block) {

        cb_reserve_back(tt::CB::c_out0, per_core_block_size);

        #ifdef SFPU_OP_INIT_PRE_IN0_0
        cb_wait_front(tt::CB::c_in0, per_core_block_size);
        cb_reserve_back(cb_inp0, per_core_block_size);
        copy_tile_init(); // need to copy from CB to DST to be able to run sfpu math
        ACQ();
        for(uint32_t i = 0; i < per_core_block_size; ++i)
        {
            copy_tile(tt::CB::c_in0, i, i); // copy from c_in[0] to DST[0]
            SFPU_OP_INIT_PRE_IN0_0
            SFPU_OP_FUNC_PRE_IN0_0
            pack_tile(i, cb_inp0); // DST[0]->cb
        }
        REL();
        cb_pop_front(tt::CB::c_in0, per_core_block_size);
        cb_push_back(cb_inp0, per_core_block_size);
        #endif

        #ifdef SFPU_OP_INIT_PRE_IN1_0
        cb_wait_front(tt::CB::c_in1, per_core_block_size);
        cb_reserve_back(cb_inp1, per_core_block_size);
        copy_tile_init(); // need to copy from CB to DST to be able to run sfpu math
        ACQ();
        for(uint32_t i = 0; i < per_core_block_size; ++i)
        {
            copy_tile(tt::CB::c_in1, i, i); // copy from c_in[0] to DST[0]
            SFPU_OP_INIT_PRE_IN1_0
            SFPU_OP_FUNC_PRE_IN1_0
            pack_tile(i, cb_inp1); // DST[0]->cb
        }
        REL();
        cb_pop_front(tt::CB::c_in1, per_core_block_size);
        cb_push_back(cb_inp1, per_core_block_size);
        #endif

        cb_wait_front(cb_inp0, per_core_block_size);
        cb_wait_front(cb_inp1, per_core_block_size);
        #if ELTWISE_OP_CODE == 0
        add_tiles_init();
        #elif ELTWISE_OP_CODE == 1
        sub_tiles_init();
        #else
        mul_tiles_init();
        #endif
        ACQ();
        for(uint32_t i = 0; i < per_core_block_size; ++i)
        {
            ELTWISE_OP(cb_inp0, cb_inp1, i, i, i);

            #if SFPU_OP_INIT_0
            SFPU_OP_INIT_0
            SFPU_OP_FUNC_0
            #endif

            #ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
            #endif

            pack_tile(i, tt::CB::c_out0);
        }
        REL();
        cb_pop_front(cb_inp0, per_core_block_size);
        cb_pop_front(cb_inp1, per_core_block_size);
        cb_push_back(tt::CB::c_out0, per_core_block_size);
    }

}
}
