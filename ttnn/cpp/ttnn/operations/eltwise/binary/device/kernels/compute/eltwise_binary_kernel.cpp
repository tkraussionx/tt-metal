// // SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
// //
// // SPDX-License-Identifier: Apache-2.0

// #include <cstdint>
// #include "compute_kernel_api/eltwise_binary.h"
// #include "compute_kernel_api/tile_move_copy.h"

// #include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

// #include "debug/dprint.h"


// // ELTWISE_OP mul_tiles
// // ELTWISE_OP_TYPE EltwiseBinaryType::ELWMUL
// // SFPU_OP_CHAIN_0
// // SFPU_OP_FUNC_PRE_IN1_0 recip_tile(0);
// // SFPU_OP_INIT_PRE_IN1_0 recip_tile_init();
// // SFPU_OP_RECIP_INCLUDE 1

// // ELTWISE_OP mul_tiles
// // ELTWISE_OP_TYPE EltwiseBinaryType::ELWMUL
// // SFPU_OP_CHAIN_0

// #define PRE_SCALE defined SFPU_OP_INIT_PRE_IN0_0 || defined SFPU_OP_INIT_PRE_IN1_0

// namespace NAMESPACE {
// void MAIN {

//     ::tt::CB v1 = ::tt::CB::c_in0;
//     ::tt::CB v2 = ::tt::CB::c_in1;
//     ::tt::CB v3 = ::tt::CB::c_out0;
//     ::tt::CB v4 = ::tt::CB::c_intermed1;

//     int32_t v5 = 1;
//     int32_t v6 = 0;

//     // metal
//     auto cb_in0 = v1;
//     auto cb_in1 = v2;
//     auto cb_inp0 = v1;;
//     auto cb_inp1 = v4;;
//     auto cb_out0 =  v3;
//     uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
//     uint32_t per_core_block_size = get_arg_val<uint32_t>(1);
//     DPRINT << per_core_block_cnt << " " << per_core_block_size << ENDL();
//     // end metal

//     binary_op_init_common(v1, v4, v3);
//     mul_tiles_init_f();

//     int32_t v7;
//     v7 = v6;

//     for(uint32_t block = 0; block < per_core_block_cnt; ++block) {
//         copy_tile_to_dst_init_short(); // need to copy from CB to DST to be able to run sfpu math

//         unpack_reconfig_data_format_srca(v1, v2);
//         pack_reconfig_data_format(v3, v4);

//         cb_wait_front(cb_in1, per_core_block_size);
//         cb_reserve_back(cb_inp1, per_core_block_size);

//         tile_regs_acquire();
//         recip_tile_init();
//         copy_tile(v2, 0, 0); // copy from c_in[0] to DST[0]
//         recip_tile(0);
//         tile_regs_commit();

//         tile_regs_wait();
//         pack_tile(0, v4);
//         tile_regs_release();

//         cb_pop_front(cb_in1, per_core_block_size);
//         cb_push_back(cb_inp1, per_core_block_size);
//         unpack_reconfig_data_format_srca(v2, v1);
//         pack_reconfig_data_format(v4, v3);

//         cb_wait_front(cb_inp0, per_core_block_size);
//         cb_wait_front(cb_inp1, per_core_block_size);
//         cb_reserve_back(cb_out0, per_core_block_size);

//         mul_tiles_init();

//         tile_regs_acquire();
//         mul_tiles(v1, v4, 0, 0, 0);
//         tile_regs_commit();

//         tile_regs_wait();
//         pack_tile(0, v3);
//         tile_regs_release();

//         cb_pop_front(cb_inp0, per_core_block_size);
//         cb_pop_front(cb_inp1, per_core_block_size);
//         cb_push_back(cb_out0, per_core_block_size);
//     }

// }
// }


// #include <cstdint>
// #include "compute_kernel_api/common.h"
// #include "compute_kernel_api/tilize.h"
// #include "compute_kernel_api/untilize.h"
// #include "compute_kernel_api/eltwise_binary.h"
// #include "compute_kernel_api/tile_move_copy.h"
// #include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
// #include "compute_kernel_api/eltwise_unary/recip.h"
// namespace NAMESPACE {
// void kernel_main() {
//     ::tt::CB v1 = ::tt::CB::c_in0;
//     ::tt::CB v2 = ::tt::CB::c_in1;
//     ::tt::CB v3 = ::tt::CB::c_out0;
//     ::tt::CB v4 = ::tt::CB::c_intermed1;
//     int32_t v5 = 1;
//     int32_t v6 = 0;
//     binary_op_init_common(v1, v4, v3);
//     mul_tiles_init_f();
//     int32_t v7;
//     v7 = v6;

//     // metal
//     auto cb_in0 = v1;
//     auto cb_in1 = v2;
//     auto cb_inp0 = v1;
//     auto cb_inp1 = v4;
//     auto cb_out0 =  v3;
//     // uint32_t per_core_block_cnt = 1;
//     // uint32_t per_core_block_size = 1;
//     uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
//     uint32_t per_core_block_size = get_arg_val<uint32_t>(1);
//     // DPRINT << per_core_block_cnt << " " << per_core_block_size << ENDL();
//     // end metal

//     for (int32_t v8 = v6; v8 < v5; v8 += v5) {
//         int32_t v9;
//         v9 = v7;
//         for (int32_t v10 = v6; v10 < v5; v10 += v5) {
//             copy_tile_to_dst_init_short();
//             unpack_reconfig_data_format_srca(v1, v2);
//             pack_reconfig_data_format(v3, v4);

//             cb_wait_front(cb_in1, per_core_block_size);
//             cb_reserve_back(cb_inp1, per_core_block_size);

//             tile_regs_acquire();
//             recip_tile_init();
//             copy_tile(v2, v9, v9);
//             recip_tile(v6);
//             tile_regs_commit();
//             tile_regs_wait();
//             pack_tile(v6, v4, v9);
//             tile_regs_release();

//             cb_pop_front(cb_in1, per_core_block_size);
//             cb_push_back(cb_inp1, per_core_block_size);

//             unpack_reconfig_data_format_srca(v2, v1);
//             pack_reconfig_data_format(v4, v3);

//             cb_wait_front(cb_inp0, per_core_block_size);
//             cb_wait_front(cb_inp1, per_core_block_size);
//             cb_reserve_back(cb_out0, per_core_block_size);

//             mul_tiles_init(v1, v4);
//             tile_regs_acquire();
//             mul_tiles(v1, v4, v9, v9, v6);
//             tile_regs_commit();
//             tile_regs_wait();
//             pack_tile(v6, v3, v9);
//             tile_regs_release();

//             cb_pop_front(cb_inp0, per_core_block_size);
//             cb_pop_front(cb_inp1, per_core_block_size);
//             cb_push_back(cb_out0, per_core_block_size);

//             uint32_t v11 = (uint32_t) v9;
//             uint32_t v12 = (uint32_t) v5;
//             uint32_t v13 = v11 + v12;
//             int32_t v14 = (int32_t) v13;
//             v9 = v14;
//         };
//         v7 = v9;
//     }
//     return;
// }

// void MAIN { kernel_main(); }
// }


#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
namespace NAMESPACE {
void kernel_main() {
  ::tt::CB v1 = ::tt::CB::c_in0;
  ::tt::CB v2 = ::tt::CB::c_in1;
  ::tt::CB v3 = ::tt::CB::c_out0;
  ::tt::CB v4 = ::tt::CB::c_intermed1;
  uint32_t v5 = 1;
  uint32_t v6 = 0;

    // metal
    auto cb_in0 = v1;
    auto cb_in1 = v2;
    auto cb_inp0 = v1;
    auto cb_inp1 = v4;
    auto cb_out0 =  v3;
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);
    // end metal

  DPRINT << per_core_block_size << " " << v5 << ENDL();

  if (per_core_block_size == v5) {
    DPRINT << "equal" << ENDL();
  } else {
    DPRINT << "not equal" << ENDL();
  }

  binary_op_init_common(v1, v4, v3);
  mul_tiles_init_f();
  int32_t v7;
  v7 = v6;
  for (int32_t v8 = v6; v8 < 1; v8 += v5) {
    int32_t v9;
    v9 = v7;
    for (int32_t v10 = v6; v10 < 1; v10 += v5) {
      copy_tile_to_dst_init_short();
      unpack_reconfig_data_format_srca(v1, v2);
      pack_reconfig_data_format(v3, v4);

      cb_wait_front(v2, v5);
      cb_reserve_back(v4, v5);

      // cb_wait_front(cb_in1, per_core_block_size);
      // cb_reserve_back(cb_inp1, per_core_block_size);

      tile_regs_acquire();
      recip_tile_init();
      copy_tile(v2, v9, v9);
      recip_tile(v6);
      tile_regs_commit();
      tile_regs_wait();
      pack_tile(v6, v4, v9);
      tile_regs_release();

      // cb_pop_front(v2, v5);
      // cb_push_back(v4, v5);

      cb_pop_front(cb_in1, per_core_block_size);
      cb_push_back(cb_inp1, per_core_block_size);

      unpack_reconfig_data_format_srca(v2, v1);
      pack_reconfig_data_format(v4, v3);

      // cb_wait_front(v1, v5);
      // cb_wait_front(v4, v5);
      // cb_reserve_back(v3, v5);

      cb_wait_front(cb_inp0, per_core_block_size);
      cb_wait_front(cb_inp1, per_core_block_size);
      cb_reserve_back(cb_out0, per_core_block_size);

      mul_tiles_init(v1, v4);
      tile_regs_acquire();
      mul_tiles(v1, v4, v9, v9, v6);
      tile_regs_commit();
      tile_regs_wait();
      pack_tile(v6, v3, v9);
      tile_regs_release();

    //   cb_pop_front(v1, v5);
    //   cb_pop_front(v4, v5);
    //   cb_push_back(v3, v5);

      cb_pop_front(cb_inp0, per_core_block_size);
      cb_pop_front(cb_inp1, per_core_block_size);
      cb_push_back(cb_out0, per_core_block_size);

      uint32_t v11 = (uint32_t) v9;
      uint32_t v12 = (uint32_t) v5;
      uint32_t v13 = v11 + v12;
      int32_t v14 = (int32_t) v13;
      v9 = v14;
    };
    v7 = v9;
  }
  DPRINT << "done" << ENDL();
  return;
}

void MAIN { kernel_main(); }
}





// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// #include <cstdint>
// #include "compute_kernel_api/eltwise_binary.h"
// #include "compute_kernel_api/tile_move_copy.h"

// #include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

// #define PRE_SCALE defined SFPU_OP_INIT_PRE_IN0_0 || defined SFPU_OP_INIT_PRE_IN1_0

// namespace NAMESPACE {
// void MAIN {
//     uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
//     uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

//     constexpr auto cb_in0 = tt::CB::c_in0;
//     constexpr auto cb_in1 = tt::CB::c_in1;

//     #ifdef SFPU_OP_INIT_PRE_IN0_0
//         constexpr auto cb_inp0 = tt::CB::c_intermed0;
//     #else
//         constexpr auto cb_inp0 = cb_in0;
//     #endif

//     #ifdef SFPU_OP_INIT_PRE_IN1_0
//         constexpr auto cb_inp1 = tt::CB::c_intermed1;
//     #else
//         constexpr auto cb_inp1 = cb_in1;
//     #endif
//     constexpr auto cb_out0 =  tt::CB::c_out0;

//     binary_op_init_common(cb_inp0, cb_inp1, cb_out0);

//     #if not PRE_SCALE
//     binary_op_specific_init<false, ELTWISE_OP_TYPE>();
//     #endif

//     #ifdef PACK_RELU
//     PACK(( llk_pack_relu_config(ReluType::ZERO_RELU) ));
//     #endif

//     for(uint32_t block = 0; block < per_core_block_cnt; ++block) {

//         #if PRE_SCALE
//         copy_tile_to_dst_init_short(); // need to copy from CB to DST to be able to run sfpu math
//         #endif

//         // #ifdef SFPU_OP_INIT_PRE_IN0_0
//         // unpack_reconfig_data_format_srca(cb_inp0, cb_in0);
//         // pack_reconfig_data_format(cb_out0, cb_inp0);
//         // cb_wait_front(cb_in0, per_core_block_size);
//         // cb_reserve_back(cb_inp0, per_core_block_size);

//         // tile_regs_acquire();
//         // SFPU_OP_INIT_PRE_IN0_0
//         // for(uint32_t i = 0; i < per_core_block_size; ++i)
//         // {
//         //     copy_tile(cb_in0, i, i); // copy from c_in[0] to DST[0]
//         //     SFPU_OP_FUNC_PRE_IN0_0
//         // }
//         // tile_regs_commit();

//         // tile_regs_wait();
//         // for(uint32_t i = 0; i < per_core_block_size; ++i)
//         // {
//         //     pack_tile(i, cb_inp0); // DST[0]->cb
//         // }
//         // tile_regs_release();

//         // cb_pop_front(cb_in0, per_core_block_size);
//         // cb_push_back(cb_inp0, per_core_block_size);
//         // #ifndef SFPU_OP_INIT_PRE_IN1_0
//         // unpack_reconfig_data_format_srca(cb_in0, cb_inp0);
//         // pack_reconfig_data_format(cb_inp0, cb_out0);
//         // #endif
//         // #endif

//         #ifdef SFPU_OP_INIT_PRE_IN1_0
//         #ifndef SFPU_OP_INIT_PRE_IN0_0
//         unpack_reconfig_data_format_srca(cb_inp0, cb_in1);
//         pack_reconfig_data_format(cb_out0, cb_inp1);
//         #else
//         unpack_reconfig_data_format_srca(cb_in0, cb_in1);
//         pack_reconfig_data_format(cb_inp0, cb_inp1);
//         #endif
//         cb_wait_front(cb_in1, per_core_block_size);
//         cb_reserve_back(cb_inp1, per_core_block_size);

//         tile_regs_acquire();
//         SFPU_OP_INIT_PRE_IN1_0
//         for(uint32_t i = 0; i < per_core_block_size; ++i)
//         {
//             copy_tile(cb_in1, i, i); // copy from c_in[0] to DST[0]
//             SFPU_OP_FUNC_PRE_IN1_0
//         }
//         tile_regs_commit();

//         tile_regs_wait();
//         for(uint32_t i = 0; i < per_core_block_size; ++i)
//         {
//             pack_tile(i, cb_inp1); // DST[0]->cb
//         }
//         tile_regs_release();

//         cb_pop_front(cb_in1, per_core_block_size);
//         cb_push_back(cb_inp1, per_core_block_size);
//         unpack_reconfig_data_format_srca(cb_in1, cb_inp0);
//         pack_reconfig_data_format(cb_inp1, cb_out0);
//         #endif

//         cb_wait_front(cb_inp0, per_core_block_size);
//         cb_wait_front(cb_inp1, per_core_block_size);
//         cb_reserve_back(cb_out0, per_core_block_size);

//         #if PRE_SCALE
//         binary_op_specific_init<true, ELTWISE_OP_TYPE>();
//         #endif

//         tile_regs_acquire();
//         for(uint32_t i = 0; i < per_core_block_size; ++i)
//         {
//             ELTWISE_OP(cb_inp0, cb_inp1, i, i, i);

//             #ifdef SFPU_OP_INIT_0
//             SFPU_OP_INIT_0
//             SFPU_OP_FUNC_0
//             #endif

//             #ifdef SFPU_OP_CHAIN_0
//             SFPU_OP_CHAIN_0
//             #endif
//         }
//         tile_regs_commit();

//         tile_regs_wait();
//         for(uint32_t i = 0; i < per_core_block_size; ++i)
//         {
//             pack_tile(i, cb_out0);
//         }
//         tile_regs_release();

//         cb_pop_front(cb_inp0, per_core_block_size);
//         cb_pop_front(cb_inp1, per_core_block_size);
//         cb_push_back(cb_out0, per_core_block_size);
//     }

// }
// }
