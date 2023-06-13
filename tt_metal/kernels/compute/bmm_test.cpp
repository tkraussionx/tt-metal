#include <cstdint>

#include "llk_3c.h"

namespace NAMESPACE {
void MAIN {

    // CB indices
    uint32_t in0_cb_id = CB::c_in0;
    uint32_t in1_cb_id = CB::c_in1;
    uint32_t out_cb_id = CB::c_out0;

    // {   // BMM
    //     // init
    //     UNPACK(( llk_setup_operands() ));
    //     // init unpack
    //     UNPACK(( llk_unpack_AB_matmul_init() ));
    //     UNPACK(( llk_unpack_AB_matmul_hw_configure_disaggregated(in0_cb_id, in1_cb_id) ));
    //     // init compute
    //     MATH(( llk_math_matmul_init<MATH_FIDELITY>(0) ));
    //     MATH(( llk_math_pack_sync_init<SYNC>()  ));
    //     // init pack
    //     PACK(( llk_pack_init() ));
    //     PACK(( llk_pack_hw_configure_disaggregated<false>(out_cb_id) ));
    //     PACK(( llk_setup_outputs() ));
    //     PACK(( llk_pack_dest_init<SYNC, DstTileFaceLayout::RowMajor, false>() ));
    //     PACK(( llk_init_packer_dest_offset_registers<SyncHalf, DstTileFaceLayout::RowMajor, false>() ));

    //     // wait for inputs
    //     cb_wait_front(in0_cb_id, 1);
    //     cb_wait_front(in1_cb_id, 1);

    //     // lock dst
    //     acquire_dst(DstMode::Half);

    //     // matmul:
    //     // unpack in0_cb -> srca, in1_cb -> srcb
    //     UNPACK(( llk_unpack_AB_matmul(in0_cb_id, in1_cb_id, 0, 0) ));
    //     // matmul srca * srcb -> dst
    //     MATH(( llk_math_matmul<MATH_FIDELITY>(0) ));

    //     // reserve output
    //     cb_reserve_back(out_cb_id, 1);
    //     // pack dst -> out_cb
    //     PACK(( llk_pack<false, SYNC, false >(0, out_cb_id) ));
    //     // push output
    //     cb_push_back(out_cb_id, 1);

    //     // release dst lock
    //     release_dst(DstMode::Half);

    //     // pop inputs
    //     cb_pop_front(in0_cb_id, 1);
    //     cb_pop_front(in1_cb_id, 1);
    // }

    {   // element-wise ADD

        // binary_op_specific_init(ELTWISE_OP_CODE);
        MATH(( llk_math_eltwise_binary_init<ELWADD, NONE>() ));

        // binary_op_init_common(0, 1);
        UNPACK(( llk_setup_operands() ));
        UNPACK(( llk_unpack_AB_init<BroadcastType::NONE>() ));
        UNPACK(( llk_unpack_AB_hw_configure_disaggregated<BroadcastType::NONE>(in0_cb_id, in1_cb_id) ));
        MATH(( llk_math_pack_sync_init<SYNC>() ));
        PACK(( llk_pack_init() ));
        PACK(( llk_pack_hw_configure_disaggregated<false>(out_cb_id) ));
        PACK(( llk_setup_outputs() ));
        PACK(( llk_pack_dest_init<SYNC, DstTileFaceLayout::RowMajor, false>() ));

        cb_reserve_back(out_cb_id, 1);

        acquire_dst(DstMode::Half);

        cb_wait_front(in0_cb_id, 1);
        cb_wait_front(in1_cb_id, 1);

        // ELTWISE_OP is passed in via add_define
        // add_tiles(in0_cb_id, in1_cb_id, 0, 0, 0);
        UNPACK(( llk_unpack_AB(in0_cb_id, in1_cb_id, 0, 0) ));
        MATH(( llk_math_eltwise_binary<ELWADD, NONE, SyncHalf, MATH_FIDELITY, false>(0) ));

        // pack_tile(0, out_cb_id);
        PACK(( llk_pack<false, SYNC, false >(0, out_cb_id) ));

        release_dst(DstMode::Half);

        cb_pop_front(in0_cb_id, 1);
        cb_pop_front(in1_cb_id, 1);

        cb_push_back(out_cb_id, 1);
    }

} // MAIN
} // NAMESPACE
