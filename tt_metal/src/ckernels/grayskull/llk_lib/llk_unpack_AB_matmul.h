#pragma once

#include "llk_io_unpack.h"
#include "llk_param_structs.h"

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_template.h"
#include "cunpack_common.h"
#include "ckernel_globals.h"

using namespace ckernel;
using namespace ckernel::unpacker;

inline void llk_unpack_AB_matmul_mop_config(bool transpose) {
    /*
    static constexpr uint unpack_srcb_top  = TT_OP_UNPACR(SrcB, 0b01000001, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0,
    0, 0, 0, 1); static constexpr uint unpack_srcb_bot =  TT_OP_UNPACR(SrcB, 0b01000001, 0, 0, 0, 1, 1,
    p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); static constexpr uint unpack_srca = TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1,
    1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1); ckernel_unpack_template tmp  = ckernel_unpack_template(false, // src B
                                                            true, // halo - just used for 4 unpacks
                                                            unpack_srcb_top,
                                                            unpack_srcb_bot,
                                                            unpack_srca,
                                                            unpack_srca,
                                                            0, 0, 0);
    */
    // UNPACK SRCB Z 0,2,1,3
    static constexpr uint unpack_src_set_z = TT_OP_SETADCZW(0b010, 0, 0, 0, 1, 0b0001);
    static constexpr uint unpack_src_set_z_transpose = TT_OP_SETADCZW(0b011, 0, 0, 0, 1, 0b0001);
#if SKIP_UNP0 == 1
    static constexpr uint unpack_srca0 = TT_OP_NOP;
    static constexpr uint unpack_srca1 = TT_OP_NOP;
    static constexpr uint unpack_srca0_transpose = TT_OP_NOP;
    static constexpr uint unpack_srca1_transpose = TT_OP_NOP;
#else
    static constexpr uint unpack_srca0 = TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint unpack_srca1 = TT_OP_UNPACR(SrcA, 0b1, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);

    static constexpr uint unpack_srca0_transpose = TT_OP_UNPACR(SrcA, 0b10, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint unpack_srca1_transpose = TT_OP_UNPACR(SrcA, 0b10, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
#endif
#if SKIP_UNP1 == 1
    static constexpr uint unpack_srcb_top = TT_OP_NOP;
    static constexpr uint unpack_srcb_bot = TT_OP_NOP;
#else
    static constexpr uint unpack_srcb_top =
        TT_OP_UNPACR(SrcB, 0b01000010, 0, 0, 0, 1, 0, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
    static constexpr uint unpack_srcb_bot =
        TT_OP_UNPACR(SrcB, 0b01000010, 0, 0, 0, 1, 1, p_unpacr::RAREFYB_DISABLE, 0, 0, 0, 0, 1);
#endif
    ckernel_unpack_template tmp = ckernel_unpack_template(
        true,  // src B
        true,  // halo - just used for 4 unpacks
        unpack_srcb_top,
        unpack_srcb_bot,
        transpose ? unpack_srca0_transpose : unpack_srca0,
        transpose ? unpack_srca1_transpose : unpack_srca1,
        0,
        transpose ? unpack_src_set_z_transpose : unpack_src_set_z,
        0);

    tmp.program(instrn_buffer);
}

inline void llk_unpack_AB_matmul_hw_configure(const llk_unpack_AB_matmul_params_t *unpack_AB_params) {
    configure_unpack_AB(
        // get_operand_id(unpack_AB_params->unpB_operand), get_operand_id(unpack_AB_params->unpA_operand), 16, 16);
        get_operand_id(unpack_AB_params->unpA_operand), get_operand_id(unpack_AB_params->unpB_operand), 16, 16);
}

inline void llk_unpack_AB_matmul_hw_configure_disaggregated(
    const std::uint32_t unpA_operand, const std::uint32_t unpB_operand, const std::uint32_t transpose_xy_srca = 0) {
    const llk_unpack_AB_matmul_params_t unpack_AB_matmul_params = {
        .unpA_operand = unpA_operand, .unpB_operand = unpB_operand, .transpose_xy_srca = transpose_xy_srca};
    llk_unpack_AB_matmul_hw_configure(&unpack_AB_matmul_params);
}


inline void llk_unpack_AB_matmul_init(const std::uint32_t transpose=0) { llk_unpack_AB_matmul_mop_config(transpose>0); }

inline void llk_unpack_AB_matmul(
    std::uint32_t operandA, std::uint32_t operandB, std::uint32_t tile_index_a, std::uint32_t tile_index_b) {
    std::uint32_t inputA = get_operand_id(operandA);
    std::uint32_t inputB = get_operand_id(operandB);

    std::uint32_t base_address_a = cb_read_interface[inputA].fifo_rd_ptr;
    std::uint32_t offset_address_a = MUL_TILE_SIZE_AND_INDEX((uint)unpack_src_format[inputA], tile_index_a);
    std::uint32_t base_address_b = cb_read_interface[inputB].fifo_rd_ptr;
    std::uint32_t offset_address_b = MUL_TILE_SIZE_AND_INDEX((uint)unpack_src_format[inputB], tile_index_b);

    // note: unpacker is programmed to automatically skip the tile header (+1)
    // since there is no tile header, we need to -1 the address (in terms of 16B words), to offet unpacker's automatic +1
    std::uint32_t address_a = base_address_a + offset_address_a - 1;
    std::uint32_t address_b = base_address_b + offset_address_b - 1;

    // DPRINT << "Float16_b: " << (uint) DataFormat::Float16_b << ", Bfp8_b: " << (uint) DataFormat::Bfp8_b << ENDL();
    // DPRINT << "IN A: " << inputA << ", IN B: " << inputB << ENDL();
    // DPRINT << "UPSF A: " << (uint) unpack_src_format[inputA] << ", UPSF B: " << (uint) unpack_src_format[inputB] << ENDL();
    // DPRINT << "OADDR A: " << offset_address_a << ", OADDR B: " << offset_address_b << ENDL();
    // DPRINT << "TIDX A: " << tile_index_a << ", TIDX B: " << tile_index_b << ENDL();

    // Clear z/w start counters
    TTI_SETADCZW(0b011, 0, 0, 0, 0, 0b1111);

    volatile uint *cfg = get_cfg_pointer();  // get pointer to registers for current state ID

    // Wait for free context
    wait_for_next_context(2);

    // Program srcA and srcB base addresses
    if (0 == unp_cfg_context) {
        cfg[THCON_SEC0_REG3_Base_address_ADDR32] = address_b;
        cfg[THCON_SEC1_REG3_Base_address_ADDR32] = address_a;
    } else {
        cfg[THCON_SEC0_REG3_Base_cntx1_address_ADDR32] = address_b;
        cfg[THCON_SEC1_REG3_Base_cntx1_address_ADDR32] = address_a;
    }

    semaphore_post(semaphore::UNPACK_SYNC);  // Trisc::SEMPOST for context acquire

#ifdef PERF_DUMP
    if (record_perf_events && !first_unpack_recorded) {
        uint32_t event_id_first_unpack = perf::get_event_id(
            0, 0, perf::EventType::UNPACK_FIRST_INSTRUCTION, current_outer_loop_iter);
        record_timestamp_64b(event_id_first_unpack);
        first_unpack_recorded = true;
    }
#endif

    // Stall unpacker until pending CFG writes from Trisc have completed
    // TTI_STALLWAIT(p_stall::STALL_UNPACK, p_stall::TRISC_CFG);

    // Run MOP
    mop_run(0, 2);

    // T6::SEMGET for context release
    t6_semaphore_get(semaphore::UNPACK_SYNC);

    // Switch unpacker config context
    switch_config_context(unp_cfg_context);
}
