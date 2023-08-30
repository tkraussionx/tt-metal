#pragma once


#include "compute_kernel_api/common.h"
#ifdef TRISC_MATH
#include "llk_math_reduce.h"
#endif


#ifdef TRISC_UNPACK
#include "llk_unpack_AB.h"
#include "llk_unpack_reduce.h"
#endif





namespace ckernel {

#if (defined(REDUCE_OP) and defined(REDUCE_DIM)) or defined(__DOXYGEN__)
ALWI void reduce_init(PoolType reduce_op, ReduceDim dim, uint32_t icb, float scaler)
{
    MATH(( llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, MATH_FIDELITY>() ));
    MATH(( llk_math_pack_sync_init<SyncFull>() )); // TODO(AP): check full

    PACK(( llk_pack_init() ));
    PACK(( llk_pack_reduce_hw_configure_disaggregated<false,REDUCE_OP, REDUCE_DIM>(16) ));
    PACK(( llk_setup_outputs() ));
    PACK(( llk_pack_dest_init<SyncFull, DstTileFaceLayout::RowMajor, false>() ));

    UNPACK(( llk_setup_operands() ));
    UNPACK(( llk_unpack_reduce_init<REDUCE_OP, REDUCE_DIM>() ));
    UNPACK(( llk_unpack_reduce_hw_configure_disaggregated<REDUCE_OP, REDUCE_DIM>(icb, scaler) ));
}

// TODO(AP): v2 is based on fusion-friendly implementation of reduce, keeping the original version around for now
template<bool at_start>
ALWI void reduce_init_v2(PoolType reduce_op, ReduceDim dim, uint32_t icb, uint32_t icb_scaler)
{
    UNPACK(( llk_setup_operands() ));
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_AB_init<>() ));
    #else
    UNPACK(( llk_unpack_AB_init<>(icb, icb_scaler) ));
    #endif
    UNPACK(( llk_unpack_AB_hw_configure_disaggregated(icb, icb_scaler) ));

    MATH(( llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, MATH_FIDELITY>() ));
    MATH(( llk_math_pack_sync_init<SYNC>() ));

    PACK(( llk_pack_init() ));
    PACK(( llk_pack_reduce_config_v2<REDUCE_DIM, at_start>(16) ));
    PACK(( llk_setup_outputs() ));
    PACK(( llk_pack_dest_init<SYNC, DstTileFaceLayout::RowMajor, false>() ));
}

// Delta from binary_op_init_common
template<bool at_start>
ALWI void reduce_init_delta_v2(PoolType reduce_op, ReduceDim dim, uint32_t ocb = 16)
{
    #ifdef ARCH_GRAYSKULL
    UNPACK(( llk_unpack_AB_init<>() ));
    #else
    // FIXME: API Update needed in compute kernel?
    UNPACK(( llk_unpack_AB_init<>(0, 1) ));
    #endif

    MATH(( llk_math_reduce_init<REDUCE_OP, REDUCE_DIM, MATH_FIDELITY>() ));

    PACK(( llk_pack_reduce_config_v2<REDUCE_DIM, at_start>(ocb) ));
}

ALWI void reduce_revert_delta_v2(uint32_t ocb = 16)
{
    PACK(( llk_pack_reduce_config_v2<REDUCE_DIM, false, true>(ocb) ));
}

/**
 * Performs a reduction operation *B = reduce(A)* using reduce_func for
 * dimension reduction on a tile in the CB at a given index and writes the
 * result to the DST register at index *dst_tile_index*. Reduction can be
 * either of type *Reduce::R*, *Reduce::C* or *Reduce::RC*, identifying the
 * dimension(s) to be reduced in size to 1. The DST register buffer must be in
 * acquired state via *acquire_dst* call.
 *
 * This call is blocking and is only available on the compute engine.
 *
 * Return value: None
 *
 * | Argument       | Description                                                     | Type     | Valid Range                                    | Required |
 * |----------------|-----------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | reduce_func    | Enum value, specifying the type of reduce function to perform.  | uint32_t | One of ReduceFunc::Sum, ReduceFunc::Max        | True     |
 * | dim            | Dimension id, identifying the dimension to reduce in size to 1. | uint32_t | One of Reduce::R, Reduce::C, Reduce::RC        | True     |
 * | in_cb_id       | The identifier of the circular buffer (CB) containing A         | uint32_t | 0 to 31                                        | True     |
 * | in_tile_index  | The index of tile A within the first CB                         | uint32_t | Must be less than the size of the CB           | True     |
 * | dst_tile_index | The index of the tile in DST REG for the result B               | uint32_t | Must be less than the acquired size of DST REG | True     |
 * | coeff          | Scaling factor applied to each element of the resulting tile.   | float    | any float number                               | True     |
 */
ALWI void reduce_tile(PoolType reduce_op, ReduceDim dim, uint32_t icb, uint32_t itile, uint32_t idst, float scaler)
{
    MATH(( llk_math_reduce<REDUCE_OP, REDUCE_DIM, MATH_FIDELITY>(idst) ));
    UNPACK(( llk_unpack_reduce<REDUCE_OP, REDUCE_DIM>(icb, itile) ));
}

// TODO(AP): v2 is based on fusion-friendly implementation of reduce, keeping the original version around for now
ALWI void reduce_tile_v2(PoolType reduce_op, ReduceDim dim, uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst)
{
    MATH(( llk_math_reduce<REDUCE_OP, REDUCE_DIM, MATH_FIDELITY>(idst) ));
    UNPACK(( llk_unpack_AB(icb0, icb1, itile0, itile1) ));
}
#endif




} // namespace ckernel
