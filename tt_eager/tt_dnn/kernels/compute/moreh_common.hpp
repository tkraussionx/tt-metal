/*
 * SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#ifndef REDUCE_OP
#define REDUCE_OP PoolType::SUM
#endif

#ifndef REDUCE_DIM
#define REDUCE_DIM ReduceDim::REDUCE_ROW
#endif

#include "compute_kernel_api.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/negative.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/mask.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/tile_move_copy.h"

ALWI void ACQ() { acquire_dst(tt::DstMode::Half); }
ALWI void REL() { release_dst(tt::DstMode::Half); }

namespace ckernel {

class ArgFetcher {
   private:
    int arg_idx = 0;

   public:
    template <typename T>
    T get_next_arg_val() {
        return get_arg_val<T>(arg_idx++);
    }
};

ALWI void mul_tiles_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb0, itile0 + 1);
    cb_wait_front(icb1, itile1 + 1);

    tile_regs_acquire();
    mul_tiles_init();
    mul_tiles(icb0, icb1, itile0, itile1, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(dst0, ocb);
    tile_regs_release();

    if (pop0)
        cb_pop_front(icb0, pop0);
    if (pop1)
        cb_pop_front(icb1, pop1);

    cb_push_back(ocb, onetile);
}

ALWI void mul_tiles_and_negative_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb0, itile0 + 1);
    cb_wait_front(icb1, itile1 + 1);

    tile_regs_acquire();
    mul_tiles_init();
    mul_tiles(icb0, icb1, itile0, itile1, dst0);

    negative_tile_init();
    negative_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(dst0, ocb);
    tile_regs_release();

    if (pop0)
        cb_pop_front(icb0, pop0);
    if (pop1)
        cb_pop_front(icb1, pop1);

    cb_push_back(ocb, onetile);
}

ALWI void mul_tiles_and_mask_tile_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t maskcb,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t mtile = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1,
    uint32_t popm = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb0, itile0 + 1);
    cb_wait_front(icb1, itile1 + 1);
    cb_wait_front(maskcb, mtile + 1);

    tile_regs_acquire();
    mul_tiles_init();
    mul_tiles(icb0, icb1, itile0, itile1, dst0);

    constexpr int dst_mask = 1;
    copy_tile_init();
    copy_tile(maskcb, mtile, dst_mask);

    mask_tile_init();
    mask_tile(dst0, dst_mask);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(dst0, ocb);
    tile_regs_release();

    if (pop0)
        cb_pop_front(icb0, pop0);
    if (pop1)
        cb_pop_front(icb1, pop1);
    if (popm)
        cb_pop_front(maskcb, popm);

    cb_push_back(ocb, onetile);
}

ALWI void mul_tiles_log_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb0, itile0 + 1);
    cb_wait_front(icb1, itile1 + 1);

    tile_regs_acquire();
    mul_tiles_init();
    mul_tiles(icb0, icb1, itile0, itile1, dst0);

    log_tile_init();
    log_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(dst0, ocb);
    tile_regs_release();

    if (pop0)
        cb_pop_front(icb0, pop0);
    if (pop1)
        cb_pop_front(icb1, pop1);

    cb_push_back(ocb, onetile);
}

ALWI void mul_tiles_bcast_rows_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);

    cb_wait_front(icb0, itile0 + 1);
    cb_wait_front(icb1, itile1 + 1);

    tile_regs_acquire();
    mul_bcast_rows_init_short();
    mul_tiles_bcast_rows(icb0, icb1, itile0, itile1, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(dst0, ocb);
    tile_regs_release();

    if (pop0)
        cb_pop_front(icb0, pop0);
    if (pop1)
        cb_pop_front(icb1, pop1);

    cb_push_back(ocb, onetile);
}

ALWI void mul_tiles_bcast_rows_log_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);

    cb_wait_front(icb0, itile0 + 1);
    cb_wait_front(icb1, itile1 + 1);

    tile_regs_acquire();
    mul_bcast_rows_init_short();
    mul_tiles_bcast_rows(icb0, icb1, itile0, itile1, dst0);

    log_tile_init();
    log_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(dst0, ocb);
    tile_regs_release();

    if (pop0)
        cb_pop_front(icb0, pop0);
    if (pop1)
        cb_pop_front(icb1, pop1);

    cb_push_back(ocb, onetile);
}

ALWI void mul_tiles_bcast_cols_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);

    cb_wait_front(icb0, itile0 + 1);
    cb_wait_front(icb1, itile1 + 1);

    tile_regs_acquire();
    mul_bcast_cols_init_short();
    mul_tiles_bcast_cols(icb0, icb1, itile0, itile1, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(dst0, ocb);
    tile_regs_release();

    if (pop0)
        cb_pop_front(icb0, pop0);
    if (pop1)
        cb_pop_front(icb1, pop1);

    cb_push_back(ocb, onetile);
}

ALWI void mul_tiles_bcast_cols_log_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);

    cb_wait_front(icb0, itile0 + 1);
    cb_wait_front(icb1, itile1 + 1);

    tile_regs_acquire();
    mul_bcast_cols_init_short();
    mul_tiles_bcast_cols(icb0, icb1, itile0, itile1, dst0);

    log_tile_init();
    log_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(dst0, ocb);
    tile_regs_release();

    if (pop0)
        cb_pop_front(icb0, pop0);
    if (pop1)
        cb_pop_front(icb1, pop1);

    cb_push_back(ocb, onetile);
}

ALWI void copy_tile_to_cb(uint32_t icb, uint32_t ocb, uint32_t itile = 0, uint32_t pop = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb, itile + 1);

    tile_regs_acquire();
    copy_tile_init();
    copy_tile(icb, itile, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(dst0, ocb);
    tile_regs_release();

    if (pop)
        cb_pop_front(icb, pop);
    cb_push_back(ocb, onetile);
}

ALWI void add_tiles_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb0, itile0 + 1);
    cb_wait_front(icb1, itile1 + 1);

    tile_regs_acquire();
    add_tiles_init();
    add_tiles(icb0, icb1, itile0, itile1, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(dst0, ocb);
    tile_regs_release();

    if (pop0)
        cb_pop_front(icb0, pop0);
    if (pop1)
        cb_pop_front(icb1, pop1);

    cb_push_back(ocb, onetile);
}

ALWI void mask_tile_to_cb(uint32_t icb, uint32_t maskcb, uint32_t ocb, uint32_t itile = 0, uint32_t mtile = 0, uint32_t pop = 1, uint32_t popm = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;
    constexpr int dst_mask = 1;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb, itile + 1);
    cb_wait_front(maskcb, mtile + 1);

    tile_regs_acquire();
    copy_tile_init();
    copy_tile(icb, itile, dst0);

    copy_tile_init();
    copy_tile(maskcb, mtile, dst_mask);

    mask_tile_init();
    mask_tile(dst0, dst_mask);

    tile_regs_commit();

    tile_regs_wait();
    pack_tile(dst0, ocb);
    tile_regs_release();

    if (pop)
        cb_pop_front(icb, pop);
    if (popm)
        cb_pop_front(maskcb, popm);

    cb_push_back(ocb, onetile);
}

ALWI void reduce_tile_to_cb(
    PoolType reduce_op,
    ReduceDim dim,
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t size,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);

    tile_regs_acquire();
    cb_wait_front(icb1, onetile);

    reduce_init_delta<false>(reduce_op, dim);
    for (uint32_t x = 0; x < size; ++x) {
        cb_wait_front(icb0, x + 1);  // must be a cumulative wait for correctness

        constexpr uint32_t bcast_scaler0 = 0;  // 0th index from bcast_scaler CB
        reduce_tile(icb0, icb1, x, bcast_scaler0, dst0);
    }
    tile_regs_commit();

    if (pop0)
        cb_pop_front(icb0, pop0);
    if (pop1)
        cb_pop_front(icb1, pop1);

    reduce_revert_delta();

    tile_regs_wait();
    pack_tile(dst0, ocb);
    tile_regs_release();

    cb_push_back(ocb, onetile);
}

ALWI void sub_tiles_bcast_cols_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);

    cb_wait_front(icb0, itile0 + 1);
    cb_wait_front(icb1, itile1 + 1);

    tile_regs_acquire();
    sub_bcast_cols_init_short();
    sub_tiles_bcast<BroadcastType::COL>(icb0, icb1, itile0, itile1, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(dst0, ocb);
    tile_regs_release();

    if (pop0)
        cb_pop_front(icb0, pop0);
    if (pop1)
        cb_pop_front(icb1, pop1);

    cb_push_back(ocb, onetile);
}

ALWI void sub_tiles_bcast_rows_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);

    cb_wait_front(icb0, itile0 + 1);
    cb_wait_front(icb1, itile1 + 1);

    tile_regs_acquire();
    // sub_bcast_rows_init_short();
    {
        MATH((llk_math_eltwise_binary_init<ELWSUB, BroadcastType::ROW, MATH_FIDELITY>()));
        UNPACK((llk_unpack_AB_init<BroadcastType::ROW>(0, 1)));
    }
    sub_tiles_bcast<BroadcastType::ROW>(icb0, icb1, itile0, itile1, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(dst0, ocb);
    tile_regs_release();

    if (pop0)
        cb_pop_front(icb0, pop0);
    if (pop1)
        cb_pop_front(icb1, pop1);

    cb_push_back(ocb, onetile);
}

ALWI void sub_tiles_to_cb(
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t itile0 = 0,
    uint32_t itile1 = 0,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb0, itile0 + 1);
    cb_wait_front(icb1, itile1 + 1);

    tile_regs_acquire();
    sub_tiles_init();
    sub_tiles(icb0, icb1, itile0, itile1, dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(dst0, ocb);
    tile_regs_release();

    if (pop0)
        cb_pop_front(icb0, pop0);
    if (pop1)
        cb_pop_front(icb1, pop1);

    cb_push_back(ocb, onetile);
}

ALWI void exp_tile_to_cb(uint32_t icb, uint32_t ocb, uint32_t itile = 0, uint32_t dst = 0, uint32_t pop = 1) {
    constexpr uint32_t onetile = 1;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb, itile + 1);

    tile_regs_acquire();
    copy_tile_init();
    copy_tile(icb, itile, dst);

    exp_tile_init();
    exp_tile(dst);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(dst, ocb);
    tile_regs_release();

    if (pop)
        cb_pop_front(icb, pop);
    cb_push_back(ocb, onetile);
}

ALWI void rexp_tile_to_cb(uint32_t icb, uint32_t ocb, uint32_t itile = 0, uint32_t dst = 0, uint32_t pop = 1) {
    constexpr uint32_t onetile = 1;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb, itile + 1);

    tile_regs_acquire();
    copy_tile_init();
    copy_tile(icb, itile, dst);

    negative_tile_init();
    negative_tile(dst);

    exp_tile_init();
    exp_tile(dst);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(dst, ocb);
    tile_regs_release();

    if (pop)
        cb_pop_front(icb, pop);
    cb_push_back(ocb, onetile);
}

ALWI void exp_tile_and_mask_tile_to_cb(
    uint32_t icb,
    uint32_t maskcb,
    uint32_t ocb,
    uint32_t itile = 0,
    uint32_t mtile = 0,
    uint32_t pop = 1,
    uint32_t popm = 1,
    uint32_t dst = 0,
    uint32_t dst_mask = 1) {
    constexpr uint32_t onetile = 1;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb, itile + 1);
    cb_wait_front(maskcb, mtile + 1);

    tile_regs_acquire();
    copy_tile_init();
    copy_tile(icb, itile, dst);

    if (pop)
        cb_pop_front(icb, pop);

    exp_tile_init();
    exp_tile(dst);

    copy_tile_init();
    copy_tile(maskcb, mtile, dst_mask);

    mask_tile_init();
    mask_tile(dst, dst_mask);
    tile_regs_commit();

    if (popm)
        cb_pop_front(maskcb, popm);

    tile_regs_wait();
    pack_tile(dst, ocb);
    tile_regs_release();

    cb_push_back(ocb, onetile);
}

ALWI void rexp_tile_and_mask_tile_to_cb(
    uint32_t icb,
    uint32_t maskcb,
    uint32_t ocb,
    uint32_t itile = 0,
    uint32_t mtile = 0,
    uint32_t pop = 1,
    uint32_t popm = 1,
    uint32_t dst = 0,
    uint32_t dst_mask = 1) {
    constexpr uint32_t onetile = 1;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb, itile + 1);
    cb_wait_front(maskcb, mtile + 1);

    tile_regs_acquire();
    copy_tile_init();
    copy_tile(icb, itile, dst);

    if (pop)
        cb_pop_front(icb, pop);

    negative_tile_init();
    negative_tile(dst);

    exp_tile_init();
    exp_tile(dst);

    copy_tile_init();
    copy_tile(maskcb, mtile, dst_mask);

    mask_tile_init();
    mask_tile(dst, dst_mask);
    tile_regs_commit();

    if (popm)
        cb_pop_front(maskcb, popm);

    tile_regs_wait();
    pack_tile(dst, ocb);
    tile_regs_release();

    cb_push_back(ocb, onetile);
}

ALWI void recip_tile_to_cb(uint32_t icb, uint32_t ocb, uint32_t itile = 0, uint32_t pop = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb, itile + 1);

    tile_regs_acquire();
    copy_tile_init();
    copy_tile(icb, itile, dst0);

    recip_tile_init();
    recip_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(dst0, ocb);
    tile_regs_release();

    if (pop)
        cb_pop_front(icb, pop);
    cb_push_back(ocb, onetile);
}

ALWI void reduce_tile_and_recip_tile_to_cb(
    PoolType reduce_op,
    ReduceDim dim,
    uint32_t icb0,
    uint32_t icb1,
    uint32_t ocb,
    uint32_t size,
    uint32_t pop0 = 1,
    uint32_t pop1 = 1) {
    constexpr uint32_t onetile = 1;
    constexpr int dst0 = 0;

    cb_reserve_back(ocb, onetile);
    cb_wait_front(icb1, onetile);

    tile_regs_acquire();
    reduce_init_delta<false>(reduce_op, dim);
    for (uint32_t x = 0; x < size; ++x) {
        cb_wait_front(icb0, x + 1);  // must be a cumulative wait for correctness

        constexpr uint32_t bcast_scaler0 = 0;  // 0th index from bcast_scaler CB
        reduce_tile(icb0, icb1, x, bcast_scaler0, dst0);
    }
    if (pop0)
        cb_pop_front(icb0, pop0);
    if (pop1)
        cb_pop_front(icb1, pop1);

    reduce_revert_delta();

    recip_tile_init();
    recip_tile(dst0);
    tile_regs_commit();

    tile_regs_wait();
    pack_tile(dst0, ocb);
    tile_regs_release();

    cb_push_back(ocb, onetile);
}

// TODO(seunghwan100): If p is 2 and decimal is 0, we can use sqrt_tile.
ALWI void power_tile_to_cb(
    std::uint8_t cb_x,
    std::uint8_t cb_xpow,
    std::uint8_t cb_logx,
    std::uint8_t cb_decimal,
    std::uint8_t cb_exp_lxmd,
    std::uint8_t cb_correct_xpow,
    uint32_t p,
    bool p_is_negative) {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    // x^p
    ACQ();
    cb_wait_front(cb_x, onetile);
    cb_reserve_back(cb_xpow, onetile);

    copy_tile_init();
    copy_tile(cb_x, 0, dst0);

    power_tile_init();
    power_tile(dst0, p);

    if (p_is_negative) {
        recip_tile_init();
        recip_tile(dst0);
    }

    pack_tile(dst0, cb_xpow);

    cb_push_back(cb_xpow, onetile);
    REL();
    // We don't pop cb_x here.

    // log(x)
    ACQ();
    cb_reserve_back(cb_logx, onetile);

    copy_tile_init();
    copy_tile(cb_x, 0, dst0);

    log_tile_init();
    log_tile(dst0);

    pack_tile(dst0, cb_logx);

    cb_pop_front(cb_x, onetile);
    cb_push_back(cb_logx, onetile);
    REL();

    // exp(log(x) * decimal)
    ACQ();
    cb_wait_front(cb_logx, onetile);
    cb_reserve_back(cb_exp_lxmd, onetile);

    mul_tiles_init();
    mul_tiles(cb_logx, cb_decimal, 0, 0, dst0);

    exp_tile_init();
    exp_tile(dst0);

    pack_tile(dst0, cb_exp_lxmd);

    cb_pop_front(cb_logx, onetile);
    cb_push_back(cb_exp_lxmd, onetile);
    REL();

    // x^p * exp(log(x) * decimal)(==(x + decimal)^p)
    ACQ();
    cb_wait_front(cb_xpow, onetile);
    cb_wait_front(cb_exp_lxmd, onetile);
    cb_reserve_back(cb_correct_xpow, onetile);

    mul_tiles_init();
    mul_tiles(cb_xpow, cb_exp_lxmd, 0, 0, dst0);

    pack_tile(dst0, cb_correct_xpow);

    cb_pop_front(cb_xpow, onetile);
    cb_pop_front(cb_exp_lxmd, onetile);
    cb_push_back(cb_correct_xpow, onetile);
    REL();
}

ALWI void power_and_recip_tile_to_cb(
    std::uint8_t cb_x,
    std::uint8_t cb_xpow,
    std::uint8_t cb_logx,
    std::uint8_t cb_decimal,
    std::uint8_t cb_exp_lxmd,
    std::uint8_t cb_recip_xpow,
    uint32_t p,
    bool p_is_negative) {
    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst0 = 0;

    // x^p
    ACQ();
    cb_wait_front(cb_x, onetile);
    cb_reserve_back(cb_xpow, onetile);

    copy_tile_init();
    copy_tile(cb_x, 0, dst0);

    power_tile_init();
    power_tile(dst0, p);

    if (p_is_negative) {
        recip_tile_init();
        recip_tile(dst0);
    }

    pack_tile(dst0, cb_xpow);

    cb_push_back(cb_xpow, onetile);
    REL();
    // We don't pop cb_x here.

    // log(x)
    ACQ();
    cb_reserve_back(cb_logx, onetile);

    copy_tile_init();
    copy_tile(cb_x, 0, dst0);

    log_tile_init();
    log_tile(dst0);

    pack_tile(dst0, cb_logx);

    cb_pop_front(cb_x, onetile);
    cb_push_back(cb_logx, onetile);
    REL();

    // exp(log(x) * decimal)
    ACQ();
    cb_wait_front(cb_logx, onetile);
    cb_reserve_back(cb_exp_lxmd, onetile);

    mul_tiles_init();
    mul_tiles(cb_logx, cb_decimal, 0, 0, dst0);

    exp_tile_init();
    exp_tile(dst0);

    pack_tile(dst0, cb_exp_lxmd);

    cb_pop_front(cb_logx, onetile);
    cb_push_back(cb_exp_lxmd, onetile);
    REL();

    // 1 / (x^p * exp(log(x) * decimal))(==1 / (x + decimal)^p)
    ACQ();
    cb_wait_front(cb_xpow, onetile);
    cb_wait_front(cb_exp_lxmd, onetile);
    cb_reserve_back(cb_recip_xpow, onetile);

    mul_tiles_init();
    mul_tiles(cb_xpow, cb_exp_lxmd, 0, 0, dst0);

    recip_tile_init();
    recip_tile(dst0);

    pack_tile(dst0, cb_recip_xpow);

    cb_pop_front(cb_xpow, onetile);
    cb_pop_front(cb_exp_lxmd, onetile);
    cb_push_back(cb_recip_xpow, onetile);
    REL();
}

}  // namespace ckernel
