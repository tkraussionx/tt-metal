// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/eltwise_unary/layernorm.h"
#include "compute_kernel_api/eltwise_unary/typecast.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/compute/moreh_common.hpp"

ALWI bool need_to_do_mask_h(uint32_t w_idx, uint32_t origin_num_h_tiles, uint32_t origin_num_w_tiles) {
    return ((w_idx / origin_num_w_tiles) + 1) % origin_num_h_tiles == 0;
}

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t num_rows_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t origin_H = get_compile_time_arg_val(1);
    constexpr uint32_t origin_W = get_compile_time_arg_val(2);
    constexpr uint32_t num_inner = get_compile_time_arg_val(3);
    constexpr uint32_t block_size = get_compile_time_arg_val(4);
    constexpr bool gamma_has_value = get_compile_time_arg_val(5) == 1;
    constexpr bool beta_has_value = get_compile_time_arg_val(6) == 1;
    constexpr bool mean_has_value = get_compile_time_arg_val(7) == 1;
    constexpr bool rstd_has_value = get_compile_time_arg_val(8) == 1;
    constexpr bool is_lastdim_layernorm = get_compile_time_arg_val(9) == 1;
    constexpr bool is_groupnorm = get_compile_time_arg_val(10) == 1;
    constexpr uint32_t scaler = get_compile_time_arg_val(11);
    constexpr uint32_t eps = get_compile_time_arg_val(12);

    binary_op_init_common(tt::CB::c_in0, tt::CB::c_in0);

    constexpr auto cb_x = tt::CB::c_in0;       // input
    constexpr auto cb_scaler = tt::CB::c_in1;  // scaler
    constexpr auto cb_eps = tt::CB::c_in2;     // epsilon
    constexpr auto cb_gamma = tt::CB::c_in3;   // gamma
    constexpr auto cb_beta = tt::CB::c_in4;    // beta
    constexpr auto cb_mask_h = tt::CB::c_in5;  // mask_h
    constexpr auto cb_mask_w = tt::CB::c_in6;  // mask_w

    constexpr auto cb_out = tt::CB::c_out0;   // output
    constexpr auto cb_mean = tt::CB::c_out1;  // mean
    constexpr auto cb_rstd = tt::CB::c_out2;  // rstd

    constexpr auto cb_ex = tt::CB::c_intermed0;          // E[x]
    constexpr auto cb_xmm = tt::CB::c_intermed1;         // x - E[x]
    constexpr auto cb_xmm2 = tt::CB::c_intermed2;        // (x - E[x])^2
    constexpr auto cb_xmm2sum = tt::CB::c_intermed3;     // Sum[(x - E[x])^2]
    constexpr auto cb_var = tt::CB::c_intermed4;         // E[(x - E[x])^2] = Var[x]
    constexpr auto cb_recip_std = tt::CB::c_intermed5;   // 1.0/(sqrt(Var[x] + eps))
    constexpr auto cb_gamma_beta = tt::CB::c_intermed6;  // p * gamm + beta
    constexpr auto cb_xsum = tt::CB::c_intermed7;        // Sum[x]

    constexpr uint32_t onetile = 1;

    cb_wait_front(cb_scaler, onetile);  // comes from the reader
    cb_wait_front(cb_eps, onetile);     // comes from the reader

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t TILE_W = 32;

    constexpr bool do_mask_h = (origin_H % TILE_H) != 0 && !is_lastdim_layernorm;
    constexpr bool do_mask_w = (origin_W % TILE_W) != 0;

    if (do_mask_h) {
        cb_wait_front(cb_mask_h, onetile);
    }
    if (do_mask_w) {
        cb_wait_front(cb_mask_w, onetile);
    }

    constexpr uint32_t dst0 = 0;
    constexpr uint32_t dst1 = 1;
    constexpr uint32_t dst2 = 2;
    constexpr uint32_t first_tile = 0;

    constexpr uint32_t origin_Ht = (origin_H + TILE_H - 1) / TILE_H;
    constexpr uint32_t origin_Wt = (origin_W + TILE_W - 1) / TILE_W;

    for (uint32_t outer_idx = 0; outer_idx < num_rows_per_core; outer_idx++) {
        /*
         * Sum[x]
         * cb_xsum
         */
        cb_wait_front(cb_x, num_inner);

        tile_regs_acquire();
        cb_reserve_back(cb_xsum, onetile);

        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            for (uint32_t j = 0; j < block_size; j++) {
                const uint32_t w_idx = inner_idx + j;

                copy_tile_init_with_dt(cb_x);
                copy_tile(cb_x, w_idx, dst0);  // input

                if (do_mask_h && need_to_do_mask_h(w_idx, origin_Ht, origin_Wt)) {
                    copy_tile_init_with_dt(cb_mask_h);
                    copy_tile(cb_mask_h, first_tile, dst1);  // mask_h

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }

                if (do_mask_w && ((w_idx + 1) % origin_Wt == 0)) {
                    copy_tile_init_with_dt(cb_mask_w);
                    copy_tile(cb_mask_w, first_tile, dst1);  // mask_w

                    mask_tile_init();
                    mask_tile(dst0, dst1);
                }

                // accumulate dst0 in dst2
                layernorm_acc_tile_init();
                layernorm_acc_tile(dst0, w_idx == 0);
            } // block_size loop
        } // num_inner loop

        // typecast_tile_init();
        // typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(dst2);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst2, cb_xsum);

        cb_push_back(cb_xsum, onetile);
        tile_regs_release();
        // We don't pop cb_x until we compute xmm.

        /*
         * E[x]
         * cb_ex
         */
        tile_regs_acquire();
        cb_wait_front(cb_xsum, onetile);
        cb_reserve_back(cb_ex, onetile);
        if (mean_has_value) {
            cb_reserve_back(cb_mean, onetile);
        }

        // reduce_init_delta_with_dt<false>(cb_ex, cb_xsum, cb_scaler);
        // reduce_tile(cb_xsum, cb_scaler, first_tile, first_tile, dst0);
        // reduce_revert_delta(cb_ex);
        copy_tile_init_with_dt(cb_xsum);
        copy_tile(cb_xsum, first_tile, dst0);
        layernorm_reduce_sum_w_tile_init();
        layernorm_reduce_sum_w_tile(dst0, scaler);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_ex);
        if (mean_has_value) {
            pack_tile_with_dt(dst0, cb_mean);
        }

        cb_pop_front(cb_xsum, onetile);
        cb_push_back(cb_ex, onetile);
        if (mean_has_value) {
            cb_push_back(cb_mean, onetile);
        }
        tile_regs_release();

        cb_wait_front(cb_ex, onetile);
        // We don't pop cb_ex here.

        /*
         * x - E[x]
         * xmm
         */
        cb_reserve_back(cb_xmm, num_inner);
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            for (uint32_t j = 0; j < block_size; j++) {
                const uint32_t w_idx = inner_idx + j;
                tile_regs_acquire();

                copy_tile_init_with_dt(cb_x);
                copy_tile(cb_x, w_idx, dst0);
                copy_tile_init_with_dt(cb_ex);
                copy_tile(cb_ex, first_tile, dst1);
                layernorm_sub_tiles_init();
                layernorm_sub_tiles(dst0);

                if (do_mask_h || do_mask_w) {
                    if (do_mask_h && need_to_do_mask_h(w_idx, origin_Ht, origin_Wt)) {
                        copy_tile_init_with_dt(cb_mask_h);
                        copy_tile(cb_mask_h, first_tile, dst1);  // mask_h

                        mask_tile_init();
                        mask_tile(dst0, dst1);
                    }
                    if (do_mask_w && (w_idx + 1) % origin_Wt == 0) {
                        copy_tile_init_with_dt(cb_mask_w);
                        copy_tile(cb_mask_w, first_tile, dst1);  // mask_w

                        mask_tile_init();
                        mask_tile(dst0, dst1);
                    }
                }
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(dst0, cb_xmm);

                cb_push_back(cb_xmm, onetile);
                tile_regs_release();
            }  // block_size loop
        }  // num_inner loop
        cb_pop_front(cb_ex, onetile);
        cb_pop_front(cb_x, num_inner);

        /*
         * Sum[(x - E[x])^2]
         * cb_xmm2sum
         */
        tile_regs_acquire();
        cb_wait_front(cb_xmm, num_inner);
        cb_reserve_back(cb_xmm2sum, onetile);

        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx++) {
            copy_tile_init_with_dt(cb_xmm);
            copy_tile(cb_xmm, inner_idx, dst0);

            // accumulate dst0 squared in dst1
            layernorm_sq_acc_tile_init();
            layernorm_sq_acc_tile(dst0, inner_idx == 0);
        }

        // typecast_tile_init();
        // typecast_tile<(uint32_t)DataFormat::Float32, (uint32_t)DataFormat::Float16_b>(dst1);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst1, cb_xmm2sum);

        cb_push_back(cb_xmm2sum, onetile);
        tile_regs_release();
        // We don't pop cb_xmm here.

        /*
         * E[(x-E[x])^2] = Var[x]
         * cb_var
         */
        tile_regs_acquire();
        cb_wait_front(cb_xmm2sum, onetile);
        cb_reserve_back(cb_var, onetile);

        // reduce_init_delta_with_dt<false>(cb_var, cb_xmm2sum, cb_scaler);
        // reduce_tile(cb_xmm2sum, cb_scaler, first_tile, first_tile, dst0);
        // reduce_revert_delta(cb_var);
        copy_tile_init_with_dt(cb_xmm2sum);
        copy_tile(cb_xmm2sum, first_tile, dst0);
        layernorm_reduce_sum_w_tile_init();
        layernorm_reduce_sum_w_tile(dst0, scaler);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_var);

        cb_pop_front(cb_xmm2sum, onetile);
        cb_push_back(cb_var, onetile);
        tile_regs_release();

        /*
         * rstd = 1.0 / (sqrt(E[(x-E[x])^2] + eps))
         * cb_recip_std
         */
        cb_wait_front(cb_var, onetile);
        cb_reserve_back(cb_recip_std, onetile);
        if (rstd_has_value) {
            cb_reserve_back(cb_rstd, onetile);
        }

        tile_regs_acquire();
        copy_tile_init_with_dt(cb_var);
        copy_tile(cb_var, first_tile, dst0);
        layernorm_rsqrt_tile_init();
        layernorm_rsqrt_tile(dst0, eps);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile_with_dt(dst0, cb_recip_std);
        if (rstd_has_value) {
            pack_tile_with_dt(dst0, cb_rstd);
        }

        cb_pop_front(cb_var, onetile);
        cb_push_back(cb_recip_std, onetile);
        if (rstd_has_value) {
            cb_push_back(cb_rstd, onetile);
        }
        tile_regs_release();

        cb_wait_front(cb_recip_std, onetile);

        /*
         * (x - E[x]) * rstd * gamma + beta
         * cb_out
         */
        constexpr auto cb_gamma_beta_or_out = (gamma_has_value || beta_has_value) ? cb_gamma_beta : cb_out;
        for (uint32_t inner_idx = 0; inner_idx < num_inner; inner_idx += block_size) {
            cb_reserve_back(cb_gamma_beta_or_out, block_size);
            for (uint32_t j = 0; j < block_size; j++) {
                tile_regs_acquire();
                if (is_lastdim_layernorm) {
                    mul_bcast_cols_init_short_with_dt(cb_xmm, cb_recip_std);
                    mul_tiles_bcast_cols(cb_xmm, cb_recip_std, inner_idx + j, first_tile, j);
                } else {
                    mul_tiles_bcast_scalar_init_short_with_dt(cb_xmm, cb_recip_std);
                    mul_tiles_bcast_scalar(cb_xmm, cb_recip_std, inner_idx + j, first_tile, j);
                }
                tile_regs_commit();

                tile_regs_wait();
                pack_tile_with_dt(j, cb_gamma_beta_or_out);
                tile_regs_release();
            }  // block_size loop
            cb_push_back(cb_gamma_beta_or_out, block_size);

            // * gamma
            if (gamma_has_value) {
                constexpr auto cb_outg = beta_has_value ? cb_gamma_beta : cb_out;
                cb_wait_front(cb_gamma_beta_or_out, block_size);
                cb_wait_front(cb_gamma, block_size);
                cb_reserve_back(cb_outg, block_size);
                for (uint32_t j = 0; j < block_size; j++) {
                    tile_regs_acquire();
                    if (is_groupnorm) {
                        mul_tiles_bcast_scalar_init_short_with_dt(cb_gamma_beta_or_out, cb_gamma);
                        mul_tiles_bcast_scalar(cb_gamma_beta_or_out, cb_gamma, j, j, j);
                    } else {
                        if (is_lastdim_layernorm) {
                            mul_bcast_rows_init_short_with_dt(cb_gamma_beta_or_out, cb_gamma);
                            mul_tiles_bcast_rows(cb_gamma_beta_or_out, cb_gamma, j, j, j);
                        } else {
                            mul_tiles_init_with_dt(cb_gamma_beta_or_out, cb_gamma);
                            mul_tiles(cb_gamma_beta_or_out, cb_gamma, j, j, j);
                        }
                    }
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(j, cb_outg);
                    tile_regs_release();
                }  // block_size loop
                cb_pop_front(cb_gamma_beta_or_out, block_size);
                cb_pop_front(cb_gamma, block_size);
                cb_push_back(cb_outg, block_size);
            }  // if (gamma_has_value)

            // + beta
            if (beta_has_value) {
                cb_wait_front(cb_gamma_beta, block_size);
                cb_wait_front(cb_beta, block_size);
                cb_reserve_back(cb_out, block_size);
                for (uint32_t j = 0; j < block_size; j++) {
                    tile_regs_acquire();
                    if (is_groupnorm) {
                        add_bcast_scalar_init_short_with_dt(cb_gamma_beta, cb_beta);
                        add_tiles_bcast_scalar(cb_gamma_beta, cb_beta, j, j, j);
                    } else {
                        if (is_lastdim_layernorm) {
                            add_bcast_rows_init_short_with_dt(cb_gamma_beta, cb_beta);
                            add_tiles_bcast_rows(cb_gamma_beta, cb_beta, j, j, j);
                        } else {
                            add_tiles_init_with_dt(cb_gamma_beta, cb_beta);
                            add_tiles(cb_gamma_beta, cb_beta, j, j, j);
                        }
                    }
                    tile_regs_commit();

                    tile_regs_wait();
                    pack_tile_with_dt(j, cb_out);
                    tile_regs_release();
                }  // block_size loop
                cb_pop_front(cb_gamma_beta, block_size);
                cb_pop_front(cb_beta, block_size);
                cb_push_back(cb_out, block_size);
            }  // if (beta_has_value)
        } // num_inner loop
        cb_pop_front(cb_recip_std, onetile);
        cb_pop_front(cb_xmm, num_inner);
    }  // num_rows_per_core loop
    cb_pop_front(cb_scaler, onetile);
    cb_pop_front(cb_eps, onetile);

    if (do_mask_h) {
        cb_pop_front(cb_mask_h, onetile);
    }
    if (do_mask_w) {
        cb_pop_front(cb_mask_w, onetile);
    }
}  // void MAIN
}  // namespace NAMESPACE
