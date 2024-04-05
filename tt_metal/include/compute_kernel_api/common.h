// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compute_kernel_api/common_globals.h"
#include "compute_kernel_api/reg_api.h"
#include "compute_kernel_api/pack.h"
#include "compute_kernel_api/unpack.h"
#include "compute_kernel_api/cb_api.h"


// JIT Build flow will set this as needed.
#ifndef COMMON_RT_ARGS_OFFSET
    #define COMMON_RT_ARGS_OFFSET 0
#endif

/**
 * Returns the address in L1 for a given runtime argument index for unique (per core) runtime arguments set via SetRuntimeArgs() API.
 *
 * Return value: Associated L1 address of given unique runtime argument index
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                    | Required |
 * |----------------|-------------------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | arg_idx        | Unique Runtime argument index                                           | uint32_t | 0 to 255                                       | True     |
 */
constexpr static uint32_t get_arg_addr(int arg_idx) {
    // args are 4B in size
    return TRISC_L1_ARG_BASE + (arg_idx << 2);
}

/**
 * Returns the address in L1 for a given runtime argument index for common (all cores) runtime arguments set via SetCommonRuntimeArgs() API.
 *
 * Return value: Associated L1 address of given common runtime argument index
 *
 * | Argument       | Description                                                             | Type     | Valid Range                                    | Required |
 * |----------------|-------------------------------------------------------------------------|----------|------------------------------------------------|----------|
 * | arg_idx        | Common Runtime argument index                                           | uint32_t | 0 to 255                                       | True     |
 */
constexpr static uint32_t get_common_arg_addr(int arg_idx) {
    // args are 4B in size
    return TRISC_L1_ARG_BASE + COMMON_RT_ARGS_OFFSET + (arg_idx << 2);
}

/**
 * Returns the value at a given runtime argument index for unique (per-core) runtime arguments set via SetRuntimeArgs() API.
 *
 * Return value: The value associated with the unique runtime argument index
 *
 * | Argument              | Description                                    | Type                  | Valid Range               | Required |
 * |-----------------------|------------------------------------------------|-----------------------|---------------------------|----------|
 * | arg_idx               | Unique Runtime argument index                  | uint32_t              | 0 to 255                  | True     |
 * | T (template argument) | Data type of the returned argument             | Any 4-byte sized type | N/A                       | True     |
 */
template <typename T>
FORCE_INLINE T get_arg_val(int arg_idx) {
    // only 4B args are supported (eg int32, uint32)
    static_assert("Error: only 4B args are supported" && sizeof(T) == 4);
    return *((volatile tt_l1_ptr T*)(get_arg_addr(arg_idx)));
}

/**
 * Returns the value at a given runtime argument index for common (all cores) runtime arguments set via SetCommonRuntimeArgs() API.
 *
 * Return value: The value associated with the common runtime argument index
 *
 * | Argument              | Description                                    | Type                  | Valid Range               | Required |
 * |-----------------------|------------------------------------------------|-----------------------|---------------------------|----------|
 * | arg_idx               | Common Runtime argument index                  | uint32_t              | 0 to 255                  | True     |
 * | T (template argument) | Data type of the returned argument             | Any 4-byte sized type | N/A                       | True     |
 */
template <typename T>
FORCE_INLINE T get_common_arg_val(int arg_idx) {
    // only 4B args are supported (eg int32, uint32)
    static_assert("Error: only 4B args are supported" && sizeof(T) == 4);
    return *((volatile tt_l1_ptr T*)(get_common_arg_addr(arg_idx)));
}
