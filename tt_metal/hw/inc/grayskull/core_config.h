// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

enum ProgrammableCoreType {
    TENSIX     = 0,
    COUNT      = 3, // for now, easier to keep structures shared across arches' the same size
};

enum class AddressableCoreType : uint8_t {
    TENSIX    = 0,
    ETH       = 1, // TODO: Make this accessible through the HAL and remove non-GS entries
    PCIE      = 2,
    DRAM      = 3,
    HARVESTED = 4,
    UNKNOWN   = 5,
    COUNT     = 6,
};

enum class TensixProcessorTypes : uint8_t {
    DM0    = 0,
    DM1    = 1,
    MATH0  = 2,
    MATH1  = 3,
    MATH2  = 4,
    COUNT  = 5
};

constexpr uint8_t MaxProcessorsPerCoreType = 5;
constexpr uint8_t NumTensixDispatchClasses = 3;
constexpr uint8_t noc_size_x = 13;
constexpr uint8_t noc_size_y = 12;
#define ALLOCATOR_ALIGNMENT 32
#define LOG_BASE_2_OF_ALLOCATOR_ALIGNMENT 5
