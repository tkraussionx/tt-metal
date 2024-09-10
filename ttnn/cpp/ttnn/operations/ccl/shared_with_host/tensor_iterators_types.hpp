// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include <type_traits>
namespace ttnn {

// TODO: Promote this to a common namespace as this is generally applicable.
namespace ccl {

namespace addrgen {
using noc_grid_index_t = std::uint8_t;


struct device_core_location_t {
    noc_grid_index_t noc_y;
    noc_grid_index_t noc_x;
};

struct test_shard_location_t {
    device_core_location_t core_location;
    std::uint32_t page_offset;
};

struct test_shard_location_with_contig_t {
    device_core_location_t core_location;
    std::uint32_t page_offset;
    std::uint32_t contig_pages_in_row;
};


/*
 * The `PrecomputedLocationInterpretMode` defines to any Page Address Generator that supports
 * precomputed locations, how to interpret the memory sequence of precomputed page locations.
 * Depending on the mode specified, the number of entries , and their layout in memory, to
 * describe each location may vary.
 *
 * This mode allows the caller to pass less information where possible. For example, if a
 * precomputed locations address generator is intended to be used for interleaved tensors,
 * then the number of contiguous pages present from each page ID would always be 1 and
 * therefore wouldn't need to be passed explicitly to the kernel.
 */
enum class InterpretMode : uint8_t {
    // number of contiguous pages at each page location is always 1 (would typically be used)
    // for interleaved tensors but may also be used for sharded tensors where the shard width
    // is a single tile
    SINGLE_PAGE_LOCATION,
    CONTIGUOUS_PAGES_PER_LOCATION
};

enum class PrecomputeType : uint8_t {
    // Just the plain address generator with no precompute
    NO_PRECOMPUTE,

    // In this mode the precomputed locations are passed to the
    // kernel and no additional (dynamic) location precomputation
    // is allowed
    FIXED_PRECOMPUTE_ONLY,

    // No precomputed locations are provided but the page address generator
    // is free to prefetch/precompute locations ahead of time during any
    // idle-time while waiting for other progress to be made
    DYNAMIC_PRECOMPUTE_ONLY,

    // Combination of both modes. After the statically precomputed addresses
    // are read in, the page address generator is also able to prefetch/precompute
    // page locations dynamically during idle time
    FIXED_AND_DYNAMIC_PRECOMPUTE
};

static constexpr std::size_t get_num_mem_words_per_precomputed_location(InterpretMode interpret_mode) {
    constexpr std::size_t words_per_noc_addr_field = 1; // noc_x and y are 16bits each shared in the same location
    constexpr std::size_t words_per_page_offset_in_bank_field = 1;
    constexpr std::size_t words_per_n_contiguous_pages_field = 1;

    if (interpret_mode == InterpretMode::SINGLE_PAGE_LOCATION) {
        return words_per_noc_addr_field + words_per_page_offset_in_bank_field;
    } else {
        return words_per_noc_addr_field + words_per_page_offset_in_bank_field + words_per_n_contiguous_pages_field;
    }
}

// Our device code currently only compiles to c++17 so we cant use span yet :(
// template <InterpretMode interpret_mode, std::size_t Extent = -1>//std::dynamic_extent>
// constexpr test_shard_location_with_contig_t read_precomputed_location_impl(std::span<uint32_t, Extent> kernel_args_location_span) {
template <InterpretMode interpret_mode>//std::dynamic_extent>
constexpr test_shard_location_with_contig_t read_precomputed_location_impl(uint32_t * kernel_args_location_span) {
    if (interpret_mode == InterpretMode::SINGLE_PAGE_LOCATION) {
        constexpr std::size_t n_words_per_entry = get_num_mem_words_per_precomputed_location(InterpretMode::SINGLE_PAGE_LOCATION);
        static_assert(n_words_per_entry == 2);
        return test_shard_location_with_contig_t{
            device_core_location_t{static_cast<noc_grid_index_t>(kernel_args_location_span[0] >> 16), static_cast<noc_grid_index_t>(kernel_args_location_span[0] & 0xFFFF)},
            kernel_args_location_span[1],
            1
        };
    } else if (interpret_mode == InterpretMode::CONTIGUOUS_PAGES_PER_LOCATION) {
        constexpr std::size_t n_words_per_entry = get_num_mem_words_per_precomputed_location(InterpretMode::CONTIGUOUS_PAGES_PER_LOCATION);
        static_assert(n_words_per_entry == 3);
        // DPRINT << "kernel_args_location_span[0]: " << (uint32_t)kernel_args_location_span[0] << "\n";
        // DPRINT << "kernel_args_location_span[1]: " << (uint32_t)kernel_args_location_span[1] << "\n";
        // DPRINT << "kernel_args_location_span[2]: " << (uint32_t)kernel_args_location_span[2] << "\n";
        return test_shard_location_with_contig_t{
            device_core_location_t{
                static_cast<noc_grid_index_t>(kernel_args_location_span[0] >> 16),
                static_cast<noc_grid_index_t>(kernel_args_location_span[0] & 0xFFFF)},
            kernel_args_location_span[1],
            kernel_args_location_span[2]};
    }
}


// template<std::size_t Extent = std::dynamic_extent>
// constexpr test_shard_location_with_contig_t read_precomputed_location(std::span<uint32_t, Extent> kernel_args_location_span, InterpretMode interpret_mode) {
constexpr test_shard_location_with_contig_t read_precomputed_location(uint32_t* kernel_args_location_span, InterpretMode interpret_mode) {
    if (interpret_mode == InterpretMode::SINGLE_PAGE_LOCATION) {
        // DPRINT << "single page loc interp\n";
        return read_precomputed_location_impl<InterpretMode::SINGLE_PAGE_LOCATION/*, Extent*/>(kernel_args_location_span);
    } else {
        // DPRINT << "contig loc interp\n";
        return read_precomputed_location_impl<InterpretMode::CONTIGUOUS_PAGES_PER_LOCATION/*, Extent*/>(kernel_args_location_span);
    }
}


}
}
}
