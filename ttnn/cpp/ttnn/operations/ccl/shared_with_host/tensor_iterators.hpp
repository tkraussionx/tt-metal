// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/tensor_iterators_types.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <span>

namespace ttnn {

// TODO: Promote this to a common namespace as this is generally applicable.
namespace ccl {
namespace addrgen {






// The interface here still needs some work for generality and for a proper and robust design, really needs to have
// some sort of tensor iterator component as a member as well. That way there is no discrepency between precomputed
// page location and the one being passed as an arg.
//
// At the moment, this implementation is a bit of an inbetween from no precomputation and some precomputation and is
// likely throaway as an interface. However, this still enables an incremental step towards the desired design outcome
//
// The current interface problem exemplified
//
template <class PRECOMPUTE_PAGE_LOCATION_ADDR_GEN_T>
struct PageAddrGenWithPrecomputedLocationsBase {
    PageAddrGenWithPrecomputedLocationsBase() {}

    // Dual-use API for the time being.


    // Short term API to bridge between current non-precomputed address generators and address generators
    // with precompute. There are limitations with this API and should only be used if the user is very
    // aware of the limitations.
    constexpr test_shard_location_with_contig_t get_page_location_with_contiguous_pages_in_row_in_bank(std::size_t page_index) {
        return PRECOMPUTE_PAGE_LOCATION_ADDR_GEN_T::get_page_location_with_contiguous_pages_in_row_in_bank(page_index);
    }

    constexpr test_shard_location_t get_page_location(std::size_t page_index) {
        return PRECOMPUTE_PAGE_LOCATION_ADDR_GEN_T::get_page_location(page_index);
    }

    constexpr std::size_t get_num_precomputed_locations() const {
        return PRECOMPUTE_PAGE_LOCATION_ADDR_GEN_T::get_num_precomputed_locations();
    }

    // Mid term API:
    // - Longer term we should migrate this to an iterator based API, but that will require some
    //   instantiation with some tensor iterator concept, which we don't have today.
    constexpr test_shard_location_with_contig_t get_next_location() const {
        return PRECOMPUTE_PAGE_LOCATION_ADDR_GEN_T::get_next_entry();
    }

    constexpr void compute_and_store_location(std::size_t page_id) {
        return PRECOMPUTE_PAGE_LOCATION_ADDR_GEN_T::compute_and_store_location(page_id);
    }

    constexpr void advance() {
        return PRECOMPUTE_PAGE_LOCATION_ADDR_GEN_T::advance();
    }

    // Where to store dynamically precomputed addresses (but perhaps that's just a different mode)
    // ... For now we don't support dynamic prefetch and instead only support precomputed args that
    //     are passed in as kernel args
};


template <PrecomputeType PRECOMPUTE_TYPE, typename PAGE_ADDR_GEN_T>
struct PageAddrGenWithPrecomputedLocations {
    static_assert(sizeof(PAGE_ADDR_GEN_T) == 0, "PageAddrGenWithPrecomputedLocations should not be instantiated without concrete precompute type.");
};


template <typename PAGE_ADDR_GEN_T>
struct PageAddrGenWithPrecomputedLocations<PrecomputeType::NO_PRECOMPUTE, PAGE_ADDR_GEN_T> :
    public PAGE_ADDR_GEN_T,
    public PageAddrGenWithPrecomputedLocationsBase<PageAddrGenWithPrecomputedLocations<PrecomputeType::NO_PRECOMPUTE, PAGE_ADDR_GEN_T>> {

    template <typename... ADDRGEN_ARGS_T>
    PageAddrGenWithPrecomputedLocations(ADDRGEN_ARGS_T&&... args)
        : PAGE_ADDR_GEN_T(std::forward<ADDRGEN_ARGS_T>(args)...) {}

    constexpr test_shard_location_with_contig_t get_page_location_with_contiguous_pages_in_row_in_bank(std::size_t page_index) const {
        return PAGE_ADDR_GEN_T::get_page_location_with_contiguous_pages_in_row_in_bank(page_index);
    }

    constexpr test_shard_location_t get_page_location(std::size_t page_index) const {
        return PAGE_ADDR_GEN_T::get_page_location(page_index);
    }

    constexpr std::size_t get_num_precomputed_locations() const {
        return 0;
    }

    constexpr test_shard_location_with_contig_t get_next_location() const {
        return next_entry;
    }

    constexpr void compute_and_store_location(std::size_t page_id) {
        next_entry = this->get_page_location_with_contiguous_pages_in_row_in_bank(page_id);
    }

    void advance() {
        // do nothing
    }


    private:
    test_shard_location_with_contig_t next_entry;
};

template <typename PAGE_ADDR_GEN_T>
struct PageAddrGenWithPrecomputedLocations<PrecomputeType::FIXED_PRECOMPUTE_ONLY, PAGE_ADDR_GEN_T> :
    public PAGE_ADDR_GEN_T,
    public PageAddrGenWithPrecomputedLocationsBase<PageAddrGenWithPrecomputedLocations<PrecomputeType::FIXED_PRECOMPUTE_ONLY, PAGE_ADDR_GEN_T>> {

    template <typename... ADDRGEN_ARGS_T>
    PageAddrGenWithPrecomputedLocations(
        uint32_t *precomputed_locations_base_ptr,
        const std::size_t num_precomputed_locations,
        std::size_t precomputed_location_read_index,
        const InterpretMode location_interpret_mode,
        ADDRGEN_ARGS_T&&... args)
        : PAGE_ADDR_GEN_T(std::forward<ADDRGEN_ARGS_T>(args)...),
        read_ptr(precomputed_locations_base_ptr),
        num_precomputed_locations(num_precomputed_locations),
        precomputed_location_read_index(precomputed_location_read_index),
        num_words_per_precomputed_location(get_num_mem_words_per_precomputed_location(InterpretMode::CONTIGUOUS_PAGES_PER_LOCATION)),
        location_interpret_mode(location_interpret_mode)
        {}

    constexpr test_shard_location_with_contig_t get_page_location_with_contiguous_pages_in_row_in_bank(std::size_t page_index) {
        this->compute_and_store_location(page_index);
        auto const& next_location = get_next_location();
        this->advance();
        return {next_location.core_location, next_location.page_offset};
    }

    constexpr test_shard_location_t get_page_location(std::size_t page_index) {
        this->compute_and_store_location(page_index);
        auto const& next_location = get_next_location();
        this->advance();
        return next_location;
    }

    constexpr std::size_t get_num_precomputed_locations() const {
        return num_precomputed_locations;
    }

    constexpr test_shard_location_with_contig_t get_next_location() const {
        if (precomputed_location_read_index < num_precomputed_locations) {
            // Device compiles only up to c++17 at the moment so we are restricted from using span
            // return read_precomputed_location(std::span{read_ptr, num_words_per_precomputed_location}, location_interpret_mode);
            return read_precomputed_location(read_ptr, location_interpret_mode);
        } else {
            return next_entry;
        }
    }

    constexpr void compute_and_store_location(std::size_t page_id) {
        if (precomputed_location_read_index >= num_precomputed_locations) {
            next_entry = this->get_page_location_with_contiguous_pages_in_row_in_bank(page_id);
        }
        // else do nothing - the entry already exists
    }

    void advance() {
        if (precomputed_location_read_index < num_precomputed_locations) {
            precomputed_location_read_index++;
            read_ptr += num_words_per_precomputed_location;
        }
        // else we are past the end of the statically precomputed locations so we should
        // stop advancing because the rest of the locations will be dynamically generated
    }


    private:
    uint32_t *read_ptr;
    const std::size_t num_precomputed_locations;
    std::size_t precomputed_location_read_index;
    const std::size_t num_words_per_precomputed_location;
    const InterpretMode location_interpret_mode;
    test_shard_location_with_contig_t next_entry;
};


}
}
}
