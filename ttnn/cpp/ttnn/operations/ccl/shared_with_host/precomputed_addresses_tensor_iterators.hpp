// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/tensor_iterators_types.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/tensor_iterators.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>

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
struct PageAddrGenWithPrecomputedLocationsBase : public PageToLocationLookupInterface<PageAddrGenWithPrecomputedLocationsBase<PRECOMPUTE_PAGE_LOCATION_ADDR_GEN_T>>{
    PageAddrGenWithPrecomputedLocationsBase(std::size_t buffer_base_address, std::size_t page_size) :
        PageToLocationLookupInterface<PageAddrGenWithPrecomputedLocationsBase<PRECOMPUTE_PAGE_LOCATION_ADDR_GEN_T>>(
            buffer_base_address, page_size) {}
    // Dual-use API for the time being.

    // Mid term API:
    // - Longer term we should migrate this to an iterator based API, but that will require some
    //   instantiation with some tensor iterator concept, which we don't have today.
    test_shard_location_with_contig_t get_next_location() {
        return static_cast<PRECOMPUTE_PAGE_LOCATION_ADDR_GEN_T const*>(this)->get_next_location_impl();
    }

    void compute_and_store_location(std::size_t page_id) {
        return static_cast<PRECOMPUTE_PAGE_LOCATION_ADDR_GEN_T*>(this)->compute_and_store_location_impl(page_id);
    }

    void advance() {
        static_cast<PRECOMPUTE_PAGE_LOCATION_ADDR_GEN_T*>(this)->advance_impl();
    }

    test_shard_location_with_contig_t get_page_location_with_contiguous_pages_in_row_in_bank(std::size_t global_page_id) {
        return static_cast<PRECOMPUTE_PAGE_LOCATION_ADDR_GEN_T *>(this)->get_page_location_with_contiguous_pages_in_row_in_bank_impl(global_page_id);
    }

    test_shard_location_t get_page_location(std::size_t global_page_id) {
        return static_cast<PRECOMPUTE_PAGE_LOCATION_ADDR_GEN_T *>(this)->get_page_location_impl(global_page_id);
    }

    // Where to store dynamically precomputed addresses (but perhaps that's just a different mode)
    // ... For now we don't support dynamic prefetch and instead only support precomputed args that
    //     are passed in as kernel args
};


template <typename PAGE_ADDR_GEN_T, PrecomputeType PRECOMPUTE_TYPE>
struct PageAddrGenWithPrecomputedLocations :
    public PageAddrGenWithPrecomputedLocationsBase<PageAddrGenWithPrecomputedLocations<PAGE_ADDR_GEN_T, PRECOMPUTE_TYPE>> {
    static_assert(sizeof(PAGE_ADDR_GEN_T) == 0, "PageAddrGenWithPrecomputedLocations should not be instantiated without concrete precompute type.");


    test_shard_location_with_contig_t get_page_location_with_contiguous_pages_in_row_in_bank_impl(std::size_t page_index) {
        return {};
    }

    test_shard_location_t get_page_location_impl(std::size_t page_index) {
        return {};
    }
};


template <typename PAGE_ADDR_GEN_T>
struct PageAddrGenWithPrecomputedLocations<PAGE_ADDR_GEN_T, PrecomputeType::NO_PRECOMPUTE > :
    public PageAddrGenWithPrecomputedLocationsBase<PageAddrGenWithPrecomputedLocations<PAGE_ADDR_GEN_T, PrecomputeType::NO_PRECOMPUTE>> {

    PageAddrGenWithPrecomputedLocations(PAGE_ADDR_GEN_T &&addrgen) :
        PageAddrGenWithPrecomputedLocationsBase<PageAddrGenWithPrecomputedLocations<PAGE_ADDR_GEN_T, PrecomputeType::NO_PRECOMPUTE>> (
            addrgen.get_buffer_base_address(), addrgen.get_page_size()),
        addrgen(std::move(addrgen)) {}

    test_shard_location_with_contig_t get_page_location_with_contiguous_pages_in_row_in_bank_impl(std::size_t page_index) {
        return addrgen.get_page_location_with_contiguous_pages_in_row_in_bank(page_index);
    }

    test_shard_location_t get_page_location_impl(std::size_t page_index) {
        return addrgen.get_page_location(page_index);
    }

    constexpr std::size_t get_num_precomputed_locations_impl() const {
        return 0;
    }

    test_shard_location_with_contig_t get_next_location_impl() const {
        return next_entry;
    }

    void compute_and_store_location_impl(std::size_t page_id) {
        next_entry = this->get_page_location_with_contiguous_pages_in_row_in_bank_impl(page_id);
    }

    void advance_impl() {
        // do nothing
    }

   private:
    PAGE_ADDR_GEN_T addrgen;

    test_shard_location_with_contig_t next_entry;
};

template <typename PAGE_ADDR_GEN_T>
struct PageAddrGenWithPrecomputedLocations<PAGE_ADDR_GEN_T, PrecomputeType::FIXED_PRECOMPUTE_ONLY> :
    public PageAddrGenWithPrecomputedLocationsBase<PageAddrGenWithPrecomputedLocations<PAGE_ADDR_GEN_T, PrecomputeType::FIXED_PRECOMPUTE_ONLY>> {

    PageAddrGenWithPrecomputedLocations(
        const std::size_t num_precomputed_locations,
        const InterpretMode location_interpret_mode,
        uint32_t *precomputed_locations_base_ptr,
        PAGE_ADDR_GEN_T && addrgen)
        : PageAddrGenWithPrecomputedLocationsBase<PageAddrGenWithPrecomputedLocations<PAGE_ADDR_GEN_T, PrecomputeType::FIXED_PRECOMPUTE_ONLY>>(addrgen.get_buffer_base_address(), addrgen.get_page_size()),
        addrgen(std::move(addrgen)),
        read_ptr(precomputed_locations_base_ptr),
        num_precomputed_locations(num_precomputed_locations),
        precomputed_location_read_index(0),
        num_words_per_precomputed_location(get_num_mem_words_per_precomputed_location(location_interpret_mode)),
        location_interpret_mode(location_interpret_mode)
        {}


    test_shard_location_with_contig_t get_page_location_with_contiguous_pages_in_row_in_bank_impl(std::size_t page_index) {
        this->compute_and_store_location_impl(page_index);
        test_shard_location_with_contig_t const& next_location = this->get_next_location_impl();
        this->advance_impl();
        return next_location;
    }

    test_shard_location_t get_page_location_impl_impl(std::size_t page_index) {
        this->compute_and_store_location_impl(page_index);
        test_shard_location_t const& next_location = this->get_next_location_impl();
        this->advance_impl();
        return next_location;
    }

    constexpr std::size_t get_num_precomputed_locations_impl() const {
        return num_precomputed_locations;
    }

    test_shard_location_with_contig_t get_next_location_impl() const {
        if (precomputed_location_read_index < num_precomputed_locations) {
            // DeviceZoneScopedN("hit");
            return read_precomputed_location(read_ptr, location_interpret_mode);
        } else {
            return next_entry;
        }
    }

    void compute_and_store_location_impl(std::size_t page_id) {
        if (precomputed_location_read_index >= num_precomputed_locations) {
            next_entry = addrgen.get_page_location_with_contiguous_pages_in_row_in_bank(page_id);
        }
        // else do nothing - the entry already exists
    }

    void advance_impl() {
        if (precomputed_location_read_index < num_precomputed_locations) {
            precomputed_location_read_index++;
            read_ptr += num_words_per_precomputed_location;
        }
        // else we are past the end of the statically precomputed locations so we should
        // stop advancing because the rest of the locations will be dynamically generated
    }


    private:
    PAGE_ADDR_GEN_T addrgen;

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
