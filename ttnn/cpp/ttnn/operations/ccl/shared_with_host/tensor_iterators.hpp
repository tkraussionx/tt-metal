// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/tensor_iterators_types.hpp"

#include <cstddef>
#include <cstdint>
#include <utility>


namespace ttnn {

// TODO: Promote this to a common namespace as this is generally applicable.
namespace ccl {
namespace addrgen {


template <class IMPL_T>
struct PageToLocationLookupInterface {
    PageToLocationLookupInterface(std::size_t bank_base_address, std::size_t page_size) :
        bank_base_address(bank_base_address),
        page_size(page_size) {}

    test_shard_location_with_contig_t get_page_location_with_contiguous_pages_in_row_in_bank(std::size_t global_page_id) {
        auto const& result = static_cast<IMPL_T *>(this)->get_page_location_with_contiguous_pages_in_row_in_bank_impl(global_page_id);
        return result;
    }

    test_shard_location_t get_page_location(std::size_t global_page_id) {
        return static_cast<IMPL_T *>(this)->get_page_location_impl(global_page_id);
    }

    std::size_t get_buffer_base_address() { return bank_base_address; }
    std::size_t get_page_size()    { return page_size; }

    // private:
    public:
    std::size_t bank_base_address;
    const std::size_t page_size;
};



template <class PageLookupT>
struct PageAddressGeneratorInferface : PageToLocationLookupInterface<PageLookupT> {
    // PageAddressGeneratorInferface(PageLookupT &&preconstructed_lookup_object, uint32_t bank_base_address, uint32_t page_size) :
    //     PageToLocationLookupInterface<PageLookupT>(std::move(preconstructed_lookup_object))
    //     { }

    PageAddressGeneratorInferface(uint32_t bank_base_address, uint32_t page_size) :
        PageToLocationLookupInterface<PageLookupT>(bank_base_address,page_size)
        { }

};




}
}
}
