// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <vector>

#include "ttnn/cpp/ttnn/operations/ccl/reduce_scatter/host/reduce_scatter_common.hpp"
// #include "ttnn/tensor/tensor.hpp"
#include "tt_metal/common/base.hpp"

namespace ttnn {
namespace ccl {
namespace reduce_scatter_detail {

WorkerTransferInfo::WorkerTransferInfo(
    std::vector<uint32_t> pages_per_full_chunk_per_worker, uint32_t num_links, uint32_t num_workers) :
    pages_per_full_chunk_per_worker(pages_per_full_chunk_per_worker),
    num_links(num_links),
    num_workers(num_workers) {}

uint32_t WorkerTransferInfo::get_num_pages_per_full_chunk(WorkerAttributes const& worker_attrs) const {
    std::size_t index = worker_attrs.link * num_workers + worker_attrs.channel;
    TT_ASSERT(index < pages_per_full_chunk_per_worker.size(), "Index {} out of bounds for pages_per_full_chunk_per_worker of size {}", index, pages_per_full_chunk_per_worker.size());
    return pages_per_full_chunk_per_worker.at(index);
}


} // namespace reduce_scatter_detail
} // namespace ccl
} // namespace ttnn
