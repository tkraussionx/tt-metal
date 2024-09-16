// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/core_coord.h"

#include <cstdint>
#include <vector>
#include <optional>

namespace ttnn {
namespace ccl {
namespace reduce_scatter_detail {

enum class Direction {
    CLOCKWISE = 0,
    RIGHT = 0,

    COUNTER_CLOCKWISE = 1,
    LEFT = 1,

    UNASSIGNED
};

static_assert(Direction::CLOCKWISE == Direction::RIGHT, "Direction::CLOCKWISE == Direction::RIGHT not equal but expected to be for current design");
static_assert(Direction::COUNTER_CLOCKWISE == Direction::LEFT, "Direction::COUNTER_CLOCKWISE == Direction::LEFT not equal but expected to be for current design");

/*
 * Contains various attributes about a given worker
 */
struct WorkerAttributes {
    std::size_t link = std::numeric_limits<std::size_t>::max();
    std::size_t channel = std::numeric_limits<std::size_t>::max();
    Direction direction = Direction::UNASSIGNED;
    CoreCoord location_logical = {std::numeric_limits<std::size_t>::max(), std::numeric_limits<std::size_t>::max()};
    std::optional<std::size_t> associated_worker_index = std::nullopt;
    std::optional<CoreCoord> associated_worker_core_logical = std::nullopt;
};

struct WorkerTransferInfo {
    WorkerTransferInfo(
        std::vector<uint32_t> pages_per_full_chunk_per_worker, uint32_t num_links, uint32_t num_workers);

    uint32_t get_num_pages_per_full_chunk(WorkerAttributes const& worker_attrs) const;

    std::vector<uint32_t> pages_per_full_chunk_per_worker;
    uint32_t num_links;
    uint32_t num_workers;
};

} // namespace reduce_scatter_detail
} // namespace ccl
} // namespace ttnn
