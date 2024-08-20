// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command.hpp"

#include <ranges>


namespace ttnn {
namespace ccl {
namespace cmd {

struct CclCommand {
    Shape4D<uint32_t> tensor_slice_shape;
    Shape4D<uint32_t> worker_start_offset_in_slice;
    uint32_t worker_pages_per_slice;
};

std::vector<uint32_t> add_ccl_command_to_args(CclCommand const& cmd);




} // namespace cmd
} // namespace ccl
} // namespace ttnn
