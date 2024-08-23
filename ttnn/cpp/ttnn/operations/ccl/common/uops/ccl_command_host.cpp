// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command_host.hpp"

#include <ranges>
#include "tt_metal/common/assert.hpp"

namespace ttnn {
namespace ccl {
namespace cmd {


std::vector<uint32_t> add_ccl_command_to_args(CclCommand const& cmd ) {
    TT_ASSERT(cmd.worker_pages_per_slice > 0, "worker_pages_per_slice: {} must be greater than 0", cmd.worker_pages_per_slice);
    TT_ASSERT(cmd.tensor_slice_shape.w > 0, "tensor_slice_shape.w: {} must be greater than 0", cmd.tensor_slice_shape.w);
    TT_ASSERT(cmd.tensor_slice_shape.z > 0, "tensor_slice_shape.z: {} must be greater than 0", cmd.tensor_slice_shape.z);
    TT_ASSERT(cmd.tensor_slice_shape.y > 0, "tensor_slice_shape.y: {} must be greater than 0", cmd.tensor_slice_shape.y);
    TT_ASSERT(cmd.tensor_slice_shape.x > 0, "tensor_slice_shape.x: {} must be greater than 0", cmd.tensor_slice_shape.x);
    // TT_ASSERT(cmd.tensor_slice_offset.w < cmd.tensor_slice_shape.w);
    // TT_ASSERT(cmd.tensor_slice_offset.z < cmd.tensor_slice_shape.z);
    // TT_ASSERT(cmd.tensor_slice_offset.y < cmd.tensor_slice_shape.y);
    // TT_ASSERT(cmd.tensor_slice_offset.x < cmd.tensor_slice_shape.x);
    TT_ASSERT(cmd.worker_start_offset_in_slice.w < cmd.tensor_slice_shape.w, "worker_start_offset_in_slice.w: {} must be less than tensor slice shape {}", cmd.worker_start_offset_in_slice.w, cmd.tensor_slice_shape.w);
    TT_ASSERT(cmd.worker_start_offset_in_slice.z < cmd.tensor_slice_shape.z, "worker_start_offset_in_slice.w: {} must be less than tensor slice shape {}", cmd.worker_start_offset_in_slice.z, cmd.tensor_slice_shape.z);
    TT_ASSERT(cmd.worker_start_offset_in_slice.y < cmd.tensor_slice_shape.y, "worker_start_offset_in_slice.w: {} must be less than tensor slice shape {}", cmd.worker_start_offset_in_slice.y, cmd.tensor_slice_shape.y);
    TT_ASSERT(cmd.worker_start_offset_in_slice.x < cmd.tensor_slice_shape.x, "worker_start_offset_in_slice.w: {} must be less than tensor slice shape {}", cmd.worker_start_offset_in_slice.x, cmd.tensor_slice_shape.x);
    return {
        cmd.tensor_slice_shape.w,
        cmd.tensor_slice_shape.z,
        cmd.tensor_slice_shape.y,
        cmd.tensor_slice_shape.x,
        cmd.tensor_slice_offset.w,
        cmd.tensor_slice_offset.z,
        cmd.tensor_slice_offset.y,
        cmd.tensor_slice_offset.x,
        cmd.worker_start_offset_in_slice.w,
        cmd.worker_start_offset_in_slice.z,
        cmd.worker_start_offset_in_slice.y,
        cmd.worker_start_offset_in_slice.x,
        cmd.worker_pages_per_slice
    };
}


} // namespace cmd
} // namespace ccl
} // namespace ttnn
