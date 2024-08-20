// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types_host.hpp"

#include <vector>
#include <string>

namespace tt {
namespace tt_metal {
class Tensor;
class Device;
} // namespace tt_metal
} // namespace tt

namespace ttnn {
namespace ccl {

using args_list_t = std::vector<uint32_t>;

template <typename T>
args_list_t emit_runtime_args(T const& args);
template <typename T>
args_list_t emit_compile_time(T const& args);

args_list_t emit_address_generator_runtime_args(tt::tt_metal::Tensor const& tensor);
args_list_t emit_address_generator_compile_time_args(tt::tt_metal::Tensor const& tensor);

struct ShardedAddrGenArgBuilder {
    static bool shard_grid_is_transposed(tt::tt_metal::Tensor const& t);
    static std::vector<uint32_t> emit_ct_args(tt::tt_metal::Tensor const& t);
    static std::vector<uint32_t> emit_rt_args(tt::tt_metal::Device const* d, tt::tt_metal::Tensor const& t);
    static void log_sharded_tensor_kernel_args(tt::tt_metal::Tensor const& t, std::string const& prefix);
};


} // namespace ccl
} // namespace ttnn
