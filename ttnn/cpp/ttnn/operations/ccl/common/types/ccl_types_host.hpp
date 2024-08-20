// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types.hpp"

#include <vector>

namespace ttnn {
namespace ccl {

template <typename T>
std::vector<uint32_t> emit_runtime_args(T const& args);
template <typename T>
std::vector<uint32_t> emit_compile_time(T const& args);


} // namespace ccl
} // namespace ttnn
