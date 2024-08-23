// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command.hpp"

#include <cstdint>
#include <type_traits>

namespace ttnn {
namespace ccl {


template<typename T>
auto build_from_args(std::size_t &rt_arg_idx) -> T {
    static_assert(!std::is_same<T, T>::value, "This base template cannot be instantiated. Please provide a specialization.");
}
template<typename T>
constexpr std::size_t ct_args_consumed(std::size_t ct_arg_offset, std::size_t &rt_arg_idx) {
    static_assert(!std::is_same<T, T>::value, "This base template cannot be instantiated. Please provide a specialization.");
    return 0;
}
template<typename T>
constexpr std::size_t ct_args_consumed() {
    static_assert(!std::is_same<T, T>::value, "This base template cannot be instantiated. Please provide a specialization.");
    return 0;
}

template <>
auto build_from_args<Shape4D<uint32_t>>(std::size_t &rt_arg_idx) -> Shape4D<uint32_t> {
    // static_assert(sizeof(Shape4D<uint32_t>) <= sizeof(uint32_t), "Shape4D doesn't support types larger than 4B.");
    auto w = get_arg_val<uint32_t>(rt_arg_idx++);
    auto z = get_arg_val<uint32_t>(rt_arg_idx++);
    auto y = get_arg_val<uint32_t>(rt_arg_idx++);
    auto x = get_arg_val<uint32_t>(rt_arg_idx++);

    return Shape4D<uint32_t>{w, z, y, x};
}

namespace cmd {

CclCommand get_command(std::size_t &arg_idx) {
    CclCommand cmd;
    cmd.tensor_slice_shape = ttnn::ccl::build_from_args<decltype(cmd.tensor_slice_shape)>(arg_idx); // Should be RT
    cmd.tensor_slice_offset = ttnn::ccl::build_from_args<decltype(cmd.tensor_slice_offset)>(arg_idx); // Should be RT
    cmd.worker_start_offset_in_slice = ttnn::ccl::build_from_args<decltype(cmd.worker_start_offset_in_slice)>(arg_idx); // Should be RT
    cmd.worker_pages_per_slice = get_arg_val<uint32_t>(arg_idx++);
    return cmd;
}

} // namespace cmd

} // namespace ccl
} // namespace ttnn
