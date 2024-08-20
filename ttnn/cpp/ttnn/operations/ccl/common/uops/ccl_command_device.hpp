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
constexpr auto build_from_args(std::size_t ct_arg_offset, std::size_t &rt_arg_idx) -> T {
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

template <typename T>
constexpr auto build_from_args(std::size_t &rt_arg_idx) -> Shape4D<T> {
    static_assert(sizeof(T) <= sizeof(uint32_t), "Shape4D doesn't support types larger than 4B.");
    return {
        get_arg_val<uint32_t>(rt_arg_idx++),
        get_arg_val<uint32_t>(rt_arg_idx++),
        get_arg_val<uint32_t>(rt_arg_idx++),
        get_arg_val<uint32_t>(rt_arg_idx++)
    };
}


} // namespace ccl
} // namespace ttnn
