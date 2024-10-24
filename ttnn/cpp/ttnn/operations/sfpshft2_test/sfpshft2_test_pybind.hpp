// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/sfpshft2_test/sfpshft2_test.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace sfpshft2_test {

void bind_sfpshft2_test_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::sfpshft2_test,
        "SFPSHFT2 test",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::sfpshft2_test)& self,
               const Tensor& input,
               const Tensor& output,
               uint8_t queue_id) -> Tensor {
                return self(queue_id, input, output);
            },
            py::arg("input"),
            py::arg("output"),
            py::kw_only(),
            py::arg("queue_id") = 0});
}

}  // namespace sfpshft2_test
}  // namespace operations
}  // namespace ttnn
