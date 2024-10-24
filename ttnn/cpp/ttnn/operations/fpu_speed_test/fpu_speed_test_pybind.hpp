// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/fpu_speed_test/fpu_speed_test.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace fpu_speed_test {

void bind_fpu_speed_test_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::fpu_speed_test,
        "fpu speed test",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::fpu_speed_test)& self,
               uint32_t num_tiles,
               bool fp32_dest_acc_en,
               const Tensor& dummy,
               uint8_t queue_id) -> Tensor {
                return self(queue_id, num_tiles, fp32_dest_acc_en, dummy);
            },
            py::arg("num_tiles"),
            py::arg("fp32_dest_acc_en"),
            py::kw_only(),
            py::arg("dummy"),
            py::arg("queue_id") = 0});
}

}  // namespace fpu_speed_test
}  // namespace operations
}  // namespace ttnn
