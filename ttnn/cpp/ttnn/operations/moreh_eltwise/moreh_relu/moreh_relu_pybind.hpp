// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/moreh_eltwise/moreh_relu/moreh_relu.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn::operations::moreh_eltwise {

void bind_moreh_relu(py::module &module) {
  bind_registered_operation(
      module, ttnn::moreh_relu,
      R"doc(moreh_relu(input_tensor: ttnn.Tensor, output_tensor: ttnn.Tensor) -> ttnn.Tensor)doc",

      // Add pybind overloads for the C++ APIs that should be exposed to python
      // There should be no logic here, just a call to `self` with the correct
      // arguments
      ttnn::pybind_overload_t{
          [](const decltype(ttnn::moreh_relu) &self,
             const ttnn::Tensor &input_tensor,
             const ttnn::Tensor &output_tensor, const bool do_max_relu,
             const uint32_t max, const uint8_t &queue_id) -> ttnn::Tensor {
            return self(queue_id, input_tensor, output_tensor, do_max_relu, max);
          },
          py::arg("input_tensor"), py::arg("output_tensor"),
          py::arg("do_max_relu"), py::arg("max"), py::kw_only(),
          py::arg("queue_id") = 0});
}

} // namespace ttnn::operations::moreh_eltwise
