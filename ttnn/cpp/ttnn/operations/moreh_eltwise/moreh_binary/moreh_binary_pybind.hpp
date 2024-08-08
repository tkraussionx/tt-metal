// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/moreh_eltwise/moreh_binary/moreh_binary.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn::operations::moreh_eltwise {

void bind_moreh_binary(py::module &module) {
  bind_registered_operation(
      module, ttnn::moreh_binary, R"doc(moreh_binary)doc",

      // Add pybind overloads for the C++ APIs that should be exposed to python
      // There should be no logic here, just a call to `self` with the correct
      // arguments
      ttnn::pybind_overload_t{
          [](const decltype(ttnn::moreh_binary) &self,
             const ttnn::Tensor &input_tensor0,
             const ttnn::Tensor &input_tensor1,
             const ttnn::Tensor &output_tensor, const float scalar0,
             const float scalar1, const uint8_t program_selector,
             const uint8_t &queue_id) -> ttnn::Tensor {
            return self(queue_id, input_tensor0, input_tensor1, output_tensor,
                        scalar0, scalar1, program_selector);
          },
          py::arg("input_tensor0"), py::arg("input_tensor1"),
          py::arg("output_tensor"), py::arg("scalar0"), py::arg("scalar1"),
          py::arg("program_selector"), py::kw_only(), py::arg("queue_id") = 0});
}

} // namespace ttnn::operations::moreh_eltwise
