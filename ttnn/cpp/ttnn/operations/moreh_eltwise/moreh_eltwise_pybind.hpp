// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

#include "ttnn/operations/moreh_eltwise/moreh_binary/moreh_binary_pybind.hpp"
#include "ttnn/operations/moreh_eltwise/moreh_relu/moreh_relu_pybind.hpp"

namespace ttnn::operations::moreh_eltwise {

void py_module(py::module &module) {
  bind_moreh_relu(module);
  bind_moreh_binary(module);
}

} // namespace ttnn::operations::moreh_eltwise
