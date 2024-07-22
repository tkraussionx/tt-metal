// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>

#include "ttnn/operations/moreh_layernorms/moreh_layernorm/moreh_layernorm_pybind.hpp"

namespace ttnn::operations::moreh_layernorms {

void py_module(py::module& module) { bind_moreh_layernorm_operation(module); }

}  // namespace ttnn::operations::moreh_layernorms
