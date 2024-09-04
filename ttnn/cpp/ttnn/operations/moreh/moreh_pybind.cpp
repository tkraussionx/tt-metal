// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_pybind.hpp"

#include "ttnn/operations/moreh/moreh_sgd/moreh_sgd_pybind.hpp"

namespace ttnn::operations::moreh {
void bind_moreh_operations(py::module &module) { moreh_sgd::bind_moreh_sgd_operation(module); }
}  // namespace ttnn::operations::moreh
