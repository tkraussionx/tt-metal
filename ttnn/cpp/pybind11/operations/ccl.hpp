// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../decorators.hpp"
#include "ttnn/operations/ccl.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace ccl {

void py_module(py::module& module) {
    ttnn::bind_registered_operation(
        module,
        ttnn::all_gather,
        R"doc(all_gather(input_tensor: ttnn.Tensor, dim: int, *, num_links: int = 1, memory_config: Optional[ttnn.MemoryConfig] = None) -> ttnn.Tensor

        Performs an all-gather operation on :attr:`input_tensor` across all devices.

        Args:
            * :attr:`input_tensor`
            * :attr:`dim`

        Keyword Args:
            * :attr:`num_links` (int): Number of links to use for the all-gather operation.
            * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.

        Example::

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.all_gather(tensor, 0)

        )doc",
        ttnn::pybind_arguments_t{
            py::arg("input_tensor"), py::arg("dim"), py::kw_only(), py::arg("num_links"), py::arg("memory_config") = std::nullopt});
}

}  // namespace ccl
}  // namespace operations
}  // namespace ttnn
