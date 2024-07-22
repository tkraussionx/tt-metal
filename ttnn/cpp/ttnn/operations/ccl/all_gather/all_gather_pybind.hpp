// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/ccl/all_gather/all_gather_op.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn {
namespace operations {
namespace ccl {

namespace detail {

template <typename ccl_operation_t>
void bind_all_gather(py::module& module, const ccl_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const ccl_operation_t& self,
               const ttnn::Tensor& input_tensor,
               const uint32_t dim,
               const uint32_t num_links,
               const std::optional<ttnn::MemoryConfig>& memory_config,
               const std::size_t num_workers,
               const std::size_t max_channel_size,
               const std::size_t buffers_per_channel
               ) -> ttnn::Tensor {
                return self(input_tensor, dim, num_links, memory_config, num_workers, max_channel_size, buffers_per_channel);
            },
            py::arg("input_tensor"),
            py::arg("dim"),
            py::kw_only(),
            py::arg("num_links") = 1,
            py::arg("memory_config") = std::nullopt,
            py::arg("num_workers") = 0,
            py::arg("max_channel_size") = 0,
            py::arg("buffers_per_channel") = 1});
}

}  // namespace detail


void py_bind_all_gather(py::module& module) {
    detail::bind_all_gather(
        module,
        ttnn::all_gather,
        R"doc(all_gather(input_tensor: ttnn.Tensor, dim: int, *, num_links: int = 1, memory_config: Optional[ttnn.MemoryConfig] = None, num_workers: int = 0, max_channel_size : int = 0, buffers_per_channel : int = 1) -> ttnn.Tensor

        Performs an all-gather operation on multi-device :attr:`input_tensor` across all devices.

        Args:
            * :attr:`input_tensor` (ttnn.Tensor): multi-device tensor
            * :attr:`dim` (int)

        Keyword Args:
            * :attr:`num_links` (int): Number of links to use for the all-gather operation.
            * :attr:`memory_config` (Optional[ttnn.MemoryConfig]): Memory configuration for the operation.
            * :attr:`num_workers` (Optional[int]): description
            * :attr:`max_channel_size` (Optional[int]): description
            * :attr:`buffers_per_channel` (Optional[int]): description

        Example:

            >>> tensor = ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16), device=device)
            >>> output = ttnn.all_gather(tensor, dim=0)

        )doc");
}

}  // namespace ccl
}  // namespace operations
}  // namespace ttnn
