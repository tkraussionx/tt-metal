// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/cpp/pybind11/decorators.hpp"
#include "ttnn/operations/moreh_layernorms/moreh_layernorm/moreh_layernorm.hpp"
#include "ttnn/types.hpp"

namespace py = pybind11;

namespace ttnn::operations::moreh_layernorms {

void bind_moreh_layernorm_operation(py::module& module) {
    bind_registered_operation(
        module,
        ttnn::moreh_layernorm,
        R"doc(moreh_layernorm(input: ttnn.Tensor, normalized_dims: int, eps: float, gamma: Optional[ttnn.Tensor] = None, beta: Optional[ttnn.Tensor] = None, output: Optional[ttnn.Tensor] = None, mean: Optional[ttnn.Tensor] = None, rstd: Optional[ttnn.Tensor] = None, memory_config: Optional[ttnn.MemoryConfig] = None, compute_kernel_config: Optional[DeviceComputeKernelConfig] ) -> ttnn.Tensor

        Args:
            * :attr:`input` (ttnn.Tensor)
            * :attr:`normalized_dims` (uint32_t)
            * :attr:`eps` (float)

        Keyword args:
            * :attr:`gamma` (Optional[ttnn.Tensor]):
            * :attr:`beta` (Optional[ttnn.Tensor]):
            * :attr:`output` (Optional[ttnn.Tensor]):
            * :attr:`mean` (Optional[ttnn.Tensor]):
            * :attr:`rstd` (Optional[ttnn.Tensor]):
            * :attr:`memory_config`: the memory configuration of the output tensor. Default is input tensor memory config.
            * :attr:`compute_kernel_config`: the compute kernel configuration for the op. If not provided, the default configuration of the op is used.
        )doc",
        ttnn::pybind_overload_t{
            [](const decltype(ttnn::moreh_layernorm)& self,
            const ttnn::Tensor& input,
            uint32_t normalized_dims,
            float eps,
            const std::optional<const ttnn::Tensor> gamma,
            const std::optional<const ttnn::Tensor> beta,
            const std::optional<const ttnn::Tensor> output,
            const std::optional<const ttnn::Tensor> mean,
            const std::optional<const ttnn::Tensor> rstd,
            const std::optional<ttnn::MemoryConfig>& memory_config,
            const std::optional<const DeviceComputeKernelConfig> compute_kernel_config,
            const uint8_t& queue_id)
                -> std::tuple<optional<ttnn::Tensor>, optional<ttnn::Tensor>, optional<ttnn::Tensor>> { return self(queue_id, input, normalized_dims, eps, gamma, beta, output, mean, rstd, memory_config, compute_kernel_config); },
            py::arg("input"),
            py::arg("normalized_dims"),
            py::arg("eps"),
            py::kw_only(),
            py::arg("gamma").noconvert() = std::nullopt,
            py::arg("beta").noconvert() = std::nullopt,
            py::arg("output").noconvert() = std::nullopt,
            py::arg("mean").noconvert() = std::nullopt,
            py::arg("rstd").noconvert() = std::nullopt,
            py::arg("memory_config").noconvert() = std::nullopt,
            py::arg("compute_kernel_config").noconvert() = std::nullopt,
            py::arg("queue_id") = 0});
}

}  // namespace ttnn::operations::moreh_layernorms
