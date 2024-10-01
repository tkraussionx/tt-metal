// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


#include "indexed_fill_pybind.hpp"
#include "indexed_fill.hpp"
#include "ttnn/cpp/pybind11/decorators.hpp"

namespace ttnn::operations::data_movement {
namespace detail {

void bind_indexed_fill(pybind11::module& module) {
    auto doc = fmt::format(
        R"doc(

            Replaces batch of input in input_b denoted by batch_ids into input_a.

            Args:
                batch_id (ttnn.Tensor): the input tensor.
                input_tensor_a (ttnn.Tensor): the input tensor.
                input_tensor_b (ttnn.Tensor): the input tensor.

            Keyword Args:
                dim (int, optional): Dimension value. Defaults to `0`.
                memory_config (ttnn.MemoryConfig, optional): Memory configuration for the operation. Defaults to `None`.

            Returns:
                ttnn.Tensor: the output tensor.

            Example:
                >>> batch_id = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.UINT32)), device=device)
                >>> input_a = ttnn.to_device(ttnn.from_torch(torch.tensor((1, 2), dtype=torch.bfloat16)), device=device)
                >>> input_b = ttnn.to_device(ttnn.from_torch(torch.tensor((0, 1), dtype=torch.bfloat16)), device=device)
                >>> output = ttnn.indexed_fill(batch_id, tensor1, tensor2)
        )doc",
        ttnn::indexed_fill.base_name());
    bind_registered_operation(
    module,
    ttnn::indexed_fill,
    "Index fill Operation",
    ttnn::pybind_arguments_t{
        pybind11::arg("batch_ids"),
        pybind11::arg("input_tensor_a"),
        pybind11::arg("input_tensor_b"),
        pybind11::kw_only(),
        pybind11::arg("dim") = 0,
        pybind11::arg("memory_config") = std::nullopt});

}

}  // detail
} // namespace ttnn::operations::data_movement::detail
