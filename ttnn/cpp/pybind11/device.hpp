// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/device.hpp"
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"

namespace py = pybind11;

namespace ttnn {
namespace device {
void py_module(py::module& module) {
    module.def(
        "open_device",
        &ttnn::open_device,
        py::kw_only(),
        py::arg("device_id"),
        py::arg("l1_small_size"),
        py::arg("trace_region_size"),
        py::arg("dispatch_core_type"),
        py::return_value_policy::reference);

    module.def("close_device", &ttnn::close_device, py::arg("device"), py::kw_only());

    module.def("enable_program_cache", &ttnn::enable_program_cache, py::arg("device"), py::kw_only());

    module.def("disable_and_clear_program_cache", &ttnn::disable_and_clear_program_cache, py::arg("device"), py::kw_only());

    module.def("SetDefaultDevice", &ttnn::operations::experimental::auto_format::AutoFormat::SetDefaultDevice, R"doc(
        Sets the default device to use for ops when inputs aren't on device.

        +------------------+------------------------+-----------------------+-------------+----------+
        | Argument         | Description            | Data type             | Valid range | Required |
        +==================+========================+=======================+=============+==========+
        | device           | TT Device to use       | tt_lib.device.Device  |             | Yes      |
        +------------------+------------------------+-----------------------+-------------+----------+
    )doc");

    module.def("GetDefaultDevice", &ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice, R"doc(
        Gets the default device to use for ops when inputs aren't on device.
    )doc");

    module.def(
        "format_input_tensor",
        &ttnn::operations::experimental::auto_format::AutoFormat::format_input_tensor,
        py::arg("input").noconvert(),
        py::arg("device").noconvert(),
        py::arg("padded_shape"),
        py::arg("pad_value"),
        py::arg("target_layout").noconvert(),
        py::arg("target_mem_config").noconvert() = std::nullopt,
        R"doc(
            Formats tensor to target layout and pads to padded shape
        )doc");
    module.def(
        "format_output_tensor",
        &ttnn::operations::experimental::auto_format::AutoFormat::format_output_tensor,
        py::arg("output").noconvert(),
        py::arg("shape"),
        py::arg("device").noconvert(),
        py::arg("target_layout").noconvert(),
        py::arg("target_mem_config").noconvert() = std::nullopt,
        R"doc(
            Formats tensor to target layout and unpads to shape
        )doc");
    module.def(
        "pad_to_tile_shape",
        [](const std::array<uint32_t, 4>& unpadded_shape,
           bool pad_c = false,
           bool pad_n = false,
           bool pad_h = true,
           bool pad_w = true) -> Shape {
            return Shape(ttnn::operations::experimental::auto_format::AutoFormat::pad_to_tile_shape(unpadded_shape, pad_c, pad_n, pad_h, pad_w));
        },
        R"doc(
            Returns shape padded to tile shape
        )doc");

}

}  // namespace device
}  // namespace ttnn
