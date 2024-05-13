// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/device.hpp"

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
        py::return_value_policy::reference);

    module.def("close_device", &ttnn::close_device, py::arg("device"), py::kw_only());

    module.def("enable_program_cache", &ttnn::enable_program_cache, py::arg("device"), py::kw_only());

    module.def("disable_and_clear_program_cache", &ttnn::disable_and_clear_program_cache, py::arg("device"), py::kw_only());

    module.def("begin_trace_capture", &ttnn::begin_trace_capture, py::arg("device"), py::arg("trace_buffer_size"), py::arg("cq_id") = 0);

    module.def("end_trace_capture", &ttnn::end_trace_capture, py::arg("device"), py::arg("cq_id") = 0);

    module.def("execute_trace", &ttnn::execute_trace, py::arg("device"), py::arg("cq_id") = 0, py::arg("blocking") = true);

    module.def("release_trace", &ttnn::release_trace, py::arg("device"), py::arg("cq_id") = 0);

}

}  // namespace device
}  // namespace ttnn
