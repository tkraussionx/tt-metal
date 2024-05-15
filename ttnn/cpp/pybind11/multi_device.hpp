// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ttnn/multi_device.hpp"

namespace py = pybind11;

namespace ttnn {

namespace multi_device {

void py_module(py::module& module) {
    py::class_<DeviceMesh>(module, "DeviceMesh")
        .def(
            py::init<DeviceGrid, std::vector<int>, size_t>(),
            py::kw_only(),
            py::arg("device_grid"),
            py::arg("device_ids"),
            py::arg("l1_small_size"))
        .def("get_device", &ttnn::multi_device::DeviceMesh::get_device, py::return_value_policy::reference)
        .def("get_num_devices", &ttnn::multi_device::DeviceMesh::num_devices)
        .def("get_device_ids", &ttnn::multi_device::DeviceMesh::get_device_ids);

    module.def(
        "open_device_mesh",
        &open_device_mesh,
        py::kw_only(),
        py::arg("device_grid"),
        py::arg("device_ids"),
        py::arg("l1_small_size"));

    module.def("close_device_mesh", &close_device_mesh, py::arg("device_mesh"), py::kw_only());
    module.def("begin_trace_capture", &begin_trace_capture, py::arg("device_mesh"), py::arg("trace_buffer_size"), py::arg("cq_id") = 0);
    module.def("end_trace_capture", &end_trace_capture, py::arg("device_mesh"), py::arg("trace_id"), py::arg("cq_id") = 0);
    module.def("execute_trace", &execute_trace, py::arg("device_mesh"), py::arg("trace_id"), py::arg("cq_id") = 0, py::arg("blocking") = true);
    module.def("release_trace", &release_trace, py::arg("device_mesh"), py::arg("trace_id"));
    module.def("get_device_tensors", &get_device_tensors, py::arg("tensor"), py::kw_only());
    module.def("aggregate_as_tensor", &aggregate_as_tensor, py::arg("tensors"), py::kw_only());
}

}  // namespace multi_device

}  // namespace ttnn
