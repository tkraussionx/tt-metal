// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "tt_metal/host_api.hpp"

namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks slice_multi_core(const Tensor& a, Tensor& output, const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end, const std::optional<ttnn::Shape>& step);

}  // namespace ttnn::operations::data_movement::detail
