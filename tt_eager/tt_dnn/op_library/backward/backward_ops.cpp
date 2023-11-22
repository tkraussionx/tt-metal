// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/composite/composite_ops.hpp"
#include "tt_dnn/op_library/backward/backward_ops.hpp"
#include "tt_numpy/functions.hpp"
#include "tt_eager/tensor/tensor_utils.hpp"
#include "tt_dnn/op_library/math.hpp"

namespace tt {

namespace tt_metal {

//addalpha(input, other, alpha) = input + (alpha * other)
Tensor _bw_addalpha(const Tensor& input_a, const Tensor& input_b, float alpha, const MemoryConfig& output_mem_config) {
    Tensor result = add(mul_unary(input_b, alpha, output_mem_config), input_a, std::nullopt, output_mem_config);
    return result;
}
Tensor bw_addalpha(const Tensor& input_a, const Tensor& input_b, float alpha, const MemoryConfig& output_mem_config)
{
    return operation::decorate_as_composite(__func__, _bw_addalpha)(input_a, input_b, alpha, output_mem_config);
}

}//namespace tt_metal

}//namespace tt
