// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/complex/complex_ops.hpp"
#include "ttnn/operations/eltwise/complex_unary/device/complex_unary_op.hpp"

namespace ttnn::operations::complex_binary {

ComplexTensor _add(const QueueId queue_id, const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config) {
    return ComplexTensor({ ttnn::add(queue_id, input_a[0], input_b[0], std::nullopt, output_mem_config),
             ttnn::add(queue_id, input_a[1], input_b[1], std::nullopt, output_mem_config) });
}

ComplexTensor _sub(const QueueId queue_id, const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config) {
    return ComplexTensor({ ttnn::subtract(queue_id, input_a[0], input_b[0], std::nullopt, output_mem_config),
             ttnn::subtract(queue_id, input_a[1], input_b[1], std::nullopt, output_mem_config) });
}

ComplexTensor _mul(const QueueId queue_id, const ComplexTensor& ab, const ComplexTensor& cd,  const MemoryConfig& output_mem_config) {
    // (a + ib)*(c + id) = (ac - bd) + i(bc + ad)
    Tensor re_part = ttnn::subtract(queue_id,
        ttnn::multiply(queue_id, ab[0],cd[0],std::nullopt,output_mem_config),
        ttnn::multiply(queue_id, ab[1],cd[1],std::nullopt,output_mem_config),
        std::nullopt, output_mem_config);

    Tensor im_part = ttnn::add(queue_id,
        ttnn::multiply(queue_id, ab[0],cd[1],std::nullopt,output_mem_config),
        ttnn::multiply(queue_id, ab[1],cd[0],std::nullopt,output_mem_config),
        std::nullopt, output_mem_config);

    return ComplexTensor({ re_part, im_part });
}

ComplexTensor _div(const QueueId queue_id, const ComplexTensor& input_a, const ComplexTensor& input_b,  const MemoryConfig& output_mem_config) {
    return ttnn::operations::complex_binary::_mul(queue_id, input_a, ttnn::operations::complex_unary::_reciprocal(queue_id, input_b, output_mem_config ), output_mem_config  );
}

}  // namespace ttnn::operations::complex_binary
