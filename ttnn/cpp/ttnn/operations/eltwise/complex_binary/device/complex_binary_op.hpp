// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include "third_party/magic_enum/magic_enum.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/complex/complex_ops.hpp"

namespace ttnn::operations::complex_binary {

enum class ComplexBinaryOpType {
    ADD,
    SUB,
    MUL,
    DIV,
};

class ComplexTensor {
    private:
        std::array<Tensor,2> m_real_imag;

    public:

        ComplexTensor(std::array<Tensor,2> val): m_real_imag(val) {
            TT_ASSERT( m_real_imag[0].get_legacy_shape() == m_real_imag[1].get_legacy_shape() , "Tensor shapes of real and imag should be identical");
        }

        const Tensor& operator[](uint32_t index) const {
            return m_real_imag[index];
        }

        const Tensor& real() const {
            return m_real_imag[0];
        }

        const Tensor& imag() const {
            return m_real_imag[1];
        }

        void deallocate() {
            m_real_imag[0].deallocate();
            m_real_imag[1].deallocate();
        }
};

// OpHandler_complex_binary_type1 = get_function_complex_binary
ComplexTensor _add(const QueueId queue_id, const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config);
ComplexTensor _sub(const QueueId queue_id, const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config);
ComplexTensor _mul(const QueueId queue_id, const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config);
ComplexTensor _div(const QueueId queue_id, const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config);

template <ComplexBinaryOpType OpType>
struct OpHandler_complex_binary_type1;

template <>
struct OpHandler_complex_binary_type1<ComplexBinaryOpType::ADD> {
    static ComplexTensor handle(const QueueId queue_id, const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config ) {
        return _add(queue_id, input_a, input_b, output_mem_config);
    }
};

template <>
struct OpHandler_complex_binary_type1<ComplexBinaryOpType::SUB> {
    static ComplexTensor handle(const QueueId queue_id, const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config ) {
        return _sub(queue_id, input_a, input_b, output_mem_config);
    }
};

template <>
struct OpHandler_complex_binary_type1<ComplexBinaryOpType::MUL> {
    static ComplexTensor handle(const QueueId queue_id, const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config ) {
        return _mul(queue_id, input_a, input_b, output_mem_config);
    }
};

template <>
struct OpHandler_complex_binary_type1<ComplexBinaryOpType::DIV> {
    static ComplexTensor handle(const QueueId queue_id, const ComplexTensor& input_a, const ComplexTensor& input_b, const MemoryConfig& output_mem_config ) {
        return _div(queue_id, input_a, input_b, output_mem_config);
    }
};

template <ComplexBinaryOpType OpType>
auto get_function_complex_binary() {
    return &OpHandler_complex_binary_type1<OpType>::handle;
}

}  // namespace ttnn::operations::complex_binary
