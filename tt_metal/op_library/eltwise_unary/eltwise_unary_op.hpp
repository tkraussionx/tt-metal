#pragma once

#include "tt_metal/tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace tt_metal {

struct UnaryOpType {
    enum Enum { EXP = 0, RECIP = 1, GELU = 2, RELU = 3, SQRT = 4, SIGMOID = 5, LOG = 6, TANH = 7 };
    static const vector<Enum> all() { return { EXP, RECIP, GELU, RELU, SQRT, SIGMOID, LOG, TANH }; }
};

struct UnaryOpParallelizationStrategy {
    enum Enum { MULTI_CORE = 0, SINGLE_CORE = 1 };
    static const vector<Enum> all() { return { MULTI_CORE, SINGLE_CORE }; }
};

Tensor eltwise_unary (const Tensor &a, UnaryOpType::Enum op_type, bool profile_device=false);
Tensor eltwise_unary_single_core (const Tensor &a, UnaryOpType::Enum op_type, bool profile_device=false);
Tensor eltwise_unary_multi_core (const Tensor &a, UnaryOpType::Enum op_type, bool profile_device=false);

inline Tensor exp     (const Tensor &a, bool profile_device=false) { return eltwise_unary(a, UnaryOpType::EXP, profile_device); }
inline Tensor recip   (const Tensor &a, bool profile_device=false) { return eltwise_unary(a, UnaryOpType::RECIP, profile_device); }
inline Tensor gelu    (const Tensor &a, bool profile_device=false) { return eltwise_unary(a, UnaryOpType::GELU, profile_device); }
inline Tensor relu    (const Tensor &a, bool profile_device=false) { return eltwise_unary(a, UnaryOpType::RELU, profile_device); }
inline Tensor sqrt    (const Tensor &a, bool profile_device=false) { return eltwise_unary(a, UnaryOpType::SQRT, profile_device); }
inline Tensor sigmoid (const Tensor &a, bool profile_device=false) { return eltwise_unary(a, UnaryOpType::SIGMOID, profile_device); }
inline Tensor log     (const Tensor &a, bool profile_device=false) { return eltwise_unary(a, UnaryOpType::LOG, profile_device); }
inline Tensor tanh     (const Tensor &a, bool profile_device=false) { return eltwise_unary(a, UnaryOpType::TANH, profile_device); }

}  // namespace tt_metal

}  // namespace tt

namespace eltwise_unary_op_utils {
using namespace tt::tt_metal;

string get_op_name(UnaryOpType::Enum op_type);

void set_compute_kernel_defines(ComputeKernel * eltwise_unary_kernel, UnaryOpType::Enum op_type);

UnaryOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a);

} // namespace eltwise_unary_op_utils
