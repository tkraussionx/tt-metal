#pragma once

#include "ll_buda/tensor/tensor.hpp"
#include "ll_buda/host_api.hpp"

using namespace tt::ll_buda;

namespace tt {

namespace ll_buda {

struct BcastOpMath {
    enum Enum { ADD = 0, SUB = 1, MUL = 2 };
    static const vector<Enum> all() { return { ADD, SUB, MUL }; }
};

struct BcastOpDim {
    enum Enum { H = 0, W = 1, HW = 2 };
    static const vector<Enum> all() { return { H, W, HW }; }
};

// TODO: Accept parallelization
struct BcastOpParallelizationStrategy {
    enum Enum { MULTI_CORE_H = 0, MULTI_CORE_W = 1, MULTI_CORE_HW = 2, SINGLE_CORE = 3 };
    static const vector<Enum> all() { return { MULTI_CORE_H, MULTI_CORE_W, MULTI_CORE_HW, SINGLE_CORE }; }
};

Tensor bcast(const Tensor &a, const Tensor &b, BcastOpMath::Enum bcast_op, BcastOpDim::Enum bcast_dim);
Tensor bcast_single_core(const Tensor &a, const Tensor &b, BcastOpMath::Enum bcast_op, BcastOpDim::Enum bcast_dim);
Tensor bcast_multi_core_h(const Tensor &a, const Tensor &b, BcastOpMath::Enum bcast_op, BcastOpDim::Enum bcast_dim);
Tensor bcast_multi_core_w(const Tensor &a, const Tensor &b, BcastOpMath::Enum bcast_op, BcastOpDim::Enum bcast_dim);
Tensor bcast_multi_core_hw(const Tensor &a, const Tensor &b, BcastOpMath::Enum bcast_op, BcastOpDim::Enum bcast_dim);

// TODO(AP): make 9 of these?
inline Tensor bcast_add_h(const Tensor &a, const Tensor &b) { return bcast(a, b, BcastOpMath::ADD, BcastOpDim::H); }

}  // namespace ll_buda

}  // namespace tt

namespace bcast {

using namespace tt::ll_buda;

struct hlk_args_t {
    uint32_t B;
    uint32_t Ht;
    uint32_t Wt;
};

const char* get_reader_name(BcastOpDim::Enum bcast_dim, BcastOpParallelizationStrategy::Enum bcast_parallelization_strategy);

const char* get_compute_name(BcastOpDim::Enum bcast_dim);

const char* get_math_to_op_define(BcastOpMath::Enum bcast_math);

void set_compute_kernel_defines(ComputeKernel * bcast_kernel, BcastOpMath::Enum bcast_math);

BcastOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a);

} // namespace bcast
