#include "ll_buda/op_library/bcast/bcast_op.hpp"
#include "ll_buda/tensor/tensor.hpp"
#include "ll_buda/host_api.hpp"

#include "constants.hpp"

// TODO(AP): duplication
namespace bcast {

using namespace tt::ll_buda;
using namespace tt::constants;

// FIXME:copy pasted the args here from the kernel file,  we could refactor the HLK file
const char* get_reader_name(BcastOpDim::Enum bcast_dim, BcastOpParallelizationStrategy::Enum bcast_parallelization_strategy) {
		if (bcast_parallelization_strategy == BcastOpParallelizationStrategy::SINGLE_CORE) {
        if (bcast_dim == BcastOpDim::H) {
            return "kernels/dataflow/reader_bcast_h_8bank.cpp";
        } else if (bcast_dim == BcastOpDim::W) {
            return "kernels/dataflow/reader_bcast_w_8bank.cpp";
        } if (bcast_dim == BcastOpDim::HW) {
            return "kernels/dataflow/reader_bcast_hw_8bank.cpp";
        }
    }
    else {
        if (bcast_dim == BcastOpDim::H) {
            return "kernels/dataflow/reader_bcast_h_8bank_for_multi_core.cpp";
        } else if (bcast_dim == BcastOpDim::W) {
            return "kernels/dataflow/reader_bcast_w_8bank_for_multi_core.cpp";
        } if (bcast_dim == BcastOpDim::HW) {
            return "kernels/dataflow/reader_bcast_hw_8bank_for_multi_core.cpp";
        }
    }
    TT_ASSERT(false && "Unexpected bcast_dim!");
    return "";
}

const char* get_compute_name(BcastOpDim::Enum bcast_dim) {
    switch (bcast_dim) {
        case BcastOpDim::H:  return "kernels/compute/bcast_h.cpp";
        case BcastOpDim::W:  return "kernels/compute/bcast_w.cpp";
        case BcastOpDim::HW: return "kernels/compute/bcast_hw.cpp";
        default:           TT_ASSERT(false && "Unexpected bcast_dim!");
    }
    return "";
}

const char* get_math_to_op_define(BcastOpMath::Enum bcast_math) {
    switch (bcast_math) {
        case BcastOpMath::ADD:  return "add_tiles_bcast";
        case BcastOpMath::SUB:  return "sub_tiles_bcast";
        case BcastOpMath::MUL:  return "mul_tiles_bcast";
        default:           TT_ASSERT(false && "Unexpected bcast_math!");
    }
    return "";
}

void set_compute_kernel_defines(ComputeKernel * bcast_kernel, BcastOpMath::Enum bcast_math){
    bcast_kernel->add_define("BCAST_OP", get_math_to_op_define(bcast_math));
    return;
}

BcastOpParallelizationStrategy::Enum get_parallelization_strategy(const Tensor &a, BcastOpDim::Enum bcast_dim){
		int32_t num_tiles = a.volume() / TILE_HW;
		int32_t Ht = a.shape()[2] / TILE_HEIGHT;
		int32_t Wt = a.shape()[3] / TILE_WIDTH;

    if(Ht > 1 and bcast_dim == BcastOpDim::H){
        return BcastOpParallelizationStrategy::MULTI_CORE_H;
    }
    else if(Wt > 1 and bcast_dim == BcastOpDim::W){
        return BcastOpParallelizationStrategy::MULTI_CORE_W;
    }
    else if(num_tiles > 1 and bcast_dim == BcastOpDim::HW){
        return BcastOpParallelizationStrategy::MULTI_CORE_HW;
    }
    else{
        return BcastOpParallelizationStrategy::SINGLE_CORE;
    }
}

} // namespace bcast


using namespace tt::ll_buda;
using namespace tt::constants;
using u32 = std::uint32_t;


namespace tt {

namespace ll_buda {


Tensor bcast(const Tensor &a, const Tensor &b, BcastOpMath::Enum bcast_math, BcastOpDim::Enum bcast_dim) {
    switch (bcast::get_parallelization_strategy(a, bcast_dim)){
				case BcastOpParallelizationStrategy::MULTI_CORE_H:
						return bcast_multi_core_h(a, b, bcast_math, bcast_dim);
						break;
				case BcastOpParallelizationStrategy::MULTI_CORE_W:
						return bcast_multi_core_w(a, b, bcast_math, bcast_dim);
						break;
				case BcastOpParallelizationStrategy::MULTI_CORE_HW:
						return bcast_multi_core_hw(a, b, bcast_math, bcast_dim);
						break;
				case BcastOpParallelizationStrategy::SINGLE_CORE:
				default:
						return bcast_single_core(a, b, bcast_math, bcast_dim);
		}
}

}  // namespace ll_buda

}  // namespace tt
