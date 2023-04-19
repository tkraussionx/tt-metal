#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"

#include "tt_metal/host_api.hpp"
#include "constants.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {


Tensor eltwise_unary_single_core(const Tensor &a, UnaryOpType::Enum op_type, DataFormat data_format, MathFidelity math_fidelity) {

  // Figure out args for shapes
  // Figure out args for cbs
  // Figure out args for datamovement, compute kernels

  return op(args);
}




}  // namespace tt_metal

}  // namespace tt
