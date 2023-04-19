#include "tt_dnn/op_library/op.hpp"
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"

#include "tt_metal/host_api.hpp"
#include "constants.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {
class EltwiseUnaryOpMultiCore;

class EltwiseUnaryOpMultiCore : public Op {
    public:

        EltwiseUnaryOpMultiCore(const Tensor &a, UnaryOpType::Enum op_type);

        virtual ~EltwiseUnaryOpMultiCore();

    protected:
        // Op Specific params that are not tensors
        UnaryOpType::Enum op_type;

        void op_asserts();

        Tensor create_output();

        void create_op(const Tensor& output);

};

Tensor eltwise_unary_multi_core(const Tensor &a, UnaryOpType::Enum op_type);

}  // namespace tt_metal

}  // namespace tt
