#include "tt_metal/host_api.hpp"
#include "constants.hpp"

#include "tensor/tensor.hpp"
using namespace tt::constants;

namespace tt {

namespace tt_metal {

class Op {
    public:
        // TODO: Move definition to .cpp
        Tensor run_op(){
            this->general_asserts();
            this->op_asserts();
            Tensor output = this->create_output();
            this->output_asserts(output);
            this->device = output.device();
            this->create_op(output);

            tt_metal::CompileProgram(device, program);

            tt_metal::ConfigureDeviceWithProgram(device, program);
            tt_metal::LaunchKernels(device, program); // Will there always be 1?

            // output does not hold any data, contains pointer to buffer on device with the data
            return output;
        }

    protected:
        tt_metal::Program *program = nullptr;
        tt_metal::Device *device = nullptr;
        // dtype / fidelity
        // core grid
        std::vector<Tensor> tensor_inputs;
        // output // vector?

        Op() {
            program = new tt_metal::Program();
        }
        virtual ~Op() {
            if (program != nullptr) {
                delete program;
            }
        }

        void general_asserts() {
            tt_metal::Device *first_device = this->tensor_inputs[0].device();
            for (const auto& tensor: this->tensor_inputs) {
                TT_ASSERT(not tensor.on_host(), "Operand tensor needs to be on device!");
                TT_ASSERT(tensor.buffer() != nullptr, "Operand tensor needs to be allocated in a buffer on device!");
                TT_ASSERT(tensor.device() == first_device, "Operand tensors needs to be on same device!");
            }
        }
        virtual void op_asserts() = 0;

        virtual Tensor create_output() = 0;

        void output_asserts(const Tensor &output) {
            TT_ASSERT(not output.on_host(), "Operand tensor needs to be on device!");
            TT_ASSERT(output.buffer() != nullptr, "Output tensor needs to be allocated in a buffer on device!");
            TT_ASSERT(output.device() == this->tensor_inputs[0].device(), "Input and output tensor needs to be on same device!");
        }
        virtual void create_op(const Tensor& output) = 0;

};
}
}
