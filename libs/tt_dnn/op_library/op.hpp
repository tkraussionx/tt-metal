#include "tt_metal/host_api.hpp"
#include "constants.hpp"

#include "tensor/tensor.hpp"
using namespace tt::constants;

namespace tt {

namespace tt_metal {

class Op {

    private:

        static inline OpProfiler op_profiler = OpProfiler();

    public:

        // TODO: Move definition to .cpp
        Tensor run_op(){
            op_profiler.startProfiling(get_op_name());


            this->general_asserts();
            this->op_asserts();
            Tensor output = this->create_output();
            this->output_asserts(output);
            this->device = output.device();
            this->create_op(output);


            tt_metal::CompileProgram(device, program, profile_ops);

            tt_metal::ConfigureDeviceWithProgram(device, program);
            tt_metal::LaunchKernels(device, program); // Will there always be 1?

            tt_metal::LaunchKernels(device, program);

            op_profiler.stopProfiling(device, program, get_op_name(), get_op_meta_data());
            // output does not hold any data, contains pointer to buffer on device with the data
            return output;
        }

        //Pybind the following static functions to profile python functions for setting global settings per model.
        static void set_profiler_flag(bool do_profile)
        {
            op_profiler.set_profiler_flag(do_profile);
        }

        static void set_profiler_location(string profiler_log_location)
        {
            op_profiler.set_profiler_location(profiler_log_location);
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


        //Child class for returning a vector of meta data strings including parallelization strategy
        virtual vector<string> get_op_meta_data () = 0;

        //Child class has to define this by keeping a const string
        virtual string get_op_name() = 0;

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
