#include "tt_metal/host_api.hpp"
#include "constants.hpp"

#include "tensor/tensor.hpp"
using namespace tt::constants;

namespace tt {

namespace tt_metal {

class Op {

    private:
        // Have to initialize these statics in source file
        static bool profile_ops;
        static string profile_folder;

    public:
        // TODO: Move definition to .cpp
        Tensor run_op(){
            //Run time check, on host side no that costly
            if (profile_ops)
            {
                op_profiler.markStart(get_op_name());
                op_profiler.setOutputDir(perf_folder + get_op_name());
                increment_call_count();
            }
            this->general_asserts();
            this->op_asserts();
            Tensor output = this->create_output();
            this->output_asserts(output);
            this->device = output.device();
            this->create_op(output);

            tt_metal::CompileProgram(device, program);

            tt_metal::ConfigureDeviceWithProgram(device, program);
            tt_metal::LaunchKernels(device, program); // Will there always be 1?

            tt_metal::SetProfilerDir(perf_folder + "/" + get_call_count());
            tt_metal::LaunchKernels(device, program);
            tt_metal::DumpDeviceProfileResults(device, program);


            if (profile_ops)
            {
                op_profiler.markStop(get_op_name());
                op_profiler.dumpHostResults(get_call_count() + "-" + get_op_meta_data());
            }
            // output does not hold any data, contains pointer to buffer on device with the data
            return output;
        }


        //Pybind to a profile python function for setting global settings per model.
        void set_profiler_settings(bool do_profile, string profile_folder_path)
        {
            profile_ops = do_profile;
            profile_folder = profile_folder_path;
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

        //Child class has to define this by keeping a private static int and increment it
        virtual uint32_t increment_call_count() = 0;

        //Child class has to define this to get call count
        virtual uint32_t get_call_count() = 0;

        //Child class has to define hyphen separated meta data including parallelization strategy
        virtual string get_op_meta_data() = 0;

        //Child class has to define this by keeping a private const string
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
