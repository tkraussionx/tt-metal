#include "tt_dnn/op_library/fill_rm/fill_rm_op.hpp"
#include "common/test_tiles.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

using namespace tt::constants;
using u32 = uint32_t;

namespace tt {

namespace tt_metal {

Program fill_rm_single_core(const Tensor& any, Tensor &output, uint32_t N, uint32_t C, uint32_t H, uint32_t W, uint32_t hFill, uint32_t wFill, float val_hi, float val_lo) {

    tt_metal::Device *device = any.device();
    tt_metal::Program program = tt_metal::Program();
    CoreCoord core = {0, 0};

    uint32_t single_tile_size = any.element_size() * TILE_HW;

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    TT_ASSERT(dst_dram_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t num_cb_tiles = 16;
    TT_ASSERT(W < 1024*num_cb_tiles); // Limitation for simplifying the kernel
    auto cb_src0 = tt_metal::CreateCircularBuffer(
        program,
        device,
        0, // cb index
        core,
        num_cb_tiles, num_cb_tiles * single_tile_size,
        DataFormat::Float16_b);
    auto cb_src1 = tt_metal::CreateCircularBuffer(
        program,
        device,
        1, // cb index
        core,
        num_cb_tiles, num_cb_tiles * single_tile_size,
        DataFormat::Float16_b);

    tt_metal::DataMovementKernel *binary_reader_kernel = tt_metal::CreateDataMovementKernel(
        program, "tt_metal/kernels/dataflow/fill_rm_8bank.cpp",
        core, tt_metal::DataMovementProcessor::RISCV_1, tt_metal::NOC::RISCV_1_default);

    tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
        program, "tt_metal/kernels/dataflow/blank.cpp",
        core, tt_metal::DataMovementProcessor::RISCV_0, tt_metal::NOC::RISCV_0_default);

    vector<uint32_t> compute_args = {
        0 // dummy
    };

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
        program, "tt_metal/kernels/compute/blank.cpp",
        core, compute_args, MathFidelity::HiFi4, fp32_dest_acc_en, math_approx_mode);

    tt_metal::WriteRuntimeArgsToDevice(
        device, binary_reader_kernel, core,
        { dst_dram_buffer->address(), u32(N*C), u32(H), u32(W), u32(hFill), u32(wFill), u32(bfloat16(val_hi).to_uint16()), u32(bfloat16(val_lo).to_uint16()) }
    );

    return program;
}

void FillRM::validate(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    TT_ASSERT((this->N > 0 && this -> C > 0 && this-> H > 0 && this-> W > 0));
    TT_ASSERT((this->hFill <= this->H && this->wFill <= this->W));
}

std::vector<Shape> FillRM::compute_output_shapes(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    Shape output_shape = {this->N, this->C, this->H, this->W};
    return {output_shape};
}

std::vector<Tensor> FillRM::create_output_tensors(const std::vector<std::reference_wrapper<const Tensor>> &input_tensors) const {
    return detail::generic_create_output_tensors(*this, input_tensors, Layout::ROW_MAJOR);
}

Program FillRM::create_program(const std::vector<std::reference_wrapper<const Tensor>>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0).get();
    auto& output_tensor = output_tensors.at(0);
    return fill_rm_single_core(input_tensor, output_tensor, this->N, this->C, this->H, this->W, this->hFill, this-> wFill, this->val_hi, this->val_lo);

}

tt_metal::Tensor fill_rm(uint32_t N, uint32_t C, uint32_t H, uint32_t W, uint32_t hFill, uint32_t wFill, const tt_metal::Tensor& any, float val_hi, float val_lo) {
    return detail::run_without_autopad(FillRM(N, C, H, W, hFill, wFill, val_hi, val_lo), any);
}

}  // namespace tt_metal

}  // namespace tt
