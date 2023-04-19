#include <algorithm>
#include "tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_dnn/op_library/eltwise_unary/multi_core/eltwise_unary_op_multi_core.hpp"

#include "tt_metal/host_api.hpp"
#include "constants.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

EltwiseUnaryOpMultiCore::EltwiseUnaryOpMultiCore(const Tensor &a, UnaryOpType::Enum op_type) : op_type(op_type) {
    this->tensor_inputs.push_back(a);
}

EltwiseUnaryOpMultiCore::~EltwiseUnaryOpMultiCore() {

}

void EltwiseUnaryOpMultiCore::op_asserts() {
    TT_ASSERT(tensor_inputs.size() == 1);
    TT_ASSERT(tensor_inputs[0].volume() % TILE_HW == 0);
}

Tensor EltwiseUnaryOpMultiCore::create_output() {
    tt_metal::Tensor output = tt_metal::Tensor(this->tensor_inputs[0].shape(), this->tensor_inputs[0].dtype(), tt::tt_metal::Layout::TILE, this->tensor_inputs[0].device());
    return output;
}

void EltwiseUnaryOpMultiCore::create_op(const Tensor& output) {

    uint32_t single_tile_size = 2 * TILE_HW;

    tt_metal::Buffer *src0_dram_buffer = this->tensor_inputs[0].buffer();

    uint32_t num_tiles = this->tensor_inputs[0].volume() / TILE_HW;

    auto dram_src0_noc_xy = src0_dram_buffer->noc_coordinates();

    auto logical_grid_size = this->device->logical_grid_size();
    uint32_t num_cores_x = logical_grid_size.x;
    uint32_t num_cores_y = logical_grid_size.y;
    auto num_cores = std::min(num_tiles, num_cores_x * num_cores_y);
    std::vector<uint32_t> num_tiles_per_core(num_cores, num_tiles / num_cores);
    for(uint32_t i = 0; i < num_tiles % num_cores; i++){
        num_tiles_per_core[i]++;
    }

    tt_metal::Buffer *dst_dram_buffer = output.buffer();
    auto dram_dst_noc_xy = dst_dram_buffer->noc_coordinates();

    std::vector<tt_metal::DataMovementKernel *> unary_reader_kernels;
    std::vector<tt_metal::DataMovementKernel *> unary_writer_kernels;
    for (uint32_t i = 0; i < num_cores; i++){
        tt_xy_pair core = {i / num_cores_y, i % num_cores_y};
        uint32_t src0_cb_index = 0;
        uint32_t num_input_tiles = 2;
        auto cb_src0 = tt_metal::CreateCircularBuffer(
            this->program,
            this->device,
            src0_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            DataFormat::Float16_b
        );

        uint32_t src1_cb_index = 1;
        auto cb_src1 = tt_metal::CreateCircularBuffer(
            this->program,
            this->device,
            src1_cb_index,
            core,
            num_input_tiles,
            num_input_tiles * single_tile_size,
            DataFormat::Float16_b
        );

        uint32_t ouput_cb_index = 16; // output operands start at index 16
        uint32_t num_output_tiles = 2;
        auto cb_output = tt_metal::CreateCircularBuffer(
            this->program,
            this->device,
            ouput_cb_index,
            core,
            num_output_tiles,
            num_output_tiles * single_tile_size,
            DataFormat::Float16_b
        );

        tt_metal::DataMovementKernel *unary_reader_kernel = tt_metal::CreateDataMovementKernel(
            this->program,
            "tt_metal/kernels/dataflow/reader_unary_8bank_start_id.cpp",
            core,
            tt_metal::DataMovementProcessor::RISCV_1,
            tt_metal::NOC::RISCV_1_default);
        unary_reader_kernels.push_back(unary_reader_kernel);

        tt_metal::DataMovementKernel *unary_writer_kernel = tt_metal::CreateDataMovementKernel(
            this->program,
            "tt_metal/kernels/dataflow/writer_unary_8bank_start_id.cpp",
            core,
            tt_metal::DataMovementProcessor::RISCV_0,
            tt_metal::NOC::RISCV_0_default);
        unary_writer_kernels.push_back(unary_writer_kernel);

        vector<uint32_t> compute_kernel_args = {
            num_tiles_per_core[i], // per_core_block_cnt
            1 // per_core_block_size
        };
        tt_metal::ComputeKernelArgs *eltwise_unary_args = tt_metal::InitializeCompileTimeComputeKernelArgs(core, compute_kernel_args);

        bool fp32_dest_acc_en = false;
        bool math_approx_mode = false;
        auto eltwise_unary_kernel = tt_metal::CreateComputeKernel(
            this->program,
            "tt_metal/kernels/compute/eltwise_sfpu.cpp",
            core,
            eltwise_unary_args,
            MathFidelity::HiFi4,
            fp32_dest_acc_en,
            math_approx_mode
        );

        eltwise_unary_op_utils::add_defines(eltwise_unary_kernel, op_type);
    }

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++){
        tt_xy_pair core = {i / num_cores_y, i % num_cores_y};
        tt_metal::WriteRuntimeArgsToDevice(
            this->device,
            unary_reader_kernels[i],
            core,
            {src0_dram_buffer->address(),
            uint32_t(dram_src0_noc_xy.x),
            uint32_t(dram_src0_noc_xy.y),
            num_tiles_per_core[i],
            num_tiles_written, 0 /*disable scaler*/ }
        );

        tt_metal::WriteRuntimeArgsToDevice(
            this->device,
            unary_writer_kernels[i],
            core,
            {dst_dram_buffer->address(),
            uint32_t(dram_dst_noc_xy.x),
            uint32_t(dram_dst_noc_xy.y),
            num_tiles_per_core[i],
            num_tiles_written }
        );
        num_tiles_written+=num_tiles_per_core[i];
    }
}

Tensor eltwise_unary_multi_core(const Tensor &a, UnaryOpType::Enum op_type) {
    EltwiseUnaryOpMultiCore eltwise_unary_op(a, op_type);
    return eltwise_unary_op.run_op();
}


}  // namespace tt_metal

}  // namespace tt
