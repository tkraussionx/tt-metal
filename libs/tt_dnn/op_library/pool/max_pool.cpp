#include <algorithm>
#include <cmath>

#include "tt_dnn/op_library/pool/max_pool.hpp"
#include "tt_metal/host_api.hpp"
#include "tensor/tensor_utils.hpp"
#include "detail/util.hpp"

// #include "tt_metal/llrt/tt_debug_print_server.hpp"


namespace tt {
namespace tt_metal {

operation::ProgramWithCallbacks max_pool_2d_single_core(const Tensor &input, Tensor& output,
                                                        uint32_t kernel_size_h, uint32_t kernel_size_w,
                                                        uint32_t stride_h, uint32_t stride_w,
                                                        uint32_t pad_h, uint32_t pad_w,
                                                        uint32_t dilation_h, uint32_t dilation_w,
                                                        const MemoryConfig& out_mem_config) {
    Program program = Program();
    CoreRange cores = {.start={0, 0}, .end={0, 0}};

    // This should allocate a DRAM buffer on the device
    Device *device = input.device();
    Buffer *src_dram_buffer = input.buffer();
    Buffer *dst_dram_buffer = output.buffer();

    Shape input_shape = input.shape();
    Shape output_shape = output.shape();

    log_debug("SHAPES: input = {}, output = {}", input_shape, output_shape);

    // TODO [AS]: Support other data formats??
    DataFormat in_df = datatype_to_dataformat_converter(input.dtype());
    DataFormat out_df = datatype_to_dataformat_converter(output.dtype());
    uint32_t in_nbytes = 2;
    uint32_t out_nbytes = 2;
    uint32_t in_nbytes_w = input_shape[3] * in_nbytes;       // row of input
    uint32_t out_nbytes_w = output_shape[3] * out_nbytes;     // row of output

    uint32_t in_cb_id = CB::c_in0;
    uint32_t out_cb_id = CB::c_out0;

    // tt_start_debug_print_server(device->cluster(), {0}, {{1, 1}});

    auto cb_in = CreateCircularBuffers(
        program,
        in_cb_id,
        cores,
        kernel_size_h,                  // # rows worth the window height
        kernel_size_h * in_nbytes_w,    // in bytes
        in_df
    );

    uint32_t out_pagesize = (out_nbytes_w % 16 == 0) ? out_nbytes_w : (out_nbytes_w + 16 - out_nbytes_w % 16);
    // Temporary hack -- no longer used, clean it up
    uint32_t out_tile_size = detail::TileSize(out_df);
    uint32_t out_pagesize_tile_aligned = out_pagesize % out_tile_size == 0
                                         ? out_pagesize
                                         : out_pagesize + out_tile_size - out_pagesize % out_tile_size;
    uint32_t out_npages = 2;    // double buf
    auto cb_out = CreateCircularBuffers(
        program,
        out_cb_id,
        cores,
        out_npages,
        out_npages * out_pagesize,   // padded row size
        out_df
    );

    uint32_t kernel_size_hw = kernel_size_h * (kernel_size_w + (kernel_size_w & 0x01));    // need to pad width if its odd so that it is even sized
    // take multiple of 4
    kernel_size_hw = kernel_size_hw % 4 == 0 ? kernel_size_hw : (kernel_size_hw + 4 - kernel_size_hw % 4);
    log_debug("kernel_size: {},{} ; kernel_size_hw: {}", kernel_size_h, kernel_size_w, kernel_size_hw);

    std::vector<uint32_t> reader_ct_args = {(input.memory_config().buffer_type == BufferType::DRAM) ? (uint) 1 : (uint) 0,
                                            (out_mem_config.buffer_type == BufferType::DRAM) ? (uint) 1 : (uint) 0};
    std::vector<uint32_t> reader_rt_args = {src_dram_buffer->address(),
                                            dst_dram_buffer->address(),
                                            kernel_size_h, kernel_size_w, kernel_size_hw,
                                            stride_h, stride_w,
                                            pad_h, pad_w,
                                            output_shape[0], output_shape[1], output_shape[2], output_shape[3],
                                            in_nbytes_w, out_nbytes_w,
                                            input_shape[0], input_shape[1], input_shape[2], input_shape[3],
                                            out_pagesize, out_pagesize_tile_aligned};
    auto reader_config = DataMovementConfig{.processor = DataMovementProcessor::RISCV_1,
                                            .noc = NOC::RISCV_1_default,
                                            .compile_args = reader_ct_args};
    auto reader_kernel = CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_max_pool_2d_single_core.cpp",
        cores,
        reader_config);

    std::vector<uint32_t> writer_ct_args = reader_ct_args;
    std::vector<uint32_t> writer_rt_args = reader_rt_args;
    auto writer_config = DataMovementConfig{.processor = DataMovementProcessor::RISCV_0,
                                            .noc = NOC::RISCV_0_default,
                                            .compile_args = writer_ct_args};
    auto writer_kernel = CreateDataMovementKernel(program,
                                                  "tt_metal/kernels/dataflow/writer_max_pool_2d_single_core.cpp",
                                                  cores,
                                                  writer_config);

    // No compute required, so using blank kernel
    auto compute_config = ComputeConfig{.math_fidelity = MathFidelity::HiFi4,
                                        .fp32_dest_acc_en = false,
                                        .math_approx_mode = false,
                                        .compile_args = {}};
    auto compute_kernel = CreateComputeKernel(program,
                                              "tt_metal/kernels/compute/blank.cpp",
                                              cores,
                                              compute_config);

    SetRuntimeArgs(program, reader_kernel, cores, reader_rt_args);
    SetRuntimeArgs(program, writer_kernel, cores, writer_rt_args);

    auto override_runtime_args_callback =
        [reader_kernel, writer_kernel](const Program& program,
                                       const std::vector<Buffer*>& input_buffers,
                                       const std::vector<Buffer*>& output_buffers) {
        auto src_dram_buffer = input_buffers.at(0);
        auto dst_dram_buffer = output_buffers.at(0);
        CoreCoord core = {0, 0};
        {
            auto runtime_args = GetRuntimeArgs(program, reader_kernel, core);
            runtime_args[0] = src_dram_buffer->address();
            runtime_args[1] = dst_dram_buffer->address();
            SetRuntimeArgs(program, reader_kernel, core, runtime_args);
        }
        {
            auto runtime_args = GetRuntimeArgs(program, writer_kernel, core);
            runtime_args[0] = src_dram_buffer->address();
            runtime_args[1] = dst_dram_buffer->address();
            SetRuntimeArgs(program, writer_kernel, core, runtime_args);
        }
    };
    return {std::move(program), override_runtime_args_callback};
}

void MaxPool::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input = input_tensors.at(0);
    TT_ASSERT(input.storage_type() == StorageType::DEVICE, "Operands to reshape need to be on device!");
    TT_ASSERT(input.buffer() != nullptr , "Operands to reshape need to be allocated in buffers on device!");
    TT_ASSERT(input.dtype() == DataType::BFLOAT16, "Only BFLOAT16 supported for now");
    TT_ASSERT(input.layout() == Layout::TILE || input.layout() == Layout::ROW_MAJOR, "Only tile and row major reshape supported!");

    TT_ASSERT(2 * pad_h_ < kernel_size_h_ && 2 * pad_w_ < kernel_size_w_, "Total padding along a dim should be less than kernel/window size along same dim");

    // Current restrictions with the kernel.
    // NOTE: Assuming 2 byte data type.
    TT_ASSERT(input.shape()[3] * 2 % constants::TILE_WIDTH == 0, "Input tensor's fastest dim should be divisible by TILE_WIDTH");
}

std::vector<Shape> MaxPool::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input = input_tensors.at(0);
    const auto input_shape = input.shape().without_padding();
    // Assuming NCHW
    Shape output_shape = {input_shape[0],   // batch
                          input_shape[1],   // channels
                          ((input_shape[2] + 2 * pad_h_ - (dilation_h_ * kernel_size_h_ - 1) - 1) / stride_h_) + 1,     // floor
                          ((input_shape[3] + 2 * pad_w_ - (dilation_w_ * kernel_size_w_ - 1) - 1) / stride_w_) + 1};    // floor
    log_debug("Output shape: {}", output_shape);
    uint32_t out_nbytes_w = output_shape[3] * 2;     // row of output
    uint32_t out_pagesize = (out_nbytes_w % 16 == 0) ? out_nbytes_w : (out_nbytes_w + 16 - out_nbytes_w % 16);
    log_debug("test out_pagesize = {}", out_pagesize);
    // TODO: what about padding? Assert out non 16 multiple page sizes for now
    // TT_ASSERT(out_nbytes_w % 16 == 0);
    return {output_shape};
}

std::vector<Tensor> MaxPool::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input= input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input.dtype(), input.layout(), out_mem_config_);
}

operation::ProgramWithCallbacks MaxPool::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return {max_pool_2d_single_core(input_tensor_a, output_tensor,
                                    kernel_size_h_, kernel_size_w_,
                                    stride_h_, stride_w_,
                                    pad_h_, pad_w_,
                                    dilation_h_, dilation_w_,
                                    out_mem_config_)};
}

operation::Hash MaxPool::compute_program_hash(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return fmt::format("{}_{}", *this, input_tensor);
}

tt::stl::reflection::Attributes MaxPool::attributes() const {
    return {
        {"kernel_size_h", kernel_size_h_},
        {"kernel_size_w", kernel_size_w_},
        {"stride_h", stride_h_},
        {"stride_w", stride_w_},
        {"pad_h", pad_h_},
        {"pad_w", pad_w_},
        {"dilation_h", dilation_h_},
        {"dilation_w", dilation_w_},
    };
}

Tensor max_pool2d(const Tensor &input,
                  uint32_t kernel_size_h, uint32_t kernel_size_w,
                  uint32_t stride_h, uint32_t stride_w,
                  uint32_t pad_h, uint32_t pad_w,
                  uint32_t dilation_h, uint32_t dilation_w,
                  const MemoryConfig& out_mem_config) {
    TT_ASSERT(dilation_h == 1 && dilation_w == 1 && "Dilation not yet supported in max_pool2d.");
    TT_ASSERT(pad_h < 2 && pad_w < 2 && "Padding > 1 not yet supported.");
    TT_ASSERT(stride_h == stride_w && "Stride should be equal for both H and W for now.");
    return operation::run_without_autoformat(MaxPool{kernel_size_h, kernel_size_w,
                                                     stride_h, stride_w,
                                                     pad_h, pad_w,
                                                     dilation_h, dilation_w,
                                                     out_mem_config},
                                             {input}).at(0);
}

} // namespace tt_metal
} // namespace tt
