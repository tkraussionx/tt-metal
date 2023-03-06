#include "ll_buda/op_library/bcast/bcast_op.hpp"
#include "ll_buda/tensor/tensor.hpp"
#include "ll_buda/host_api.hpp"

#include "constants.hpp"


using namespace tt::ll_buda;
using namespace tt::constants;
using u32 = std::uint32_t;


namespace tt {

namespace ll_buda {

Tensor bcast_single_core(const Tensor &a, const Tensor &b, BcastOpMath::Enum bcast_math, BcastOpDim::Enum bcast_dim) {

    const auto ashape = a.shape();
    const auto bshape = b.shape();
    u32 N  = ashape[0], C  = ashape[1], H  = ashape[2], W  = ashape[3];
    u32 bN = bshape[0], bC = bshape[1], bH = bshape[2], bW = bshape[3];
    u32 NC = N*C;
    u32 HW = H*W;

    TT_ASSERT(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0);
    TT_ASSERT(H > 0 && W > 0 && NC > 0);
    TT_ASSERT(a.volume() % TILE_HW == 0);

    TT_ASSERT((bN*bC == 1 || (bN == N && bC == C)) && "Broadcast is currently only supported when bN*bC=1 or N & C match");
    // validate input dimensions
    if (bcast_dim == BcastOpDim::W)
        TT_ASSERT(H == bH && bW == TILE_WIDTH);
    if (bcast_dim == BcastOpDim::H)
        TT_ASSERT(W == bW && bH == TILE_HEIGHT);
    if (bcast_dim == BcastOpDim::HW)
        TT_ASSERT(bW == TILE_WIDTH && bH == TILE_HEIGHT);

    u32 Wt = W/TILE_WIDTH;
    u32 Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = NC*Ht*Wt;
    uint32_t num_btensor_tiles = NC*bH*bW / TILE_HW;

    ll_buda::Program *program = new ll_buda::Program();

    tt_xy_pair core = {0, 0};

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(a.device() != nullptr and b.device() != nullptr, "Operands to eltwise binary need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to bcast need to be on the same device!");
    TT_ASSERT(a.buffer() != nullptr and b.buffer() != nullptr, "Operands to eltwise binary need to be allocated in buffers on device!");

    uint32_t single_tile_size = 2 * 1024;

    // This should allocate a DRAM buffer on the device
    ll_buda::Device *device = a.device();
    ll_buda::Tensor output = Tensor(a.shape(), a.dtype(), tt::ll_buda::Layout::TILE, device);

    uint32_t src0_cb_index = 0;
    uint32_t src0_cb_addr = 200 * 1024;
    uint32_t num_input_tiles = 2;
    auto cb_src0 = ll_buda::CreateCircularBuffer(
        program,
        device,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        src0_cb_addr,
        DataFormat::Float16_b
    );

    uint32_t src1_cb_index = 1;
    uint32_t src1_cb_addr = 300 * 1024;
    auto cb_src1 = ll_buda::CreateCircularBuffer(
        program,
        device,
        src1_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        src1_cb_addr,
        DataFormat::Float16_b
    );

    uint32_t ouput_cb_index = 16; // output operands start at index 16
    uint32_t output_cb_addr = 400 * 1024;
    uint32_t num_output_tiles = 2;
    auto cb_output = ll_buda::CreateCircularBuffer(
        program,
        device,
        ouput_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        output_cb_addr,
        DataFormat::Float16_b
    );

    const char* reader_name = bcast::get_reader_name(bcast_dim, BcastOpParallelizationStrategy::SINGLE_CORE);
    ll_buda::DataMovementKernel *binary_reader_kernel = ll_buda::CreateDataMovementKernel(
        program,
        reader_name,
        core,
        ll_buda::DataMovementProcessor::RISCV_1,
        ll_buda::NOC::RISCV_1_default);

    ll_buda::DataMovementKernel *unary_writer_kernel = ll_buda::CreateDataMovementKernel(
        program,
        "kernels/dataflow/writer_unary_8bank.cpp",
        core,
        ll_buda::DataMovementProcessor::RISCV_0,
        ll_buda::NOC::RISCV_0_default);

    // TODO(AP): add dimensions and op params
    void *hlk_args = new bcast::hlk_args_t { .B = NC, .Ht = Ht, .Wt = Wt };
    ll_buda::ComputeKernelArgs *compute_args = ll_buda::InitializeCompileTimeComputeKernelArgs(core, hlk_args, sizeof(bcast::hlk_args_t));

    bool fp32_dest_acc_en = false;
    bool math_approx_mode = false;
    const char* compute_name = bcast::get_compute_name(bcast_dim);
    auto bcast_kernel = ll_buda::CreateComputeKernel(
        program,
        compute_name,
        core,
        compute_args,
        MathFidelity::HiFi4,
        fp32_dest_acc_en,
        math_approx_mode
    );
    bcast::set_compute_kernel_defines(bcast_kernel, bcast_math);

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    bool skip_hlkc = false;
    ll_buda::CompileProgram(device, program, skip_hlkc);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////

        uint32_t bnc1 = (bN*bC == 1) ? 1 : 0;
        ll_buda::WriteRuntimeArgsToDevice(
            device,
            binary_reader_kernel,
            core,
            {a.buffer()->address(), // 0
            0, // 1
            0, // 2
            num_tensor_tiles, // 3
            b.buffer()->address(), // 4
            0, // 5
            0, // 6
            num_btensor_tiles, NC*Ht*Wt, NC, Ht, Wt, bnc1}); // 7 8 9 10 11 12

        ll_buda::WriteRuntimeArgsToDevice(
            device,
            unary_writer_kernel,
            core,
            { output.buffer()->address(),
              0, 0, num_tensor_tiles});

    ll_buda::ConfigureDeviceWithProgram(device, program);

    ll_buda::LaunchKernels(device, program);

    delete program;

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

}  // namespace ll_buda

}  // namespace tt
