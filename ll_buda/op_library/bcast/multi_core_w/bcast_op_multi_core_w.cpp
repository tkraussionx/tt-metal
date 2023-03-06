#include "ll_buda/op_library/bcast/bcast_op.hpp"
#include "ll_buda/tensor/tensor.hpp"
#include "ll_buda/host_api.hpp"

#include "constants.hpp"


using namespace tt::ll_buda;
using namespace tt::constants;
using u32 = std::uint32_t;


namespace tt {

namespace ll_buda {

Tensor bcast_multi_core_w(const Tensor &a, const Tensor &b, BcastOpMath::Enum bcast_math, BcastOpDim::Enum bcast_dim) {
    TT_ASSERT(bcast_dim == BcastOpDim::W);

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

    ll_buda::Device *device = a.device();

		auto logical_grid_size = device->logical_grid_size();
    uint32_t num_cores_x = logical_grid_size.x;
    uint32_t num_cores_y = logical_grid_size.y;
    auto num_cores = std::min(Wt, num_cores_x * num_cores_y);
    std::vector<uint32_t> Wt_per_core(num_cores, Wt / num_cores);
    for(uint32_t i = 0; i < Wt % num_cores; i++){
        Wt_per_core[i]++;
    }

    // TODO: Build some sort of dispatcher based on location of op operands
    TT_ASSERT(a.device() != nullptr and b.device() != nullptr, "Operands to eltwise binary need to be on device!");
    TT_ASSERT(a.device() == b.device(), "Operands to bcast need to be on the same device!");
    TT_ASSERT(a.buffer() != nullptr and b.buffer() != nullptr, "Operands to eltwise binary need to be allocated in buffers on device!");

    uint32_t single_tile_size = 2 * 1024;

    // This should allocate a DRAM buffer on the device
    ll_buda::Tensor output = Tensor(a.shape(), a.dtype(), tt::ll_buda::Layout::TILE, device);

    const char* reader_name = bcast::get_reader_name(bcast_dim, BcastOpParallelizationStrategy::MULTI_CORE_W);
    const char* compute_name = bcast::get_compute_name(bcast_dim);

		std::vector<ll_buda::DataMovementKernel *> binary_reader_kernels;
    std::vector<ll_buda::DataMovementKernel *> unary_writer_kernels;
    for (uint32_t i = 0; i < num_cores; i++){
				tt_xy_pair core = {i / num_cores_y, i % num_cores_y};

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

    		ll_buda::DataMovementKernel *binary_reader_kernel = ll_buda::CreateDataMovementKernel(
    		    program,
    		    reader_name,
    		    core,
    		    ll_buda::DataMovementProcessor::RISCV_1,
    		    ll_buda::NOC::RISCV_1_default);
				binary_reader_kernels.push_back(binary_reader_kernel);

    		ll_buda::DataMovementKernel *unary_writer_kernel = ll_buda::CreateDataMovementKernel(
    		    program,
    		    "kernels/dataflow/writer_unary_8bank_for_multi_core.cpp",
    		    core,
    		    ll_buda::DataMovementProcessor::RISCV_0,
    		    ll_buda::NOC::RISCV_0_default);
				unary_writer_kernels.push_back(unary_writer_kernel);

    		// TODO(AP): add dimensions and op params
    		void *hlk_args = new bcast::hlk_args_t { .B = NC, .Ht = Ht, .Wt = Wt_per_core[i] };
    		ll_buda::ComputeKernelArgs *compute_args = ll_buda::InitializeCompileTimeComputeKernelArgs(core, hlk_args, sizeof(bcast::hlk_args_t));

    		bool fp32_dest_acc_en = false;
    		bool math_approx_mode = false;
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
		}

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile Application
    ////////////////////////////////////////////////////////////////////////////
    bool skip_hlkc = false;
    ll_buda::CompileProgram(device, program, skip_hlkc);

    ////////////////////////////////////////////////////////////////////////////
    //                      Execute Application
    ////////////////////////////////////////////////////////////////////////////
		for (uint32_t i = 0, num_Wtiles_read = 0; i < num_cores; num_Wtiles_read+=Wt_per_core[i], i++){
        tt_xy_pair core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tensor_tiles_per_core = NC*Ht*Wt_per_core[i];
        uint32_t Wt_skip = Wt - Wt_per_core[i];

        uint32_t bnc1 = (bN*bC == 1) ? 1 : 0;
        ll_buda::WriteRuntimeArgsToDevice(
            device,
            binary_reader_kernels[i],
            core,
            {a.buffer()->address(), // 0
            0, // 1
            0, // 2
            num_tensor_tiles_per_core, // 3
            b.buffer()->address(), // 4
            0, // 5
            0, // 6
            num_btensor_tiles, // 7
						num_tensor_tiles_per_core, // 8
						NC, // 9
						Ht, // 10
						Wt_per_core[i], // 11
						bnc1, // 12
						Wt_skip, // 13
						num_Wtiles_read, // 14
						Ht*Wt, // 15
				});

        ll_buda::WriteRuntimeArgsToDevice(
            device, unary_writer_kernels[i], core,
            {
                output.buffer()->address(),
                0,
                0,
                Ht,
                Wt_per_core[i],
                num_Wtiles_read,
                Wt_skip,
                NC,
                Ht*Wt,
            });
		}

    ll_buda::ConfigureDeviceWithProgram(device, program);

    ll_buda::LaunchKernels(device, program);

    delete program;

    ll_buda::dumpProfilerResults("multi_core_w_" + std::to_string(num_cores), true);

    // output does not hold any data, contains pointer to buffer on device with the data
    return output;
}

}  // namespace ll_buda

}  // namespace tt
