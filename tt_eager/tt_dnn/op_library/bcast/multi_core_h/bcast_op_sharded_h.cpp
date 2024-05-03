// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_dnn/op_library/bcast/bcast_op.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tensor/tensor.hpp"
#include "tt_metal/host_api.hpp"

#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"


using namespace tt::tt_metal;
using namespace tt::constants;


namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks bcast_sharded_h(const Tensor &a, const Tensor &b, const Tensor& output, BcastOpMath bcast_math, BcastOpDim bcast_dim){
	const auto ashape = a.get_legacy_shape();
    const auto bshape = b.get_legacy_shape();
    uint32_t N  = ashape[0], C  = ashape[1], H  = ashape[2], W  = ashape[3];
    uint32_t bN = bshape[0], bC = bshape[1], bH = bshape[2], bW = bshape[3];
    uint32_t NC = N*C;
    uint32_t HW = H*W;

	uint32_t num_btensor_tiles = NC*bH*bW / TILE_HW;
	uint32_t bnc1 = (bN*bC == 1) ? 1 : 0;

	std::cout << "bcast fuction " << std::endl;
	tt_metal::Program program = tt_metal::CreateProgram();
    Device *device = a.device();

	auto shard_spec = a.shard_spec().value();
    auto all_cores = shard_spec.grid;
    uint32_t ncores = shard_spec.num_cores();


	uint32_t ncores_x = device->compute_with_storage_grid_size().x;

    auto out_shard_spec = output.shard_spec().value();
    TT_FATAL(out_shard_spec.num_cores() == ncores, "Output tensor should have same number of cores {} as input tensor {}", out_shard_spec.num_cores(), ncores);

    DataFormat act_df = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
    DataFormat out_df = tt_metal::datatype_to_dataformat_converter(output.get_dtype());

    uint32_t input_tile_size = tt::tt_metal::detail::TileSize(act_df);
    uint32_t output_tile_size = tt::tt_metal::detail::TileSize(out_df);

    TT_FATAL(input_tile_size == output_tile_size, "Input and output tile size should be same");
    uint32_t shard_size_in_bytes = shard_spec.numel() * datum_size(act_df);

    uint32_t num_tile_per_core = (shard_size_in_bytes + input_tile_size - 1) / input_tile_size; //ceil value
    TT_FATAL(input_tile_size <= shard_size_in_bytes, "Input tile size should be less than shard size");

	uint32_t Wt, Ht;
	if(a.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED){
		ncores_x = all_cores.ranges().begin()->end.y + 1;
		Wt = shard_spec.shape[1] / TILE_WIDTH;
		Ht = shard_spec.shape[0] / TILE_HEIGHT;
		std::cout << " ncores_x " << ncores_x << " input_tile_size " << input_tile_size << "	num_tile_per_core " << num_tile_per_core << std::endl;
	} else if(a.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED){
		Wt = shard_spec.shape[1] / TILE_WIDTH;
		Ht = shard_spec.shape[0] / TILE_HEIGHT;
	} else{
		TT_FATAL(false, "Unsupported memory layout");
	}

	uint32_t src0_cb_index = CB::c_in0;
	uint32_t aligned_input_tile_nbytes = round_up_to_mul32(input_tile_size); //will have issue if the page is not multiple of 32
    uint32_t in_cb_pagesize = aligned_input_tile_nbytes;
	tt_metal::CircularBufferConfig src0_cb_config = tt_metal::CircularBufferConfig(aligned_input_tile_nbytes * num_tile_per_core,  {{src0_cb_index, act_df}})
                                          .set_page_size(src0_cb_index, in_cb_pagesize)
                                          .set_globally_allocated_address(*a.buffer());
	auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);
	std::cout << "input buffer" << std::endl;

	uint32_t output_cb_index = CB::c_out0; // output operands start at index 16
	tt_metal::CircularBufferConfig output_cb_config = tt_metal::CircularBufferConfig(aligned_input_tile_nbytes * num_tile_per_core,
										  {{output_cb_index, out_df}})
                                          .set_page_size(output_cb_index, in_cb_pagesize)
                                          .set_globally_allocated_address(*output.buffer());
	auto out_cb = tt_metal::CreateCircularBuffer(program, all_cores, output_cb_config);

	uint32_t num_input_tiles = (b.get_legacy_shape()[-1] * output.element_size() + input_tile_size - 1)/ input_tile_size;
	uint32_t src1_cb_index = CB::c_in1;
	tt_metal::CircularBufferConfig src1_cb_config = tt_metal::CircularBufferConfig(num_input_tiles * aligned_input_tile_nbytes, {{src1_cb_index, act_df}})
		.set_page_size(src1_cb_index, aligned_input_tile_nbytes);
	 std::cout << "input_2 " << std::endl;
	auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_cores, src1_cb_config);

	auto src0_buffer = a.buffer();
	auto src1_buffer = b.buffer();
	auto dst_buffer = output.buffer();
    bool src1_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_cb_index, (uint32_t)src1_is_dram};

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};

	KernelHandle binary_reader_kernel_id = tt_metal::CreateKernel(
		program,
		"tt_eager/tt_dnn/op_library/bcast/kernels/dataflow/reader_bcast_h_sharded.cpp",
		all_cores,
		tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

	std::map<std::string, std::string> bcast_defines = bcast_op_utils::get_defines(bcast_dim, bcast_math);
	const char* compute_name = bcast_op_utils::get_compute_name(bcast_dim);
	auto bcast_kernel_id = tt_metal::CreateKernel(
		program,
		compute_name,
		all_cores,
		tt_metal::ComputeConfig{.compile_args = {}, .defines = bcast_defines}
	);

	for (uint32_t i = 0, num_Wtiles_read = 0; i < ncores; i++){
		CoreCoord core;
		uint32_t offset = 0;
		if(a.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED){
			core = {i / ncores_x, i % ncores_x};
			offset = Wt * (i % ncores_x);
		}  else if(a.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED){
			core = {i % ncores_x, i / ncores_x};
			int amb = ncores / ncores_x;
			offset = Wt * (amb*core.x + core.y);
			if(core.y == amb){
				offset = amb * ncores_x + core.x;
			}
		}
		uint32_t Ht_per_core = 1;

		uint32_t num_tensor_tiles_per_core = NC * Ht_per_core * Wt;
		//std::cout << "num_tensor_tiles_per_core " << num_tile_per_core << " NC " << NC << " Ht_per_core " << Ht_per_core << " Wt " << Wt << "offset" << offset << std::endl;
		tt_metal::SetRuntimeArgs(
			program,
			binary_reader_kernel_id,
			core,
			{
				a.buffer()->address(), // 0
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
				Wt, // 11
				bnc1, // 12
				num_Wtiles_read, // 13
				Ht*Wt, // 14
				offset, // 15
			}
		);

		tt_metal::SetRuntimeArgs(
			program,
			bcast_kernel_id,
			core,
			{
				NC, // B
				Ht, // Ht
				Wt,  // Wt
			}
		);

		num_Wtiles_read += Ht_per_core * Wt;
	}

	auto override_runtime_args_callback = [binary_reader_kernel_id, cb_src0, out_cb](
        const void* operation,
        Program &program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();
        UpdateDynamicCircularBufferAddress(program, cb_src0, *src_buffer);
        UpdateDynamicCircularBufferAddress(program, out_cb, *dst_buffer);
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_args_callback};
}


}  // namespace tt_metal

}  // namespace tt
