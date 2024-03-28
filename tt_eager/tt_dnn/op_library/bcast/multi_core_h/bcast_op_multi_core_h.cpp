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

operation::ProgramWithCallbacks bcast_multicore_sharded_h(const Tensor &a, const Tensor &b, const Tensor& output, BcastOpMath bcast_math, BcastOpDim bcast_dim){
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

operation::ProgramWithCallbacks bcast_multi_core_h(const Tensor &a, const Tensor &b, const Tensor& output, BcastOpMath bcast_math, BcastOpDim bcast_dim) {
	if(a.is_sharded()){
		return bcast_multicore_sharded_h(a, b, output, bcast_math, bcast_dim);
	}
    TT_ASSERT(bcast_dim == BcastOpDim::H);

    const auto ashape = a.get_legacy_shape();
    const auto bshape = b.get_legacy_shape();
    uint32_t N = ashape.rank() >= 4 ? ashape[-4] : 1;
    uint32_t C = ashape.rank() >= 3 ? ashape[-3] : 1;
    uint32_t H = ashape[-2];
    uint32_t W = ashape[-1];
    uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;
    uint32_t bC = bshape.rank() >= 3 ? bshape[-3] : 1;
    uint32_t bH = bshape[-2];
    uint32_t bW = bshape[-1];
    uint32_t NC = N*C;
    uint32_t HW = H*W;

    uint32_t Wt = W/TILE_WIDTH;
    uint32_t Ht = H/TILE_HEIGHT;

    uint32_t num_tensor_tiles = NC*Ht*Wt;
    uint32_t num_btensor_tiles = NC*bH*bW / TILE_HW;

	uint32_t bnc1 = (bN*bC == 1) ? 1 : 0;

    tt_metal::Program program = tt_metal::CreateProgram();

    tt_metal::Device *device = a.device();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());

    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, Ht_per_core_group_1, Ht_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, Ht);

	auto src0_buffer = a.buffer();
	auto src1_buffer = b.buffer();
	auto dst_buffer = output.buffer();
	TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const char* reader_name = bcast_op_utils::get_reader_name(bcast_dim, BcastOpParallelizationStrategy::MULTI_CORE_H);
    const char* compute_name = bcast_op_utils::get_compute_name(bcast_dim);

	uint32_t src0_cb_index = 0;
	uint32_t num_input_tiles = 2;

	tt_metal::CircularBufferConfig src0_cb_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
		.set_page_size(src0_cb_index, single_tile_size);
	auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_device_cores, src0_cb_config);

	uint32_t src1_cb_index = 1;
	tt_metal::CircularBufferConfig src1_cb_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
		.set_page_size(src1_cb_index, single_tile_size);
	auto cb_src1 = tt_metal::CreateCircularBuffer(program, all_device_cores, src1_cb_config);

	uint32_t output_cb_index = 16; // output operands start at index 16
	uint32_t num_output_tiles = 2;
	tt_metal::CircularBufferConfig output_cb_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
		.set_page_size(output_cb_index, single_tile_size);
	auto cb_output = tt_metal::CreateCircularBuffer(program, all_device_cores, output_cb_config);

	bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool src1_is_dram = src1_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src0_is_dram, (uint32_t)src1_is_dram};

    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)dst_is_dram};

	KernelHandle binary_reader_kernel_id = tt_metal::CreateKernel(
		program,
		"tt_eager/tt_dnn/op_library/bcast/kernels/dataflow/reader_bcast_h_sharded.cpp",
		all_device_cores,
		tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

	KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
		program,
		"tt_eager/tt_dnn/op_library/bcast/kernels/dataflow/writer_unary_interleaved_input_cols_batched.cpp",
		all_device_cores,
		tt_metal::WriterDataMovementConfig(writer_compile_time_args));

	std::map<std::string, std::string> bcast_defines = bcast_op_utils::get_defines(bcast_dim, bcast_math);
	auto bcast_kernel_id = tt_metal::CreateKernel(
		program,
		compute_name,
		all_device_cores,
		tt_metal::ComputeConfig{.compile_args = {}, .defines = bcast_defines}
	);

	for (uint32_t i = 0, num_Wtiles_read = 0; i < num_cores_y * num_cores_x; i++){
		CoreCoord core = {i / num_cores_y, i % num_cores_y};
		uint32_t Ht_per_core;
		if (core_group_1.core_coord_in_core_ranges(core)) {
			Ht_per_core = Ht_per_core_group_1;
		} else if (core_group_2.core_coord_in_core_ranges(core)) {
			Ht_per_core = Ht_per_core_group_2;
		} else {
			tt_metal::SetRuntimeArgs(program, binary_reader_kernel_id, core, std::vector<uint32_t>(15, 0));
			tt_metal::SetRuntimeArgs(program, bcast_kernel_id, core, std::vector<uint32_t>(3, 0));
			tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, std::vector<uint32_t>(9, 0));
			continue;
		}
		uint32_t num_tensor_tiles_per_core = NC * Ht_per_core * Wt;

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
				Ht_per_core, // 10
				Wt, // 11
				bnc1, // 12
				num_Wtiles_read, // 13
				Ht*Wt, // 14
			}
		);

		tt_metal::SetRuntimeArgs(
			program,
			bcast_kernel_id,
			core,
			{
				NC, // B
				Ht_per_core, // Ht
				Wt  // Wt
			}
		);

		tt_metal::SetRuntimeArgs(
			program, unary_writer_kernel_id, core,
			{
				output.buffer()->address(),
				0,
				0,
				Ht_per_core,
				Wt,
				num_Wtiles_read,
				0,
				NC,
				Ht*Wt,
			}
		);

		num_Wtiles_read += Ht_per_core * Wt;
	}

    auto override_runtime_arguments_callback = [
            binary_reader_kernel_id,
            unary_writer_kernel_id,
			bcast_kernel_id,
            compute_with_storage_grid_size
        ]
    (
        const void* operation,
        const Program& program,
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>&,
        const std::vector<Tensor>& output_tensors
    ) {
		uint32_t num_cores_x = compute_with_storage_grid_size.x;
		uint32_t num_cores_y = compute_with_storage_grid_size.y;

        auto src_dram_buffer_a = input_tensors.at(0).buffer();
        auto src_dram_buffer_b = input_tensors.at(1).buffer();

        auto dst_dram_buffer = output_tensors.at(0).buffer();

		const auto ashape = input_tensors.at(0).get_legacy_shape();
		const auto bshape = input_tensors.at(1).get_legacy_shape();
        uint32_t N = ashape.rank() >= 4 ? ashape[-4] : 1;
        uint32_t C = ashape.rank() >= 3 ? ashape[-3] : 1;
        uint32_t H = ashape[-2];
        uint32_t W = ashape[-1];
        uint32_t bN = bshape.rank() >= 4 ? bshape[-4] : 1;
        uint32_t bC = bshape.rank() >= 3 ? bshape[-3] : 1;
        uint32_t bH = bshape[-2];
        uint32_t bW = bshape[-1];
        uint32_t NC = N * C;
        uint32_t HW = H * W;

        uint32_t Wt = W/TILE_WIDTH;
		uint32_t Ht = H/TILE_HEIGHT;

		uint32_t num_tensor_tiles = NC*Ht*Wt;
		uint32_t num_btensor_tiles = NC*bH*bW / TILE_HW;

		uint32_t bnc1 = (bN*bC == 1) ? 1 : 0;

		auto [num_cores, all_cores, core_group_1, core_group_2, Ht_per_core_group_1, Ht_per_core_group_2] = split_work_to_cores(compute_with_storage_grid_size, Ht);

		for (uint32_t i = 0, num_Wtiles_read = 0; i < num_cores_y * num_cores_x; i++){
			CoreCoord core = {i / num_cores_y, i % num_cores_y};
			uint32_t Ht_per_core;
			if (core_group_1.core_coord_in_core_ranges(core)) {
				Ht_per_core = Ht_per_core_group_1;
			} else if (core_group_2.core_coord_in_core_ranges(core)) {
				Ht_per_core = Ht_per_core_group_2;
			} else {
				tt_metal::SetRuntimeArgs(program, binary_reader_kernel_id, core, std::vector<uint32_t>(15, 0));
				tt_metal::SetRuntimeArgs(program, bcast_kernel_id, core, std::vector<uint32_t>(3, 0));
				tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, std::vector<uint32_t>(9, 0));
				continue;
			}
			uint32_t num_tensor_tiles_per_core = NC * Ht_per_core * Wt;

			tt_metal::SetRuntimeArgs(
				program,
				binary_reader_kernel_id,
				core,
				{
					src_dram_buffer_a->address(), // 0
					0, // 1
					0, // 2
					num_tensor_tiles_per_core, // 3
					src_dram_buffer_b->address(), // 4
					0, // 5
					0, // 6
					num_btensor_tiles, // 7
					num_tensor_tiles_per_core, // 8
					NC, // 9
					Ht_per_core, // 10
					Wt, // 11
					bnc1, // 12
					num_Wtiles_read, // 13
					Ht*Wt, // 14
				}
			);

			tt_metal::SetRuntimeArgs(
				program,
				bcast_kernel_id,
				core,
				{
					NC, // B
					Ht_per_core, // Ht
					Wt  // Wt
				}
			);

			tt_metal::SetRuntimeArgs(
				program, unary_writer_kernel_id, core,
				{
					dst_dram_buffer->address(),
					0,
					0,
					Ht_per_core,
					Wt,
					num_Wtiles_read,
					0,
					NC,
					Ht*Wt,
				}
			);

			num_Wtiles_read += Ht_per_core * Wt;
		}
    };

    return {.program=std::move(program), .override_runtime_arguments_callback=override_runtime_arguments_callback};
}

}  // namespace tt_metal

}  // namespace tt
