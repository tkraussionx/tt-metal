#include <cmath>

#include "tt_dnn/op_library/untilize/untilize_op.hpp"
#include "tt_dnn/op_library/math.hpp"
#include "tt_dnn/op_library/work_split.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"

using namespace tt::constants;

namespace tt {
namespace tt_metal {

operation::ProgramWithCallbacks untilize_single_core(const Tensor &a, Tensor& output) {

    tt_metal::Program program = tt_metal::Program();

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    tt_metal::Buffer *src0_buffer = a.buffer();

    int32_t num_tiles = a.volume() / TILE_HW;

    uint32_t num_sticks = a.shape()[0] * a.shape()[1] * a.shape()[2];
    uint32_t stick_size = a.shape()[3] * a.element_size();

    uint32_t stick_s = a.shape()[3];
    uint32_t num_tiles_in_row = stick_s / TILE_WIDTH;
    // Ensure we don't intrude into storage space
    uint32_t max_l1_size = a.device()->l1_size() / 2 - UNRESERVED_BASE;
    uint32_t max_tiles = max_l1_size / (2 * single_tile_size); // 2 CBs
    // Currently need the number of tiles in a row to be divisible by tiles in a block
    uint32_t num_tiles_per_block = 1;
    if (num_tiles_in_row <= max_tiles) {
        num_tiles_per_block = num_tiles_in_row;
    } else {
        for(uint32_t n_t = max_tiles; n_t > 0; n_t--) {
            if (num_tiles_in_row % n_t == 0) {
                num_tiles_per_block = n_t;
                break;
            }
        }
    }
    uint32_t block_width_size = num_tiles_per_block * TILE_WIDTH * a.element_size();
    uint32_t num_full_blocks_in_row = num_tiles_in_row / num_tiles_per_block;
    uint32_t num_leftover_tiles = num_tiles_in_row % num_tiles_per_block;
    uint32_t leftover_width_in_row = num_leftover_tiles * a.element_size();

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = num_tiles_per_block;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        cb_data_format
    );

    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = num_tiles_per_block;
    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        output_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        cb_data_format
    );

    // Writer compile-time args
    vector<uint32_t> writer_kernel_args = {
        dst_buffer->address(),
        num_sticks,
        stick_size,
        num_tiles_per_block,
        block_width_size,
        num_full_blocks_in_row,
        num_leftover_tiles,
        leftover_width_in_row,
    };

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) src0_is_dram
    };

    bool out_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t)log2(stick_size) : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) out_is_dram,
        (std::uint32_t) stick_size_is_power_of_two,
        (std::uint32_t) log2_stick_size,
    };

    // Tilized reader
    tt_metal::KernelID unary_reader_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

    // Untilized writer
    tt_metal::KernelID unary_writer_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_stick_layout_split_rows_interleaved.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

    vector<uint32_t> compute_args = {
        uint32_t(num_tiles / num_tiles_per_block), // per_core_block_cnt
        uint32_t(num_tiles_per_block) // per_core_block_tile_cnt
    };

    auto untilize_kernel_id = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/untilize.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_args}
    );

    tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        {src0_buffer->address(),
        uint32_t(num_tiles), 0 }
    );

    tt_metal::SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        writer_kernel_args
    );

    auto override_runtime_args_callback = [
        reader_kernel_id=unary_reader_kernel_id,
        writer_kernel_id=unary_writer_kernel_id
    ](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
            SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
        }

        {
            auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
            SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

inline std::tuple<> split_blocks_across_cores(CoreCoord grid_size, uint32_t nblocks) {
    uint32_t ncores_x = grid_size.x;
    uint32_t ncores_y = grid_size.y;
    uint32_t ncores = ncores_x * ncores_y;
    uint32_t nblocks_per_core = nblocks;
    uint32_t nblocks_per_core_cliff = 0;
   std::set<CoreRange> core_range, core_range_cliff;
    if (nblocks <= ncores) {
        nblocks_per_core = 1;
        ncores = nblocks;
        ncores_y = ceil((float) ncores / ncores_x);
        uint32_t ncores_y_cliff = ncores - (ncores_y - 1) * ncores_x;
        if (ncores_y_cliff == ncores_y) {
            core_range.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 1)});
        } else if (ncores_y_cliff < ncores_y) {
            core_range.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 2)});
            core_range.insert(CoreRange{.start = CoreCoord(0, ncores_y - 1), .end = CoreCoord(ncores_y_cliff - 2, ncores_y - 1)});
            core_range_cliff.insert(CoreRange{.start = CoreCoord(ncores_y_cliff - 2, ncores_y - 1), .end = CoreCoord(ncores_y_cliff - 1, ncores_y - 1)});
        } else {
            TT_ASSERT(false, "Something went really wrong in splitting blocks across cores!!");
        }
    } else {
        nblocks_per_core = ceil((float) nblocks / ncores);
        nblocks_per_core_cliff = nblocks - nblocks_per_core * (ncores - 1);
        if (nblocks_per_core_cliff < nblocks_per_core) {
            core_range.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 2, ncores_y - 1)});
            core_range_cliff.insert(CoreRange{.start = CoreCoord(ncores_x - 2, ncores_y - 1), .end = CoreCoord(ncores_x - 1, ncores_y - 1)});
        } else if (nblocks_per_core_cliff == nblocks_per_core) {
            core_range.insert(CoreRange{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 1)});
        } else {
            TT_ASSERT(false, "Somehting went really wrong in splitting blocks across cores (case else)")
        }
    }
    CoreRange all_cores{.start = CoreCoord(0, 0), .end = CoreCoord(ncores_x - 1, ncores_y - 1)};

    return std::make_tuple(ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff);
}

operation::ProgramWithCallbacks untilize_multi_core(const Tensor& a, Tensor& output) {
    tt_metal::Program program = tt_metal::Program();

    DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    Device *device = a.device();

    int32_t num_tiles = a.volume() / TILE_HW;
    uint32_t num_sticks = a.shape()[0] * a.shape()[1] * a.shape()[2];
    uint32_t stick_len = a.shape()[3];
    uint32_t num_tiles_in_row = stick_len / TILE_WIDTH;      // NOTE: assumed to be perfectly divisible
    uint32_t stick_size_nbytes = stick_len * a.element_size();

    // Ensure we don't intrude into storage space
    uint32_t max_l1_size = device->l1_size() / 2 - UNRESERVED_BASE;
    uint32_t max_tiles = max_l1_size / (2 * single_tile_size); // 2 CBs

    // Currently need the number of tiles in a row to be divisible by tiles in a block
    uint32_t num_tiles_per_block = 1;
    if (num_tiles_in_row <= max_tiles) {
        num_tiles_per_block = num_tiles_in_row;
    } else {
        for(uint32_t n_t = max_tiles; n_t > 0; n_t--) {
            if (num_tiles_in_row % n_t == 0) {
                num_tiles_per_block = n_t;
                break;
            }
        }
    }
    uint32_t block_width_nbytes = num_tiles_per_block * TILE_WIDTH * a.element_size();
    uint32_t num_full_blocks_in_row = num_tiles_in_row / num_tiles_per_block;
    uint32_t num_leftover_tiles = num_tiles_in_row % num_tiles_per_block;
    uint32_t leftover_width_in_row = num_leftover_tiles * a.element_size();

    {
        log_debug(LogOp, "block_width_nbytes: {}", block_width_nbytes);
        log_debug(LogOp, "num_full_blocks_in_row: {}", num_full_blocks_in_row);
        log_debug(LogOp, "num_leftover_tiles: {}", num_leftover_tiles);
        log_debug(LogOp, "leftover_width_in_row: {}", leftover_width_in_row);
    }

    auto grid_size = device->compute_with_storage_grid_size();
    uint32_t ncores_x = grid_size.x;
    uint32_t ncores_y = grid_size.y;
    uint32_t nblocks = num_tiles / num_tiles_per_block;

    auto [ncores, all_cores, core_g1, core_g2, nblocks_per_core_g1, nblocks_per_core_g2] = split_blocks_across_cores(grid_size, nblocks);

    {
        log_debug(LogOp, "ncores: {}", ncores);
        log_debug(LogOp, "ncores_x: {}", ncores_x);
        log_debug(LogOp, "ncores_y: {}", ncores_y);
        log_debug(LogOp, "ntiles_per_core_g1: {}", ntiles_per_core_g1);
        log_debug(LogOp, "ntiles_per_core_g2: {}", ntiles_per_core_g2);
        log_debug(LogOp, "core_g1 size: {}", core_g1.ranges().size());
        log_debug(LogOp, "core_g2 size: {}", core_g2.ranges().size());
    }

    uint32_t src0_cb_index = CB::c_in0;
    uint32_t num_input_tiles = num_tiles_per_block;
    auto cb_src0 = CreateCircularBuffers(
        program,
        src0_cb_index,
        all_cores,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        cb_data_format
    );

    uint32_t output_cb_index = CB::c_out0;
    uint32_t num_output_tiles = num_tiles_per_block;
    auto cb_output = CreateCircularBuffers(
        program,
        output_cb_index,
        all_cores,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        cb_data_format
    );

    Buffer *src0_buffer = a.buffer();
    Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    // reader compile time args
    bool src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    vector<uint32_t> reader_ct_args = {
        (uint32_t) src0_is_dram
    };

    // writer compile time args
    bool out_is_dram = dst_buffer->buffer_type() == BufferType::DRAM ? 1 : 0;
    bool stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size_nbytes);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t) std::log2(stick_size_nbytes) : 0;
    vector<uint32_t> writer_ct_args = {
        (uint32_t) out_is_dram,
        (uint32_t) stick_size_is_power_of_two,
        (uint32_t) log2_stick_size,
    };

    uint32_t ncores_used = ncores;
    uint32_t nblocks_per_core = nblocks;
    uint32_t nblocks_cliff = 0;
    if (nblocks <= ncores) {
        // this is the good case
        ncores_used = nblocks;
        nblocks_per_core = 1;
    } else {
        // TODO: this case needs multiple blocks per core.
        nblocks_per_core = ceil((float) nblocks / ncores);
        nblocks_cliff = nblocks - nblocks_per_core * (ncores - 1);
    }

    {
        log_debug(LogOp, "nblocks: {}", nblocks);
        log_debug(LogOp, "ncores_used: {}", ncores_used);
        log_debug(LogOp, "nblocks_per_core: {}", nblocks_per_core);
        log_debug(LogOp, "nblocks_cliff: {}", nblocks_cliff);
    }

    // compute compile times args
    vector<uint32_t> compute_args = {
        (uint32_t) nblocks_per_core,    // per_core_block_cnt. NOTE: assumed to be perfectly divisible
        (uint32_t) num_tiles_per_block  // per_block_ntiles
    };

    // tilized reader
    KernelID unary_reader_kernel_id = CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_ct_args});

    // untilized writer
    KernelID unary_writer_kernel_id = CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_stick_layout_split_rows_interleaved.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_ct_args});

    auto untilize_kernel_id = CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/untilize.cpp",
        all_cores,
        ComputeConfig{
            .compile_args = compute_args});

    // 1D distribution of blocks across all cores
    uint32_t start_id = 0;
    for (int32_t i = 0; i < ncores; ++ i) {
        CoreCoord core(i % ncores_x, i / ncores_x);

    }

    // reader runtime args
    vector<uint32_t> reader_rt_args = {
        src0_buffer->address(),     // src_addr
        (uint32_t) num_tiles,       // ntiles
        0                           // start_id
    };

    // writer runtime args
    vector<uint32_t> writer_rt_args = {
        dst_buffer->address(),      // dst_addr
        num_sticks,                 // nsticks
        stick_size_nbytes,          // stick_size_nbytes
        num_tiles_per_block,        // ntiles_per_block
        block_width_nbytes,         // block_width_nbytes
        num_full_blocks_in_row,     // full blocks in a row
        num_leftover_tiles,         // UNUSED
        leftover_width_in_row,      // UNUSED
    };

    tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        all_cores,
        reader_rt_args
    );

    tt_metal::SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        all_cores,
        writer_rt_args
    );

    auto override_runtime_args_callback = [
        reader_kernel_id=unary_reader_kernel_id,
        writer_kernel_id=unary_writer_kernel_id
    ](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
            SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
        }

        {
            auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
            SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

void Untilize::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_ASSERT(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to untilize need to be on device!");
    TT_ASSERT(input_tensor_a.buffer() != nullptr , "Operands to untilize need to be allocated in buffers on device!");
    TT_ASSERT(input_tensor_a.dtype() == DataType::BFLOAT16, "Only bloat16 dataformat supported");
    TT_ASSERT(input_tensor_a.layout() == Layout::TILE, "Can only untilize tile major data");

    TT_ASSERT(input_tensor_a.volume() % TILE_HW == 0);
}

std::vector<Shape> Untilize::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    return {input_tensor_a.shape()};
}

std::vector<Tensor> Untilize::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor_a.dtype(), Layout::ROW_MAJOR, this->output_mem_config);
}

operation::ProgramWithCallbacks Untilize::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    if (this->use_multicore) {
        return {untilize_multi_core(input_tensor_a, output_tensor)};
    } else {
        return {untilize_single_core(input_tensor_a, output_tensor)};
    }
}

tt::stl::reflection::Attributes Untilize::attributes() const {
    return {
        {"output_mem_config", this->output_mem_config},
    };
}

Tensor untilize(const Tensor &input_tensor_a, const MemoryConfig& mem_config, bool use_multicore) {
    // No-op (Will do a tensor copy)
    if (input_tensor_a.layout() == Layout::ROW_MAJOR) {
        log_warning("Perf warning: Trying to untilize non-tilized data.");
        return input_tensor_a;
    }
    return operation::run_without_autoformat(Untilize{mem_config, use_multicore}, {input_tensor_a}).at(0);
}


operation::ProgramWithCallbacks untilize_with_unpadding_single_core(const Tensor &a, Tensor& output, const Shape &output_tensor_start, const Shape &output_tensor_end) {

    const Shape output_shape = output.shape();

    tt_metal::Program program = tt_metal::Program();

    CoreRange core = {.start={0, 0}, .end={0, 0}};

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

    tt_metal::Buffer *src0_buffer = a.buffer();

    int32_t num_tiles = a.volume() / TILE_HW;

    // This should allocate a DRAM buffer on the device
    tt_metal::Device *device = a.device();

    tt_metal::Buffer *dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t num_padded_sticks = a.shape()[0] * a.shape()[1] * a.shape()[2];
    uint32_t num_unpadded_sticks = a.shape()[0] * a.shape()[1] * output_shape[2];
    uint32_t padded_stick_size = a.shape()[3] * a.element_size(); // Assuming bfloat16 dataformat
    uint32_t unpadded_stick_size = output_shape[3] * a.element_size();

    constexpr uint32_t alignment = 32;

    uint32_t num_tiles_in_row = a.shape()[3] / TILE_WIDTH;
    // Ensure we don't intrude into storage space
    uint32_t max_l1_size = a.device()->l1_size() / 2 - UNRESERVED_BASE;
    // Memory usage is 2 CBs of width W, plus buffer of size alignment + (W * datum size)
    uint32_t max_X = (max_l1_size - alignment) / (a.element_size() * TILE_HEIGHT * 2 + a.element_size());
    uint32_t max_tiles = max_X / TILE_WIDTH;

    // Currently need the number of tiles in a row to be divisible by tiles in a block
    uint32_t num_tiles_per_block = 1;
    if (num_tiles_in_row <= max_tiles) {
        num_tiles_per_block = num_tiles_in_row;
    } else {
        for(uint32_t n_t = max_tiles; n_t > 0; n_t--) {
            if (num_tiles_in_row % n_t == 0) {
                num_tiles_per_block = n_t;
                break;
            }
        }
    }
    uint32_t block_width = num_tiles_per_block * TILE_WIDTH;
    uint32_t block_row_size = block_width * a.element_size();
    uint32_t num_blocks_w_output = unpadded_stick_size / block_row_size;
    uint32_t num_blocks_w_input = padded_stick_size / block_row_size;
    uint32_t block_row_leftover_size = unpadded_stick_size - num_blocks_w_output * block_row_size;

    // Number of blocks that differ between input and output
    const uint32_t num_blocks_w_diff = num_blocks_w_input - num_blocks_w_output - (block_row_leftover_size > 0 ? 1 : 0);

    const uint32_t padded_Y_diff_blocks = (a.shape()[2] - output_shape[2]) / TILE_HEIGHT * num_blocks_w_input;
    const uint32_t padded_Z_diff_blocks = (a.shape()[1] - output_shape[1]) * a.shape()[2] / TILE_HEIGHT * num_blocks_w_input;
    const uint32_t padded_W_diff_blocks = (a.shape()[0] - output_shape[0]) * a.shape()[1] * a.shape()[2] / TILE_HEIGHT * num_blocks_w_input;
    const uint32_t num_leftover_Y = output_shape[2] - output_shape[2] / TILE_HEIGHT * TILE_HEIGHT;

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = num_tiles_per_block;
    auto cb_src0 = tt_metal::CreateCircularBuffers(
        program,
        src0_cb_index,
        core,
        num_input_tiles,
        num_input_tiles * single_tile_size,
        cb_data_format
    );

    uint32_t output_cb_index = 16; // output operands start at index 16
    uint32_t num_output_tiles = num_tiles_per_block;
    auto cb_output = tt_metal::CreateCircularBuffers(
        program,
        output_cb_index,
        core,
        num_output_tiles,
        num_output_tiles * single_tile_size,
        cb_data_format
    );

    vector<uint32_t> writer_kernel_args = {
        dst_buffer->address(),
        output_shape[0],
        padded_W_diff_blocks,
        output_shape[1],
        padded_Z_diff_blocks,
        output_shape[2],
        padded_Y_diff_blocks,
        num_leftover_Y,
        output_shape[3],
        unpadded_stick_size,
        padded_stick_size,
        num_blocks_w_input,
        num_blocks_w_output,
        num_blocks_w_diff,
        block_row_size,
        block_row_leftover_size
    };

    bool src0_is_dram = src0_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t) src0_is_dram
    };

    bool out_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    uint32_t stick_size = unpadded_stick_size;
    bool stick_size_is_power_of_two = is_power_of_two_at_least_32(stick_size);
    uint32_t log2_stick_size = stick_size_is_power_of_two ? (std::uint32_t)log2(stick_size) : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) out_is_dram,
        (std::uint32_t) stick_size_is_power_of_two,
        (std::uint32_t) log2_stick_size,
    };

    // Tilized reader
    tt_metal::KernelID unary_reader_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default, .compile_args = reader_compile_time_args});

    // Untilized writer
    tt_metal::KernelID unary_writer_kernel_id = tt_metal::CreateDataMovementKernel(
        program,
        "tt_metal/kernels/dataflow/writer_unary_unpad_dims_split_rows.cpp",
        core,
        tt_metal::DataMovementConfig{.processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default, .compile_args = writer_compile_time_args});

    vector<uint32_t> compute_args = {
        uint32_t(num_tiles / num_tiles_per_block),
        uint32_t(num_tiles_per_block)
    };

    auto untilize_kernel_id = tt_metal::CreateComputeKernel(
        program,
        "tt_metal/kernels/compute/untilize.cpp",
        core,
        tt_metal::ComputeConfig{.compile_args = compute_args}
    );

    tt_metal::SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        {src0_buffer->address(),
        uint32_t(num_tiles), 0}
    );

    tt_metal::SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        writer_kernel_args
    );

    auto override_runtime_args_callback = [
        reader_kernel_id=unary_reader_kernel_id,
        writer_kernel_id=unary_writer_kernel_id
    ](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);
        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
            SetRuntimeArgs(program, reader_kernel_id, core, runtime_args);
        }

        {
            auto runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
            SetRuntimeArgs(program, writer_kernel_id, core, runtime_args);
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

void UntilizeWithUnpadding::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_ASSERT(input_tensor_a.storage_type() == StorageType::DEVICE, "Operandsneed to be on device!");
    TT_ASSERT(input_tensor_a.buffer() != nullptr , "Operands need to be allocated in buffers on device!");
    TT_ASSERT(input_tensor_a.dtype() == DataType::BFLOAT16, "Only bloat16 dataformat supported");
    TT_ASSERT(input_tensor_a.layout() == Layout::TILE, "Can only untilize tile major data");

    TT_ASSERT(
        (this->output_tensor_start[0] == 0 && this->output_tensor_start[1] == 0 && this->output_tensor_start[2] == 0 && this->output_tensor_start[3] == 0),
        "On device unpadding only supports unpadding at end of dims"
    );

    TT_ASSERT(input_tensor_a.volume() % TILE_HW == 0);
    TT_ASSERT(this->output_tensor_start[0] < input_tensor_a.shape()[0]);
    TT_ASSERT(this->output_tensor_end[0] < input_tensor_a.shape()[0]);
    TT_ASSERT(this->output_tensor_start[1] < input_tensor_a.shape()[1]);
    TT_ASSERT(this->output_tensor_end[1] < input_tensor_a.shape()[1]);
    TT_ASSERT(this->output_tensor_start[2] < input_tensor_a.shape()[2]);
    TT_ASSERT(this->output_tensor_end[2] < input_tensor_a.shape()[2]);
    TT_ASSERT(this->output_tensor_start[3] < input_tensor_a.shape()[3]);
    TT_ASSERT(this->output_tensor_end[3] < input_tensor_a.shape()[3]);

    // Check if start shape is <= end shape
    TT_ASSERT(this->output_tensor_start[0] <= this->output_tensor_end[0]);
    TT_ASSERT(this->output_tensor_start[1] <= this->output_tensor_end[1]);
    TT_ASSERT(this->output_tensor_start[2] <= this->output_tensor_end[2]);
    TT_ASSERT(this->output_tensor_start[3] <= this->output_tensor_end[3]);

    TT_ASSERT(((this->output_tensor_end[3] - this->output_tensor_start[3] + 1) % 2 == 0), "Can only unpad to row major tensor of even width");

}
std::vector<Shape> UntilizeWithUnpadding::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    Shape output_tensor_shape = {
        this->output_tensor_end[0] - this->output_tensor_start[0] + 1,
        this->output_tensor_end[1] - this->output_tensor_start[1] + 1,
        this->output_tensor_end[2] - this->output_tensor_start[2] + 1,
        this->output_tensor_end[3] - this->output_tensor_start[3] + 1,
    };
    return {output_tensor_shape};
}
std::vector<Tensor> UntilizeWithUnpadding::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor_a.dtype(), Layout::ROW_MAJOR, this->output_mem_config);
}

operation::ProgramWithCallbacks UntilizeWithUnpadding::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    return {untilize_with_unpadding_single_core(input_tensor_a, output_tensor, output_tensor_start, output_tensor_end)};
}

tt::stl::reflection::Attributes UntilizeWithUnpadding::attributes() const {
    return {
        {"output_tensor_start", this->output_tensor_start},
        {"output_tensor_end", this->output_tensor_end},
        {"output_mem_config", this->output_mem_config},
    };
}

Tensor untilize_with_unpadding(const Tensor &input_tensor_a, const Shape &output_tensor_start, const Shape &output_tensor_end, const MemoryConfig& mem_config) {
    // No-op (Will do a tensor copy)
    // TODO: We need to run asserts before this
    const Shape output_tensor_shape = {
        output_tensor_end[0] - output_tensor_start[0] + 1,
        output_tensor_end[1] - output_tensor_start[1] + 1,
        output_tensor_end[2] - output_tensor_start[2] + 1,
        output_tensor_end[3] - output_tensor_start[3] + 1,
    };
    if (input_tensor_a.layout() != Layout::TILE) {
        if (input_tensor_a.shape() == output_tensor_shape) {
            log_warning("Perf warning: Untilize with unpadding called on already untilized tensor of target shape");
            return input_tensor_a;
        } else {
            TT_ASSERT(false, "Cannot untilize and unpad input which is not tilized");
        }
    }
    return operation::run_without_autoformat(UntilizeWithUnpadding{output_tensor_start, output_tensor_end, mem_config}, {input_tensor_a}).at(0);
}

}  // namespace tt_metal

}  // namespace tt
