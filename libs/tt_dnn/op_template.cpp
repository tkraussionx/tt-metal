

tt::Device create_op(Args... args) {
  assert_shapes(args);
  assert_on_device(args);
  create_cbs(args);
  create_data_movement_kernels(args);
  create_compute_kernels(args);

  return device
}

Tensor op(Args... args) {
  tt_metal::Program *program = new tt_metal::Program();

  tt::Device *device = create_op(args);

  tt_metal::CompileProgram(device, program);
  tt_metal::ConfigureDeviceWithProgram(device, program);
  tt_metal::LaunchKernels(device, program);
}


void create_cbs(tt_metal::Program *program, tt_metal::Device *device, std::map<tt_xy_pair, std::map<uint32_t, std::tuple<uint32_t, uint32_t, DataFormat>>> cb_args) {
  for (const auto& [core, cb_arg_dict]: cb_args) {
    for (const auto& [cb_id, cb_arg]]: cb_args) {

      uint32_t num_input_tiles = cb_arg.get(0);
      uint32_t cb_size = cb_arg.get(1);
      DataFormat data_format  = cb_arg.get(2);

      auto cb = tt_metal::CreateCircularBuffer(
          program,
          device,
          cb_id,
          core,
          num_input_tiles,
          cb_size,
          data_format
      );
    }
  }
}

void create_data_movement_kernels(tt_metal::Program *program, tt_metal::Device *device, std::map<tt_metal::CoreBlocks, std::tuple<std::string, tt_metal::DataMovementProcessor, tt_metal::NOC>> data_movement_creation_args, data_movement_compile_time_args, data_movement_runtime_args ) {

}
