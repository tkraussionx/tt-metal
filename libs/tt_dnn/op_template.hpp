

Tensor op(Args... args) {};

//template<typename... Args>
//create_op(OpEnum t, Args... args) {
//  assert_shapes(t, args);
//  assert_on_device(t);
//  create_cbs(t);
//  create_data_movement_kernels(t);
//  create_compute_kernels(t);
//}
//
//template<typename... Args>
//Tensor op(OpEnum T, Args... args) {
//  tt_metal::Program *program = new tt_metal::Program();
//
//  tt::Device *device = create_op(t, args);
//  tt_metal::CompileProgram(device, program);
//  tt_metal::ConfigureDeviceWithProgram(device, program);
//  tt_metal::LaunchKernels(device, program);
//}
//
//template<typename... Args>
//assert_shapes(OpEnum t, Args... args) {
//  switch (t) {
//    case OpEnum::UnaryOp:
//      //
//  }
//}
