template<typename T, typename... Args>
create_op(T t, Args... args) {
    assert_shapes<T>(args);
    assert_on_device<T>(args);
    create_cbs<T>();
    create_data_movement_kernels<T>();
    create_compute_kernels<T>();
}

template <typename T, typename... Args>
Tensor op(T t, Args... args) {
    tt_metal::Program *program = new tt_metal::Program();

    tt::Device *device = create_op<T, args>(...);
    tt_metal::CompileProgram(device, program);
    tt_metal::ConfigureDeviceWithProgram(device, program);
    tt_metal::LaunchKernels(device, program);
}


Tensor op(MatmulOp t, ...) {

}
