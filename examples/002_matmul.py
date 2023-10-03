import tt_lib

def main():
    device_id = 0
    device = tt_lib.device.CreateDevice(device_id)
    tt_lib.device.SetDefaultDevice(device)

    input_tensor_a = tt_lib.operations.random((1, 1, 64, 256)).to(device, tt_lib.tensor.MemoryConfig(True, tt_lib.tensor.BufferType.L1))
    input_tensor_b = tt_lib.operations.random((1, 1, 256, 128)).to(device, tt_lib.tensor.MemoryConfig(True, tt_lib.tensor.BufferType.L1))
    input_tensor_c = tt_lib.operations.random((1, 1, 128, 96)).to(device, tt_lib.tensor.MemoryConfig(True, tt_lib.tensor.BufferType.L1))

    output_tensor = tt_lib.operations.primary.matmul(
        input_tensor_a, input_tensor_b,
        program_config=tt_lib.operations.primary.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(4, 2),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fused_activation=None,
        ),
        output_dtype=tt_lib.tensor.DataType.BFLOAT8_B,
        output_mem_config=tt_lib.tensor.MemoryConfig(True, tt_lib.tensor.BufferType.L1),
    )
    output_tensor = tt_lib.operations.primary.matmul(
        output_tensor, input_tensor_c,
        output_dtype=tt_lib.tensor.DataType.BFLOAT16
    )

    tt_lib.device.CloseDevice(device)

if __name__ == "__main__":
    main()
