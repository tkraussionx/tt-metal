import tt_lib

def main():
    device_id = 0
    device = tt_lib.device.CreateDevice(device_id)
    tt_lib.device.SetDefaultDevice(device)

    input_tensor_a = tt_lib.operations.random((1, 1, 64, 256)).to(device)
    input_tensor_b = tt_lib.operations.random((1, 1, 256, 128)).to(device)
    input_tensor_c = tt_lib.operations.random((1, 1, 128, 96)).to(device)

    output_tensor = tt_lib.operations.primary.matmul(input_tensor_a, input_tensor_b)
    output_tensor = tt_lib.operations.primary.matmul(output_tensor, input_tensor_c)

    tt_lib.device.CloseDevice(device)

if __name__ == "__main__":
    main()
