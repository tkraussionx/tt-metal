import ttnn

kernel_h = 2
kernel_w = 2
stride_h = 2
stride_w = 2
pad_h = 0
pad_w = 0
dilation_h = 1
dilation_w = 1
dtype = ttnn.bfloat8_b

device_id = 0
device = ttnn.open(device_id)
in_n = 4
in_h = 1056
in_w = 160
reader_patterns_cache = {}

max_pool = ttnn.MaxPool2D(
    kernel_size=(kernel_h, kernel_w),
    stride=(stride_h, stride_w),
    padding=(pad_h, pad_w),
    dilation=(dilation_h, dilation_w),
    dtype=dtype,
    device=device,
    batch_size=in_n,
    input_height=in_h,
    input_width=in_w,
    reader_patterns_cache=reader_patterns_cache,
)


act_shape = (in_n, 1, in_h * in_w, in_c)
act = torch.randn(act_shape, dtype=torch.bfloat16)
# act_shape = (in_n, 1, in_h * in_w, in_c)
act_reshaped = act_permuted.reshape(act_shape)
ttact = ttnn.from_torch(act_reshaped, dtype, layout=ttnn.TILE_LAYOUT)
ttact_d = max_pool.copy_input_to_device(ttact)
out_d = max_pool(ttact_d)
out_padded = max_pool.copy_output_from_device(out_d)
reader_patterns_cache.clear()
out_pytorch_padded = ttnn.to_torch(out_padded)
out_pytorch = out_pytorch_padded[:, :, :, :in_c]
out_pytorch = torch.permute(out_pytorch, (0, 3, 1, 2))  ## N, C, 1, HW


## reference
golden_pytorch = torch.nn.MaxPool2d(
    kernel_size,
    stride=stride,
    padding=padding,
    dilation=dilation,
    return_indices=False,
    ceil_mode=False,
)(act)


# out_pytorch = out_pytorch.reshape(golden_pytorch.shape)
ttnn.close(device)
