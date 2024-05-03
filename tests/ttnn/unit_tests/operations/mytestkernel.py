import torch
import ttnn
import pytest

# set ENV before build
# export TT_METAL_DPRINT_CORES=all
# export TT_METAL_DPRINT_ETH_CORES=all
# export TT_METAL_DPRINT_FILE=log.txt
# export TT_METAL_DPRINT_CHIPS=0
# export TT_METAL_DPRINT_RISCVS=TR0

device_id = 0
device = ttnn.open_device(device_id=device_id)
# torch_tensor = torch.rand(3, 4)
# numbers = torch.arange(1, 11)
# result = numbers.unsqueeze(0).unsqueeze(0).expand((1, 1, 32, 10))
result = torch.arange(1, 32 * 32 + 1).reshape(1, 1, 32, 32)
torch_tensor = result.bfloat16()
ttnn_tensor = ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
output_tensor = ttnn.neg(ttnn_tensor)
output = ttnn.to_torch(output_tensor)
torch.set_printoptions(linewidth=200, threshold=10000, precision=5, sci_mode=False, edgeitems=17)
print(f"\ntorch tensor:")
# Print the tensor
for i in range(32):
    print(torch_tensor[:, :, i : i + 1 :, :], "\n")
print(f"shape: {ttnn_tensor.shape}")
print(f"layout: {ttnn_tensor.layout}")
print(f"dtype: {ttnn_tensor.dtype}")
# print(f"tensor: {ttnn_tensor}")
print(f"Tensor in row-major layout (default?):\nShape {output.shape}\nLayout: {output.layout}\n{output}")

print(
    "hw0_32_16() = .h0 = 0, .h1 = 32 (include upto 32 rows), .hs = 16 (skips every 16 row), .w0 = 0, .w1 = 32(include upto 32cols), .ws = 16 (skip every 16 cols)"
)
print("hw0_32_16() gives 0,0 tile1- 0,16 tile2 - 16,0 tile3- 16,16 tile4 values first element of each subtile")
print("1 17 \n512 528")
print("\nhw0_32_8() { return SliceRange{ .h0 = 0, .h1 = 32, .hs = 8, .w0 = 0, .w1 = 32, .ws = 8 }; }")
print("skip every 8 row and columnwise")
print("1 9 17 25 \n256 264 272 280 \n512 520 528 536 \n768 776 784 792\n")
print("h0_w0_32() - print an entire row")
print("h0_w0_32() - SliceRange{ .h0 = 0, .h1 = 1, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }")
print("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32")
print("\n alter row h0 to print the row you want SliceRange{ .h0 = 31, .h1 = 32, .hs = 1, .w0 = 0, .w1 = 32, .ws = 1 }")
print(
    "\n992 992 996 996 996 1000 1000 1000 1000 1000 1004 1004 1004 1008 1008 1008 1008 1008 1012 1012 1012 1016 1016 1016 1016 1016 1020 1020 1020 1024 1024 1024"
)
ttnn.close_device(device)
