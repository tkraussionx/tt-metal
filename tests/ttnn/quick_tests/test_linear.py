import ttnn
import torch
import time
import tt_lib

torch.set_printoptions(profile="full")

device_id = 0
device = ttnn.open(device_id)

activations = torch.randn((10, 64, 32), dtype=torch.bfloat16)
weights = torch.randn((32, 128), dtype=torch.bfloat16)
bias = torch.randn((128,), dtype=torch.bfloat16)

act = ttnn.from_torch(activations)
act = ttnn.to_device(act, device)
act = ttnn.to_layout(act, ttnn.TILE_LAYOUT)
wei = ttnn.from_torch(weights)
wei = ttnn.to_device(wei, device)
wei = ttnn.to_layout(wei, ttnn.TILE_LAYOUT)
bi = ttnn.from_torch(bias)
bi = ttnn.to_device(bi, device)
bi = ttnn.to_layout(bi, ttnn.TILE_LAYOUT)
output_1 = ttnn.linear(act, wei, bias=bi)
output_2 = ttnn.linear(act, wei)

print("\n\n")
print(
    "output1 which has the bias and output2 which doesn't have the bias are equal! ",
    False not in (ttnn.to_torch(output_2) == ttnn.to_torch(output_1)),
)
print("\n\n")


ttnn.close(device)
