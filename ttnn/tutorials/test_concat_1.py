import ttnn
import torch

device_id = 0
device = ttnn.open(device_id)

torch_input_tensor_a = torch.rand((256, 1056, 160, 16), dtype=torch.bfloat16)
torch_input_tensor_b = torch.rand((256, 1056, 160, 16), dtype=torch.bfloat16)
torch_output_tensor = torch.concat([torch_input_tensor_a, torch_input_tensor_b], dim=3)

input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

output = ttnn.concat([input_tensor_a, input_tensor_b], dim=3)
output_torch = ttnn.to_torch(output)
ttnn.deallocate(output)

print(output_torch.shape)

ttnn.close(device)
