import torch
import ttnn

device_id = 0
device = ttnn.open_device(device_id=device_id)

torch.manual_seed(0)

torch_input_tensor_a = torch.rand((32, 32), dtype=torch.bfloat16)
torch_input_tensor_b = torch.rand((32, 32), dtype=torch.bfloat16)

input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
