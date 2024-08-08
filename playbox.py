import torch
import ttnn

device_id = 0
device = ttnn.open_device(device_id=device_id)

size = [1, 1, 32, 32 * 32]
torch_input_tensor = torch.Tensor(size=size).uniform_(-10, 10)
input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
output_tensor = ttnn.tilize(input_tensor)
torch_output_tensor = ttnn.to_torch(output_tensor)

print(output_tensor)

ttnn.close_device(device)
