import torch
import ttnn

device_id = 0
device = ttnn.open(device_id)


# torch_input_tensor_a = torch.rand( (256, 1056, 160, 16) , dtype=torch.bfloat16)
# torch_input_tensor_a = torch.rand( (256, 528, 80, 16) , dtype=torch.bfloat16)
# torch_input_tensor_a = torch.rand( (256, 264, 40, 16) , dtype=torch.bfloat16)
# torch_input_tensor_a = torch.rand( (256, 132, 20, 16) , dtype=torch.bfloat16)
torch_input_tensor_a = torch.rand((256, 66, 10, 16), dtype=torch.bfloat16)

tensor = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)


output = ttnn.relu(tensor)
output = ttnn.to_torch(output)

print(output.shape)
print(output[1, :2, :2, 0])

ttnn.close(device)
