import ttnn
import torch

device_id = 0
device = ttnn.open(device_id)

torch_input_tensor_a = torch.rand((256, 132, 20, 32), dtype=torch.bfloat16)
torch_input_tensor_b = torch.rand((256, 132, 20, 64), dtype=torch.bfloat16)
torch_output_tensor = torch.concat([torch_input_tensor_a, torch_input_tensor_b], dim=3)

input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)


input_tensor_a = ttnn.to_device(input_tensor_a, device, memory_config=ttnn.L1_MEMORY_CONFIG)
input_tensor_b = ttnn.to_device(input_tensor_b, device, memory_config=ttnn.L1_MEMORY_CONFIG)

output = ttnn.concat([input_tensor_a, input_tensor_b], dim=3)
output_torch = ttnn.to_torch(output)
output = ttnn.to_device(output, device, memory_config=ttnn.L1_MEMORY_CONFIG)
output = ttnn.to_layout(output, layout=ttnn.TILE_LAYOUT)
ttnn.deallocate(output)
ttnn.deallocate(input_tensor_a)
ttnn.deallocate(input_tensor_b)


print(output_torch.shape)


torch_input_tensor_a = torch.rand((256, 264, 40, 32), dtype=torch.bfloat16)
torch_input_tensor_b = torch.rand((256, 264, 40, 32), dtype=torch.bfloat16)
torch_output_tensor = torch.concat([torch_input_tensor_a, torch_input_tensor_b], dim=3)

input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

input_tensor_a = ttnn.to_device(input_tensor_a, device, memory_config=ttnn.L1_MEMORY_CONFIG)
input_tensor_b = ttnn.to_device(input_tensor_b, device, memory_config=ttnn.L1_MEMORY_CONFIG)

output2 = ttnn.concat([input_tensor_a, input_tensor_b], dim=3)
output_torch = ttnn.to_torch(output2)
output2 = ttnn.to_device(output2, device, memory_config=ttnn.L1_MEMORY_CONFIG)
output2 = ttnn.to_layout(output2, layout=ttnn.TILE_LAYOUT)
ttnn.deallocate(output2)
ttnn.deallocate(input_tensor_a)
ttnn.deallocate(input_tensor_b)


print(output_torch.shape)


torch_input_tensor_a = torch.rand((256, 528, 80, 16), dtype=torch.bfloat16)
torch_input_tensor_b = torch.rand((256, 528, 80, 32), dtype=torch.bfloat16)
torch_output_tensor = torch.concat([torch_input_tensor_a, torch_input_tensor_b], dim=3)

input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

input_tensor_a = ttnn.to_device(input_tensor_a, device, memory_config=ttnn.L1_MEMORY_CONFIG)
input_tensor_b = ttnn.to_device(input_tensor_b, device, memory_config=ttnn.L1_MEMORY_CONFIG)


output3 = ttnn.concat([input_tensor_a, input_tensor_b], dim=3)
output_torch = ttnn.to_torch(output3)
output3 = ttnn.to_device(output3, device, memory_config=ttnn.L1_MEMORY_CONFIG)
output3 = ttnn.to_layout(output3, layout=ttnn.TILE_LAYOUT)
ttnn.deallocate(output3)
ttnn.deallocate(input_tensor_a)
ttnn.deallocate(input_tensor_b)


print(output_torch.shape)


torch_input_tensor_a = torch.rand((256, 1056, 160, 16), dtype=torch.bfloat16)
torch_input_tensor_b = torch.rand((256, 1056, 160, 16), dtype=torch.bfloat16)
torch_output_tensor = torch.concat([torch_input_tensor_a, torch_input_tensor_b], dim=3)

input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

input_tensor_a = ttnn.to_device(input_tensor_a, device, memory_config=ttnn.L1_MEMORY_CONFIG)
input_tensor_b = ttnn.to_device(input_tensor_b, device, memory_config=ttnn.L1_MEMORY_CONFIG)


output4 = ttnn.concat([input_tensor_a, input_tensor_b], dim=3)
output_torch = ttnn.to_torch(output4)
output4 = ttnn.to_device(output4, device, memory_config=ttnn.L1_MEMORY_CONFIG)
output4 = ttnn.to_layout(output4, layout=ttnn.TILE_LAYOUT)
ttnn.deallocate(output4)
ttnn.deallocate(input_tensor_a)
ttnn.deallocate(input_tensor_b)


print(output_torch.shape)


torch_input_tensor_a = torch.rand((256, 1056, 160, 16), dtype=torch.bfloat16)
torch_input_tensor_b = torch.rand((256, 1056, 160, 16), dtype=torch.bfloat16)
torch_output_tensor = torch.concat([torch_input_tensor_a, torch_input_tensor_b], dim=3)

input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

input_tensor_a = ttnn.to_device(input_tensor_a, device, memory_config=ttnn.L1_MEMORY_CONFIG)
input_tensor_b = ttnn.to_device(input_tensor_b, device, memory_config=ttnn.L1_MEMORY_CONFIG)


output5 = ttnn.concat([input_tensor_a, input_tensor_b], dim=3)
output_torch = ttnn.to_torch(output5)
output5 = ttnn.to_device(output5, device, memory_config=ttnn.L1_MEMORY_CONFIG)
output5 = ttnn.to_layout(output5, layout=ttnn.TILE_LAYOUT)
ttnn.deallocate(output5)
ttnn.deallocate(input_tensor_a)
ttnn.deallocate(input_tensor_b)


print(output_torch.shape)

ttnn.close(device)
