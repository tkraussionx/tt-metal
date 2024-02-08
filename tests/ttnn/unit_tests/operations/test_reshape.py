import ttnn
import torch
import time

device_id = 0
device = ttnn.open(device_id)

# Reshape into specified shape
start = time.time()
x = torch.rand((3, 224, 224), dtype=torch.bfloat16)
x = ttnn.from_torch(x)
x = ttnn.to_device(x, device, memory_config=ttnn.L1_MEMORY_CONFIG)
# x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
# output = ttnn.reshape(x,(224, 3, 224))
output = ttnn.reshape(x, (224, 224, 3))
# a or b cannot be reshaped because they are in TILE_LAYOUT. Only ROW_MAJOR_LAYOUT can be reshaped
end = time.time()
execution_time = end - start
print("Total time: {}ms".format(execution_time * 1000))

ttnn.close(device)
