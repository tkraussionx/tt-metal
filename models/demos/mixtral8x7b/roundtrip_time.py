from time import time
import torch
import ttnn
from models.lock import WaitLock

w = WaitLock()
with ttnn.manage_device(device_id=0) as d:
    x = ttnn.from_torch(torch.randn((32, 1, 1, 4096), dtype=torch.bfloat16), device=d, layout=ttnn.TILE_LAYOUT)

    # Warmup
    host = ttnn.to_torch(x)
    x = ttnn.from_torch(host, device=d, layout=ttnn.TILE_LAYOUT)

    start = time()
    for _ in range(1000):
        host = ttnn.to_torch(x)
        x = ttnn.from_torch(host, device=d, layout=ttnn.TILE_LAYOUT)
    duration = time() - start

    print(f"Host roundtrip: {duration:.1f} ms")  # 69 ms
w.release()
