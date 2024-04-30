from time import time
import torch
import ttnn
from models.lock import WaitLock

lock = WaitLock()
d = ttnn.open_device(device_id=0)

xs = [
    ttnn.from_torch(
        torch.randn((1, 1, r, 4096), dtype=torch.bfloat16),
        device=d,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    for r in [32, 64, 128, 256]
]
w = ttnn.from_torch(
    torch.randn((4096, 14336), dtype=torch.bfloat16),
    device=d,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat8_b,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
)

for stage in ("Warmup", "Perf"):
    print(f"{stage}:")
    for x in xs:
        print(f"  {x.shape}:")
        ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
        y = ttnn.linear(
            x, w, core_grid=ttnn.CoreGrid(x=8, y=7), use_1d_systolic_array=True, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        _ = ttnn.to_torch(y)
        ttnn.deallocate(y)

ttnn.close_device(d)
lock.release()

# Increasing number of rows doesn't meaningfully affect speed (30us, 7%) until >128 rows (4 tiles)
#                              OP CODE  DEVICE FW DURATION [ns]  % DRAM (240)  % FPU (82)        GB/s  ...  INPUT_0_X INPUT_1_Y  INPUT_1_X  OUTPUT_0_Y  OUTPUT_0_X
# 62   tt::operations::primary::Matmul                   596322     41.029466   61.484156   98.470719  ...       4096      4096      14336         256       14336
# 38   tt::operations::primary::Matmul                   595558     41.082100   61.563030   98.597040  ...       4096      4096      14336         256       14336
# 32   tt::operations::primary::Matmul                   390951     62.582711   46.891241  150.198506  ...       4096      4096      14336         128       14336
# 56   tt::operations::primary::Matmul                   389465     62.821494   47.070154  150.771587  ...       4096      4096      14336         128       14336
# 50   tt::operations::primary::Matmul                   373256     65.549578   24.557110  157.318988  ...       4096      4096      14336          64       14336
# 26   tt::operations::primary::Matmul                   371774     65.810878   24.655002  157.946107  ...       4096      4096      14336          64       14336
# 44   tt::operations::primary::Matmul                   366260     66.801653   12.513090  160.323967  ...       4096      4096      14336          32       14336
# 20   tt::operations::primary::Matmul                   364612     67.103588   12.569648  161.048611  ...       4096      4096      14336          32       14336
