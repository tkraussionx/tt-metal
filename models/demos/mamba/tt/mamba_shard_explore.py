import torch
import ttnn
import sys

def AbarMulH(num_users, hidden_dim):

    abar = torch.rand((1,1,num_users,hidden_dim), dtype=torch.bfloat16)
    h = torch.rand((1,1,num_users,hidden_dim), dtype=torch.bfloat16)

    cfg = ttnn.create_sharded_memory_config(shape=(1,1,num_users,hidden_dim), core_grid=ttnn.CoreGrid(y=num_users//32, x=8), strategy=ttnn.ShardStrategy.BLOCK, orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=False)

    abar = ttnn.from_torch(abar, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)
    h = ttnn.from_torch(h, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)

    output_tensor = ttnn.mul(abar, h, memory_config=cfg)

def mamba_ssm_block(num_users, hidden_dim):
    # call elementwise mul of Abar and h
    AbarMulH(num_users, hidden_dim)



# number of users is arg 1
num_users = int(sys.argv[1])

# hidden dimension is arg 2
hidden_dim = int(sys.argv[2])

device_id = 0
device = ttnn.open_device(device_id=device_id)

torch.manual_seed(0)
ttnn.enable_program_cache()

mamba_ssm_block(num_users, hidden_dim)


ttnn.close_device(device)