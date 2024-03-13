# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn


def create_model_config(num_users, hidden_size):
    configs = {}
    
    row = 1
    col = 8

    configs["sharded"] = ttnn.create_sharded_memory_config(
        shape=(1, 1, num_users, hidden_size*2),
        core_grid=ttnn.CoreGrid(y=row, x=col),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=False,
    )
    
    configs["sharded_large"] = ttnn.create_sharded_memory_config(
        shape=(1, 1, num_users, hidden_size*2 * 16),
        core_grid=ttnn.CoreGrid(y=row, x=col),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=False,
    )
    
    configs["sharded_rank"] = ttnn.create_sharded_memory_config(
        shape=(1, 1, hidden_size*2 // 32, hidden_size*2),
        core_grid=ttnn.CoreGrid(y=row, x=col),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=False,
    )
    return configs


def get_model_config(configs, model_config_str):
    return configs[model_config_str]
