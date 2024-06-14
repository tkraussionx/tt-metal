# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


class TtMixtralEmbedding(torch.nn.Module):
    def __init__(
        self,
        device_mesh,
        args,
        weight_cache_path,
        state_dict,
        dtype,
    ):
        super().__init__()

        base_name = "tok_embeddings.weight"
        torch_weight = state_dict[base_name]

        if args.dummy_weights:
            cache_name = None
        else:
            cache_name = weight_cache_path / base_name

        self.weights = ttnn.as_tensor(
            torch_weight,
            dtype=dtype,
            device=device_mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=args.get_model_config()["EMB_WEIGHTS_MEMCFG"],
            cache_file_name=cache_name,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device_mesh),
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.embedding(x, self.weights)
