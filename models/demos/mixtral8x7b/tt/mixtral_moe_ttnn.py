import torch
import torch.nn as nn
import ttnn
from typing import List


class TtMoeLayer(nn.Module):
    def __init__(self, experts, moe_args, devices, state_dict):
        super().__init__()
        assert len(experts) > 0
        self.experts = experts
        self.args = moe_args
        self.devices = devices
        self.gate = ttnn.from_torch(
            state_dict["gate.weight"], dtype=ttnn.bfloat16, device=self.devices[0], layout=ttnn.TILE_LAYOUT
        )
        self.gates_H8 = [ttnn.to_device(self.gate.clone(), device) for device in self.devices]

    def forward(self, inputs: ttnn.Tensor):
        output_BS1O = []
        for i in range(len(self.devices)):
            device_i = self.devices[i]
            input_i_BSH = ttnn.to_device(inputs, device_i)
            expert_i_HO = self.experts[i]
            gate_logits_BS8 = ttnn.matmul(input_i_BSH, self.gates_H8[i], core_grid=ttnn.CoreGrid(8, 8))
            # TODO: falling back to pytorch for now
            gate_logits_BS8_torch = ttnn.to_torch(gate_logits_BS8)
            weights_BSK, selected_experts_BSK = torch.topk(gate_logits_BS8_torch, self.args.num_experts_per_tok)
            weights_BSK = ttnn.from_torch(weights_BSK, dtype=ttnn.bfloat16, device=device_i, layout=ttnn.TILE_LAYOUT)
            # only choose i-th index
            selected_experts_0_B1 = ttnn.from_torch(
                selected_experts_BSK[:, :, 0], dtype=ttnn.bfloat16, device=device_i, layout=ttnn.TILE_LAYOUT
            )
            selected_experts_1_B1 = ttnn.from_torch(
                selected_experts_BSK[:, :, 1], dtype=ttnn.bfloat16, device=device_i, layout=ttnn.TILE_LAYOUT
            )
            weights_BSK = ttnn.softmax(weights_BSK, dim=2)

            head_pos_B1 = ttnn.eq(selected_experts_0_B1, selected_experts_1_B1)  # ttnn.eq(selected_experts_1_B1, i)
            batch_ids_B1 = ttnn.logical_or(
                ttnn.eq(selected_experts_0_B1, selected_experts_1_B1), selected_experts_0_B1
            )  # , ttnn.eq(selected_experts_1_B1, i))

            # send to host
            weights_BSK_torch = ttnn.to_torch(weights_BSK)
            batch_ids_B1_torch = ttnn.to_torch(batch_ids_B1)
            input_i_BSH_torch = ttnn.to_torch(input_i_BSH)
            head_pos_B1_torch = ttnn.to_torch(head_pos_B1)

            # convert batch_ids to list of indices
            batch_ids_1b_torch = batch_ids_B1_torch.view(-1).nonzero().view(-1)
            # slice input
            input_i_bSH_torch = input_i_BSH_torch[batch_ids_1b_torch, :, :]
            # slice weights
            weights_bS_torch = weights_BSK_torch[batch_ids_1b_torch, :, 0]  # head_pos_B1_torch[batch_ids_b_torch]]

            # send to device
            batch_ids_b = ttnn.from_torch(
                batch_ids_1b_torch, dtype=ttnn.uint16, device=device_i, layout=ttnn.TILE_LAYOUT
            )
            input_i_bSH = ttnn.from_torch(
                input_i_bSH_torch, dtype=ttnn.bfloat16, device=device_i, layout=ttnn.TILE_LAYOUT
            )
            weights_bS = ttnn.from_torch(
                weights_bS_torch, dtype=ttnn.bfloat16, device=device_i, layout=ttnn.TILE_LAYOUT
            )

            results_bSO = expert_i_HO(input_i_bSH) * weights_bS

            # create output tensor with results_bO at batch positions batch_ids_b
            output_i_BSO_torch = torch.zeros(32, 1, 4096, dtype=torch.bfloat16)
            results_bSO_torch = ttnn.to_torch(results_bSO)
            output_i_BSO_torch[batch_ids_1b_torch] = results_bSO_torch
            output_i_BS1O = ttnn.from_torch(
                output_i_BSO_torch.unsqueeze(2), dtype=ttnn.bfloat16, device=device_i, layout=ttnn.TILE_LAYOUT
            )

            output_BS1O.append(output_i_BS1O)
        """
        # all gather
        output_BS1O = ttnn.experimental.tensor.all_gather(output_BS1O, dim=3, num_links=2)
        print("output_BSO", output_BS.shape)

        # sum on each device
        for i in range(self.devices):
            device_i = self.devices[i]
            output_BS1O[i] = ttnn.sum(output_BS1O[i])
        """
        return output_BS1O
