import torch
import torch.nn as nn
import ttnn
from typing import List


class TtMoeLayer(nn.Module):
    def __init__(self, experts, moe_args, devices):
        super().__init__()
        assert len(experts) > 0
        self.experts = experts
        self.args = moe_args
        self.devices = devices
        self.gate = ttnn.linear(self.args.dim, self.args.num_experts, core_grid=ttnn.CoreGrid(8, 8))
        self.gates = [self.gate.to_device(device) for device in self.devices]

    def forward(self, inputs: ttnn.Tensor):
        output_BO = []
        for i in range(self.devices):
            device_i = self.devices[i]
            input_i_BH = inputs.to_device(device_i)
            expert_i_HO = self.experts[i]
            gate_logits_BO = self.gates[i](input_i_BH)

            # TODO: falling back to pytorch for now
            gate_logits_BO_torch = ttnn.to_torch(gate_logits_BO)
            weights_BK, selected_experts_BK = torch.topk(gate_logits_BO_torch, self.args.num_experts_per_tok)
            weights_BK = ttnn.from_torch(weights_BK, dtype=ttnn.bfloat16, device=device_i)
            # only choose i-th index
            selected_experts_B1 = ttnn.from_torch(selected_experts_BK[:, i], dtype=ttnn.int8, device=device_i)

            weights_BK = ttnn.softmax(weights_BK, dim=1, dtype=ttnn.bfloat16)

            batch_ids_B1 = ttnn.eq(selected_experts_B1, i)

            # send to host
            weights_BK_torch = ttnn.to_torch(weights_BK)
            batch_ids_B1_torch = ttnn.to_torch(batch_ids_B1)
            input_i_BH_torch = ttnn.to_torch(input_i_BH)

            # convert batch_ids to list of indices
            batch_ids_b_torch = torch.where(batch_ids_B1_torch).nonzero()
            # slice input
            input_i_bH_torch = input_i_BH_torch[batch_ids_b]
            # slice weights
            weights_b1_torch = weights_BK_torch[batch_ids_b, i]

            # send to device
            batch_ids_b = ttnn.from_torch(batch_ids_b_torch, dtype=ttnn.int8, device=device_i)
            input_i_bH = ttnn.from_torch(input_i_bH_torch, dtype=ttnn.bfloat16, device=device_i)
            weights_b1 = ttnn.from_torch(weights_b1_torch, dtype=ttnn.bfloat16, device=device_i)

            results_bO = weights_b1 * expert_i_HO(input_i_bH)

            # create output tensor with results_bO at batch positions batch_ids_b
            output_i_BO_torch = torch.zeros_like(inputs)
            results_bO_torch = ttnn.to_torch(results_bO)
            output_i_BO_torch[batch_ids_b_torch] = results_bO_torch
            output_i_BO = ttnn.from_torch(output_i_BO_torch, dtype=ttnn.bfloat16, device=device_i)

            output_BO.append(output_i_BO)

        output_BO = ttnn.experimental.tensor.all_gather(output_BO)

        # sum on each device
        for i in range(self.devices):
            device_i = self.devices[i]
            output_BO[i] = ttnn.sum(output_BO[i])

        return output_BO
