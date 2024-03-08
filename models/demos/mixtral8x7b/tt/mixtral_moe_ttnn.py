import torch
import torch.nn as nn
import ttnn
import time


class TtMoeLayer(nn.Module):
    def __init__(self, experts, moe_args, devices, state_dict, num_devices, dtype):
        super().__init__()
        assert len(experts) > 0
        self.experts = experts
        self.args = moe_args
        self.devices = devices
        self.dtype = dtype
        self.gates_H8 = [
            ttnn.from_torch(
                state_dict["gate.weight"].permute(1, 0),
                dtype=self.dtype,
                device=device,
                layout=ttnn.TILE_LAYOUT,
            )
            for device in self.devices
        ]
        self.num_devices = num_devices

    def forward(self, inputs):
        output_BS1O = []
        results = []
        start_time = time.time()
        for i in range(len(self.devices)):
            print(f"started device {i}, time: {time.time() - start_time} ")
            self.devices[i] = self.devices[i]
            input_i_BSH = inputs[i]
            expert_i_HO = self.experts[i]
            gate_logits_BS8 = ttnn.matmul(input_i_BSH, self.gates_H8[i], core_grid=ttnn.CoreGrid(y=7, x=8))
            # TODO: falling back to pytorch for now
            # for i in range(len(self.devices)):
            gate_logits_BS8_torch = ttnn.to_torch(gate_logits_BS8)
            weights_BSK, selected_experts_BSK = torch.topk(gate_logits_BS8_torch, self.args.num_experts_per_tok)
            weights_BSK = ttnn.from_torch(
                weights_BSK, dtype=ttnn.bfloat16, device=self.devices[i], layout=ttnn.TILE_LAYOUT
            )
            # only choose i-th index
            selected_experts_0_1B = ttnn.from_torch(
                selected_experts_BSK[:, :, 0], dtype=ttnn.bfloat16, device=self.devices[i], layout=ttnn.TILE_LAYOUT
            )
            selected_experts_1_1B = ttnn.from_torch(
                selected_experts_BSK[:, :, 1], dtype=ttnn.bfloat16, device=self.devices[i], layout=ttnn.TILE_LAYOUT
            )
            print("weight bsk pre sm", weights_BSK.shape)
            # for i in range(len(self.devices)):
            weights_BSK = ttnn.softmax(weights_BSK, dim=-1)
            comp = ttnn.Tensor(
                ttnn.experimental.tensor.full(
                    ttnn.Shape([32, 32]),
                    i,
                )
            )
            comp = ttnn.to_layout(comp, layout=ttnn.TILE_LAYOUT)
            comp = ttnn.to_device(comp, device=self.devices[i])
            head_pos_1B = ttnn.eq(selected_experts_1_1B, comp)
            batch_ids_1B = ttnn.logical_or(ttnn.eq(selected_experts_0_1B, comp), head_pos_1B)

            # for i in range(len(self.devices)):
            # send to host
            weights_BSK_torch = ttnn.to_torch(weights_BSK)
            batch_ids_1B_torch = ttnn.to_torch(batch_ids_1B)
            input_i_BSH_torch = ttnn.to_torch(input_i_BSH)
            head_pos_1B_torch = ttnn.to_torch(head_pos_1B)

            # convert batch_ids to list of indices
            batch_ids_1b_torch = batch_ids_1B_torch.view(-1).nonzero().view(-1)
            # slice input
            input_i_bSH_torch = input_i_BSH_torch[batch_ids_1b_torch, :, :]
            # slice weights
            head_pos_1b_torch = head_pos_1B_torch[batch_ids_1b_torch].to(dtype=torch.int64).view(-1)
            weights_bS_torch = weights_BSK_torch[batch_ids_1b_torch, :, head_pos_1b_torch].unsqueeze(2)
            print("weights_bS_torch", weights_bS_torch.shape, weights_BSK_torch.shape, weights_BSK.shape)

            # send to device
            batch_ids_b = ttnn.from_torch(
                batch_ids_1b_torch, dtype=ttnn.uint16, device=self.devices[i], layout=ttnn.TILE_LAYOUT
            )
            input_i_bSH = ttnn.from_torch(
                input_i_bSH_torch, dtype=ttnn.bfloat16, device=self.devices[i], layout=ttnn.TILE_LAYOUT
            )
            weights_bS = ttnn.from_torch(
                weights_bS_torch, dtype=ttnn.bfloat16, device=self.devices[i], layout=ttnn.TILE_LAYOUT
            )
            # for i in range(len(self.devices)):
            results_bSO = expert_i_HO(input_i_bSH) * weights_bS

            # for i in range(len(self.devices)):
            # create output tensor with results_bO at batch positions batch_ids_b
            output_i_BSO_torch = torch.zeros(32, 1, 4096, dtype=torch.bfloat16)
            results_bSO_torch = ttnn.to_torch(results_bSO)
            results.append(
                (results_bSO_torch, weights_bS_torch, batch_ids_1b_torch, head_pos_1b_torch, weights_BSK_torch)
            )
            output_i_BSO_torch[batch_ids_1b_torch] = results_bSO_torch
            output_i_BS1O = ttnn.from_torch(
                output_i_BSO_torch.unsqueeze(2), dtype=ttnn.bfloat16, device=self.devices[i], layout=ttnn.TILE_LAYOUT
            )
            output_BS1O.append(output_i_BS1O)
            for l in [
                input_i_bSH,
                weights_bS,
                batch_ids_b,
                weights_BSK,
                comp,
                gate_logits_BS8,
                results_bSO,
                head_pos_1B,
                batch_ids_1B,
                selected_experts_0_1B,
                selected_experts_1_1B,
            ]:
                ttnn.deallocate(l)
            print(f"finished device {i}, time: {time.time() - start_time} ")
        # all gather
        print(f"started ALL GATHER, time: {time.time() - start_time} ")
        num_links = 1
        if self.num_devices == 4:
            for i in range(4):
                output_BS1O[i] = ttnn.experimental.tensor.add(output_BS1O[i], output_BS1O[4 + i])
            output_BS1O = output_BS1O[:4]
            num_links = 2
        output_BS1O_gathered = ttnn.experimental.tensor.all_gather(output_BS1O, dim=2, num_links=num_links)
        print(f"finished ALL GATHER, time: {time.time() - start_time}")

        # sum on each device
        for i in range(len(output_BS1O_gathered)):
            output_BS1O_gathered[i] = ttnn.experimental.tensor.reduce(
                output_BS1O_gathered[i],
                ttnn.experimental.tensor.ReduceOpMath.SUM,
                ttnn.experimental.tensor.ReduceOpDim.H,
                1.0,
            )
            print("Reduce sum done")
        return output_BS1O_gathered, selected_experts_BSK, results
