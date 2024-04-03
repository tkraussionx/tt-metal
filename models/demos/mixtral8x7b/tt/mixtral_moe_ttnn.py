import torch
import torch.nn as nn
import ttnn
import time


def top_2(gate_logits_1SB8, top_2_mask, expert_mask, id_18, ones_11B1):
    weights_ex0_1SB1 = ttnn.experimental.tensor.max(gate_logits_1SB8, dim=3)
    cond0 = ttnn.eq(gate_logits_1SB8, ttnn.matmul(weights_ex0_1SB1, id_18))

    gate_logits_1SB8 = ttnn.where(cond0, top_2_mask, gate_logits_1SB8)
    weights_ex1_1SB1 = ttnn.experimental.tensor.max(gate_logits_1SB8, dim=3)
    cond1 = ttnn.eq(gate_logits_1SB8, ttnn.matmul(weights_ex1_1SB1, id_18))

    weights_1SB1_pre_softmax = ttnn.reciprocal(ones_11B1 + ttnn.exp(weights_ex1_1SB1 - weights_ex0_1SB1))

    cond0 = ttnn.matmul(cond0, expert_mask)
    cond1 = ttnn.matmul(cond1, expert_mask)

    weights_1SB1 = cond0 * weights_1SB1_pre_softmax - cond1 * (weights_1SB1_pre_softmax - ones_11B1)
    return weights_1SB1


class TtMoeLayer(nn.Module):
    def __init__(self, devices, state_dict, experts, args, layer_num, dtype):
        super().__init__()
        assert len(experts) > 0
        self.devices = devices
        self.experts = experts
        self.args = args
        self.dtype = dtype
        self.model_config = args.get_model_config()

        gate_name = f"layers.{layer_num}.feed_forward.gate.weight"
        self.gates_H8 = [
            ttnn.as_tensor(
                state_dict[gate_name].permute(1, 0),
                dtype=ttnn.bfloat16,
                device=device,
                layout=self.model_config["GATE_W_LAYOUT_TILE"],
                memory_config=self.model_config["GATE_WEIGHTS_MEMCFG"],
                cache_file_name=args.weight_cache_path(dtype) / gate_name,
            )
            for device in self.devices
        ]
        self.num_devices = len(devices)
        self.compute_kernel = args.get_compute_kernel_attn_config()

        # TODO Should we add the layout of the masks below to model_config?
        self.top_2_mask = [
            ttnn.from_torch(
                torch.full((1, 1, 32, 8), fill_value=torch.finfo(torch.float).min),
                dtype=ttnn.bfloat16,
                device=device,
                layout=ttnn.TILE_LAYOUT,
            )
            for device in self.devices
        ]
        self.expert_mask = []
        for i in range(len(self.devices)):
            torch_tensor = torch.zeros(1, 1, 8, 1)
            torch_tensor[:, :, i, :] = 1
            self.expert_mask.append(
                ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, device=self.devices[i], layout=ttnn.TILE_LAYOUT)
            )

        self.id_18 = [
            ttnn.from_torch(torch.ones(1, 1, 1, 8), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            for device in self.devices
        ]

        self.ones_11B1 = [
            ttnn.from_torch(torch.ones(1, 1, 32, 1), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            for device in self.devices
        ]
        reduce_mask_torch = torch.zeros(1, 1, 32, 256)
        for i in range(32):
            reduce_mask_torch[:, :, i, range(i, 256, 32)] = 1
        self.reduce_mask = [
            ttnn.from_torch(reduce_mask_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            for device in self.devices
        ]

    def forward(self, inputs):
        output_11BD = []
        start_time = time.time()
        for i in range(len(self.devices)):
            print(f"started device {i}, time: {time.time() - start_time} ")
            self.devices[i] = self.devices[i]
            input_i_1SBH = inputs[i]
            expert_i_HD = self.experts[i]

            gate_logits_1SB8 = ttnn.linear(
                input_i_1SBH,
                self.gates_H8[i],
                memory_config=self.model_config["GATE_MM_OUTPUT_MEMCFG"],
                compute_kernel_config=self.compute_kernel,
                use_1d_systolic_array=True,
            )

            if False:
                weights_1SBK, selected_experts_1SBK = torch.topk(ttnn.to_torch(gate_logits_1SB8)[0, 0], 2)
                selected_experts_1SBK = (selected_experts_1SBK == i).to(torch.int)
                weights_1SBK = ttnn.from_torch(
                    weights_1SBK.unsqueeze(0).unsqueeze(0),
                    device=self.devices[i],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                )
                selected_experts_1SBK = ttnn.from_torch(
                    selected_experts_1SBK.unsqueeze(0).unsqueeze(0),
                    device=self.devices[i],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                )
                weights_1SBK = ttnn.softmax(weights_1SBK, dim=-1)
                weights_1SB1 = ttnn.experimental.tensor.sum(weights_1SBK * selected_experts_1SBK, dim=3)

            else:
                weights_1SB1 = top_2(
                    gate_logits_1SB8, self.top_2_mask[i], self.expert_mask[i], self.id_18[i], self.ones_11B1[i]
                )

            results_11BD = expert_i_HD(input_i_1SBH) * weights_1SB1
            print("done output tensor creation")

            output_11BD.append(results_11BD)

            print(f"finished device {i}, time: {time.time() - start_time} ")
        # all gather
        print(f"started ALL GATHER, time: {time.time() - start_time} ")
        output_11BD_gathered = ttnn.experimental.tensor.all_gather(output_11BD, dim=2, num_links=1)
        print(f"finished ALL GATHER, time: {time.time() - start_time}")

        # sum on each device
        for i in range(len(output_11BD_gathered)):
            output_11BD_gathered[i] = ttnn.matmul(self.reduce_mask[i], output_11BD_gathered[i])
        print("Reduce sum done")
        return output_11BD_gathered
