import torch
import torch.nn as nn
import ttnn
import time


def concat(tt_0, tt_1, mask_0, mask_1):
    tt_0 = ttnn.repeat_interleave(tt_0, repeats=2, dim=3)
    tt_1 = ttnn.repeat_interleave(tt_1, repeats=2, dim=3)
    output = tt_0 * mask_0 + tt_1 * mask_1
    return output


def top_2(gate_logits_1SB8, top_2_mask, expert_mask, mask_0, mask_1):
    weights_ex0_1SB1 = ttnn.experimental.tensor.max(gate_logits_1SB8, dim=3)
    cond0 = ttnn.eq(gate_logits_1SB8, ttnn.repeat_interleave(weights_ex0_1SB1, 8, dim=3))

    gate_logits_1SB8 = ttnn.where(cond0, top_2_mask, gate_logits_1SB8)
    weights_ex1_1SB1 = ttnn.experimental.tensor.max(gate_logits_1SB8, dim=3)
    cond1 = ttnn.eq(gate_logits_1SB8, ttnn.repeat_interleave(weights_ex1_1SB1, 8, dim=3))
    m = ttnn.reciprocal(weights_ex0_1SB1 + weights_ex1_1SB1)
    weights_1SBK = concat(m * weights_ex0_1SB1, m * weights_ex1_1SB1, mask_0, mask_1)

    cond0 = ttnn.sum(cond0 * expert_mask, dim=3)
    cond1 = ttnn.sum(cond1 * expert_mask, dim=3)

    return weights_1SBK, concat(cond0, cond1, mask_0, mask_1)


class TtMoeLayer(nn.Module):
    def __init__(self, devices, state_dict, experts, args, layer_num, dtype):
        super().__init__()
        assert len(experts) > 0
        self.devices = devices
        self.experts = experts
        self.args = args
        self.dtype = dtype

        gate_name = f"layers.{layer_num}.block_sparse_moe.gate.weight"
        self.gates_H8 = [
            ttnn.as_tensor(
                state_dict[gate_name].permute(1, 0),
                dtype=ttnn.bfloat16,
                device=device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=args.weight_cache_path(dtype) / gate_name,
            )
            for device in self.devices
        ]
        self.num_devices = len(devices)
        self.compute_kernel = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.top_2_mask = [
            ttnn.experimental.tensor.tilize_with_zero_padding(
                ttnn.experimental.tensor.full(
                    ttnn.Shape([1, 1, 32, 8]), fill_value=0.0, data_type=ttnn.bfloat16, device=device
                )
            )
            for device in self.devices
        ]
        self.expert_mask = []
        for i in range(len(self.devices)):
            torch_tensor = torch.zeros(1, 1, 32, 8)
            torch_tensor[:, :, :, i] = 1
            self.expert_mask.append(
                ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, device=self.devices[i], layout=ttnn.TILE_LAYOUT)
            )

        self.mask_0 = [
            ttnn.from_torch(
                torch.tensor([1, 0] * 32).view(1, 1, 32, 2),
                device=device,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
            )
            for device in self.devices
        ]

        self.mask_1 = [
            ttnn.from_torch(
                torch.tensor([[0, 1]] * 32).view(1, 1, 32, 2),
                device=device,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
            )
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
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel,
            )
            gate_logits_1SB8 = ttnn.softmax(
                gate_logits_1SB8 - ttnn.experimental.tensor.max(gate_logits_1SB8, dim=3), dim=-1
            )
            weights_1SBK, selected_experts_1SBK = top_2(
                gate_logits_1SB8, self.top_2_mask[i], self.expert_mask[i], self.mask_0[i], self.mask_1[i]
            )
            # weights_1SBK = ttnn.softmax(weights_1SBK - ttnn.experimental.tensor.max(weights_1SBK, dim=3), dim=-1)
            # weights_1SBK = weights_1SBK * ttnn.reciprocal(2 *ttnn.mean(weights_1SBK, dim=3, keepdim=True))
            print("done top 2")

            # mask weights
            weights_1SB1 = ttnn.experimental.tensor.sum(weights_1SBK * selected_experts_1SBK, dim=3)

            # MLP
            results_11BD = expert_i_HD(input_i_1SBH) * weights_1SB1
            print("done output tensor creation")

            for n, l in enumerate(
                [
                    weights_1SB1,
                    weights_1SBK,
                    selected_experts_1SBK,
                    gate_logits_1SB8,
                ]
            ):
                ttnn.deallocate(l)

            output_11BD.append(results_11BD)

            print(f"finished device {i}, time: {time.time() - start_time} ")
        # all gather
        print(f"started ALL GATHER, time: {time.time() - start_time} ")
        output_11BD_gathered = ttnn.experimental.tensor.all_gather(output_11BD, dim=1, num_links=1)
        print(f"finished ALL GATHER, time: {time.time() - start_time}")

        # sum on each device
        for i in range(len(output_11BD_gathered)):
            output_11BD_gathered[i] = ttnn.experimental.tensor.sum(output_11BD_gathered[i], dim=1)
        print("Reduce sum done")
        return output_11BD_gathered
