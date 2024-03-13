import torch
import torch.nn as nn
import ttnn
import time


def top_2(gate_logits_1SB8, device):
    onehot_88 = ttnn.from_torch(
        (1 - torch.eye(8, 8)),
        device=device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    selected_experts_0_1S1B = ttnn.experimental.tensor.unpad(
        ttnn.experimental.tensor.argmax(gate_logits_1SB8, dim=3),
        [0, 0, 0, 0],
        [0, 0, 0, 31],
        output_mem_config=ttnn.L1_MEMORY_CONFIG,
    )
    weights_ex0_1SB1 = ttnn.experimental.tensor.max(gate_logits_1SB8, dim=3)
    # selected_experts_0_1S1B = ttnn.to_layout(ttnn.experimental.tensor.typecast(ttnn.to_layout(selected_experts_0_1S1B, layout=ttnn.TILE_LAYOUT), dtype=ttnn.uint32), layout=ttnn.ROW_MAJOR_LAYOUT)

    # TODO: Fix this
    selected_experts_0_1S1B_torch = ttnn.to_torch(selected_experts_0_1S1B)
    selected_experts_0_1S1B = ttnn.from_torch(
        selected_experts_0_1S1B_torch, dtype=ttnn.uint32, device=device, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    selected_experts_0_1S1B_tile = ttnn.from_torch(
        selected_experts_0_1S1B_torch, dtype=ttnn.uint32, device=device, layout=ttnn.TILE_LAYOUT
    )

    mask_1B8 = ttnn.embedding(ttnn.reshape(selected_experts_0_1S1B, ttnn.Shape([1, 32])), onehot_88)
    mask_1B8 = ttnn.to_layout(mask_1B8, layout=ttnn.TILE_LAYOUT)
    gate_logits_1SB8 = gate_logits_1SB8 * mask_1B8

    selected_experts_1_1S1B = ttnn.experimental.tensor.unpad(
        ttnn.experimental.tensor.argmax(gate_logits_1SB8, dim=3),
        [0, 0, 0, 0],
        [0, 0, 0, 31],
        output_mem_config=ttnn.L1_MEMORY_CONFIG,
    )
    weights_ex1_1SB1 = ttnn.experimental.tensor.max(gate_logits_1SB8, dim=3)
    weights_1SBK = ttnn.concat([weights_ex0_1SB1, weights_ex1_1SB1], dim=3)

    # selected_experts_1_1S1B = ttnn.experimental.tensor.typecast(ttnn.to_layout(selected_experts_1_1S1B, layout=ttnn.TILE_LAYOUT), dtype=ttnn.uint32)
    # TODO: Fix this
    selected_experts_1_1S1B_torch = ttnn.to_torch(selected_experts_1_1S1B)
    selected_experts_1_1S1B = ttnn.from_torch(
        selected_experts_1_1S1B_torch, dtype=ttnn.uint32, device=device, layout=ttnn.TILE_LAYOUT
    )

    selected_experts_1SBK = ttnn.permute(
        ttnn.concat([selected_experts_0_1S1B_tile, selected_experts_1_1S1B], dim=2), (0, 1, 3, 2)
    )
    return weights_1SBK, selected_experts_1SBK


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

    def forward(self, inputs):
        output_B1SD = []
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
            # gate_logits_1SB8 = ttnn.reshape(gate_logits_1SB8, ttnn.Shape([1, 1, 32, 8]))
            # TODO: falling back to pytorch for now
            # for i in range(len(self.devices)):
            if True:
                gate_logits_1SB8_torch = ttnn.to_torch(gate_logits_1SB8)
                weights_1SBK, selected_experts_1SBK = torch.topk(gate_logits_1SB8_torch, self.args.num_experts_per_tok)
                weights_1SBK = ttnn.from_torch(
                    weights_1SBK, dtype=ttnn.bfloat16, device=self.devices[i], layout=ttnn.TILE_LAYOUT
                )
                selected_experts_1SBK = ttnn.from_torch(
                    selected_experts_1SBK, dtype=ttnn.bfloat16, device=self.devices[i], layout=ttnn.TILE_LAYOUT
                )
            else:
                top_2(gate_logits_1SB8, self.devices[i])

            # for i in range(len(self.devices)):
            weights_1SBK = ttnn.softmax(weights_1SBK - ttnn.experimental.tensor.max(weights_1SBK, dim=3), dim=-1)
            comp = ttnn.experimental.tensor.full(
                ttnn.Shape([1, 1, 32, 2]),
                i,
            )

            comp = ttnn.to_layout(comp, layout=ttnn.TILE_LAYOUT)
            comp = ttnn.to_device(comp, device=self.devices[i])
            selected_experts_1SBK = ttnn.eq(selected_experts_1SBK, comp)
            batch_ids_1SB1 = ttnn.experimental.tensor.sum(selected_experts_1SBK, dim=3)
            batch_ids_1SB1 = batch_ids_1SB1 - 30

            # for i in range(len(self.devices)):
            # send to host
            batch_ids_B_torch = ttnn.to_torch(batch_ids_1SB1)
            # convert batch_ids to list of indices
            batch_ids_1b_torch = batch_ids_B_torch.view(-1).nonzero().view(1, -1)
            print("BATCHES", batch_ids_1b_torch)

            # in case, no batch selected this head
            if len(batch_ids_1b_torch) == 0:
                # TODO: double check this is correct
                output_i_B1SD = ttnn.from_torch(
                    [32, 1, 1, 4096], dtype=ttnn.bfloat16, device=self.devices[i], layout=ttnn.TILE_LAYOUT
                )

            else:
                # slice input
                batch_ids_1b = ttnn.from_torch(
                    batch_ids_1b_torch, dtype=ttnn.uint32, device=self.devices[i], layout=ttnn.ROW_MAJOR_LAYOUT
                )
                b = batch_ids_1b.shape[1]
                input_i_1SBH = ttnn.to_layout(input_i_1SBH, layout=ttnn.ROW_MAJOR_LAYOUT)
                input_i_1bH = ttnn.embedding(batch_ids_1b, input_i_1SBH)
                input_i_11bH = ttnn.reshape(input_i_1bH, ttnn.Shape([1, 1, b, 4096]))
                print("done input slicing")

                # slice weights
                weights_1SB1 = ttnn.experimental.tensor.sum(weights_1SBK * selected_experts_1SBK, dim=3)
                weights_1SB2 = ttnn.concat([weights_1SB1, weights_1SB1], dim=3)
                # TODO: Fix this
                weights_1SB2 = ttnn.to_torch(weights_1SB2)
                weights_1SB2 = ttnn.from_torch(
                    weights_1SB2, dtype=ttnn.bfloat16, device=self.devices[i], layout=ttnn.ROW_MAJOR_LAYOUT
                )
                weights_1SB2 = ttnn.to_layout(weights_1SB2, layout=ttnn.ROW_MAJOR_LAYOUT)
                weights_1b2 = ttnn.embedding(batch_ids_1b, weights_1SB2)
                weights_11b2 = ttnn.reshape(weights_1b2, ttnn.Shape([1, 1, b, 2]))
                weights_11b2 = ttnn.experimental.tensor.pad(weights_11b2, [1, 1, 32, 32], [0, 0, 0, 0], 0.0)
                weights_11b2 = ttnn.to_layout(weights_11b2, layout=ttnn.TILE_LAYOUT)
                weights_11b1 = ttnn.experimental.tensor.max(weights_11b2, dim=3)
                print("done weight slicing")

                # for i in range(len(self.devices)):
                input_i_11bH = ttnn.to_layout(input_i_11bH, layout=ttnn.TILE_LAYOUT)
                results_11bD = expert_i_HD(input_i_11bH) * weights_11b1
                results_b1SD = ttnn.to_layout(ttnn.permute(results_11bD, (2, 1, 0, 3)), layout=ttnn.ROW_MAJOR_LAYOUT)
                print("done expert MLP")

                # for i in range(len(self.devices)):
                # create output tensor with results_bO at batch positions batch_ids_b
                if False:
                    output_i_B1SD_torch = torch.zeros(32, 1, 1, 4096, dtype=torch.bfloat16)
                    results_b1SD_torch = ttnn.to_torch(results_b1SD)
                    output_i_B1SD_torch[batch_ids_1b_torch.view(-1)] = results_b1SD_torch
                    output_i_B1SD = ttnn.from_torch(
                        output_i_B1SD_torch, dtype=ttnn.bfloat16, device=self.devices[i], layout=ttnn.TILE_LAYOUT
                    )
                else:
                    output_i_B1SD = ttnn.zeros(
                        input_shape=ttnn.Shape([32, 1, 1, 4096]),
                        device=self.devices[i],
                        dtype=ttnn.bfloat16,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )
                    output_i_B1SD = ttnn.experimental.tensor.indexed_fill(batch_ids_1b, output_i_B1SD, results_b1SD)
                print("done output tensor creation")

            output_B1SD.append(output_i_B1SD)

            print(f"finished device {i}, time: {time.time() - start_time} ")
        # all gather
        print(f"started ALL GATHER, time: {time.time() - start_time} ")
        num_links = 1
        if self.num_devices == 4:
            for i in range(4):
                output_B1SD[i] = ttnn.experimental.tensor.add(output_B1SD[i], output_B1SD[4 + i])
            output_B1SD = output_B1SD[:4]
            num_links = 2
        output_B1SD_gathered = ttnn.experimental.tensor.all_gather(output_B1SD, dim=2, num_links=num_links)
        print(f"finished ALL GATHER, time: {time.time() - start_time}")

        # sum on each device
        for i in range(len(output_B1SD_gathered)):
            output_B1SD_gathered[i] = ttnn.experimental.tensor.reduce(
                output_B1SD_gathered[i],
                ttnn.experimental.tensor.ReduceOpMath.SUM,
                ttnn.experimental.tensor.ReduceOpDim.H,
                1.0,
            )
            print("Reduce sum done")
        return output_B1SD_gathered
