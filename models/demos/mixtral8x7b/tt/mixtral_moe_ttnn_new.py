import torch
import torch.nn as nn
import ttnn
import time


def top_2(gate_logits_1SB8, top_2_mask, expert_mask):
    weights_ex0_1SB1 = ttnn.experimental.tensor.max(gate_logits_1SB8, dim=3)
    cond0 = ttnn.eq(gate_logits_1SB8, ttnn.repeat_interleave(weights_ex0_1SB1, 8, dim=3))

    gate_logits_1SB8 = ttnn.where(cond0, top_2_mask, gate_logits_1SB8)
    weights_ex1_1SB1 = ttnn.experimental.tensor.max(gate_logits_1SB8, dim=3)
    cond1 = ttnn.eq(gate_logits_1SB8, ttnn.repeat_interleave(weights_ex1_1SB1, 8, dim=3))

    weights_1SBK = ttnn.concat([weights_ex0_1SB1, weights_ex1_1SB1], dim=3)

    cond0 = ttnn.sum(cond0 * expert_mask, dim=3)
    cond1 = ttnn.sum(cond1 * expert_mask, dim=3)
    experts = cond0 + cond1

    return weights_1SBK, experts, ttnn.concat([cond0, cond1], dim=3)


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

    def forward(self, inputs):
        output_B1SD = []
        start_time = time.time()
        for i in [0, 3, 4, 7]:  # range(len(self.devices)):
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

            weights_1SBK, batch_ids_1SB1, selected_experts_1SBK = top_2(
                gate_logits_1SB8, self.top_2_mask[i], self.expert_mask[i]
            )
            weights_1SBK = ttnn.softmax(weights_1SBK - ttnn.experimental.tensor.max(weights_1SBK, dim=3), dim=-1)
            print("done top 2", ttnn.to_torch(weights_1SBK))
            # send to host
            batch_ids_B_torch = ttnn.to_torch(batch_ids_1SB1)
            # convert batch_ids to list of indices
            batch_ids_1b_torch = batch_ids_B_torch.view(-1).nonzero().view(1, -1).to(torch.int)
            print("NON ZERO BATCHES", batch_ids_1b_torch)
            batch_ids_1b = ttnn.from_torch(
                batch_ids_1b_torch, dtype=ttnn.uint32, device=self.devices[i], layout=ttnn.TILE_LAYOUT
            )
            batch_ids_1b = ttnn.to_torch(batch_ids_1b)
            batch_ids_1b = ttnn.from_torch(
                batch_ids_1b, dtype=ttnn.uint32, device=self.devices[i], layout=ttnn.ROW_MAJOR_LAYOUT
            )
            print("tt batches", batch_ids_1b)
            b = batch_ids_1b.shape[1]

            # in case, no batch selected this head
            if b == 0:
                print("no batch selected this head")
                output_i_B1SD = ttnn.zeros(
                    input_shape=ttnn.Shape([32, 1, 1, 4096]),
                    device=self.devices[i],
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )

            else:
                # slice input
                input_i_1SBH = ttnn.to_layout(input_i_1SBH, layout=ttnn.ROW_MAJOR_LAYOUT)
                input_i_1bH = ttnn.embedding(batch_ids_1b, input_i_1SBH)
                print("done input slicing embedding")
                input_i_11bH = ttnn.reshape(input_i_1bH, ttnn.Shape([1, 1, b, 4096]))
                print("done input slicing embedding reshape", input_i_11bH)
                # input_i_11bH = ttnn.experimental.tensor.pad(input_i_11bH, [1, 1, 32, 4096], [0, 0, 0, 0], pad_value=0.0)
                # print("done input slicing embedding padding")
                # input_i_11bH = ttnn.to_layout(input_i_11bH, layout=ttnn.TILE_LAYOUT)
                input_i_11bH = ttnn.experimental.tensor.tilize_with_zero_padding(input_i_11bH)
                print("done input slicing", input_i_11bH)

                # slice weights
                weights_1SB1 = ttnn.experimental.tensor.sum(weights_1SBK * selected_experts_1SBK, dim=3)
                print("done weight creation", ttnn.to_torch(weights_1SB1))
                weights_1SB2 = ttnn.experimental.tensor.untilize_with_unpadding(
                    weights_1SB1,
                    output_tensor_start=ttnn.Shape([0, 0, 0, 0]),
                    output_tensor_end=ttnn.Shape([0, 0, 31, 1]),
                    output_mem_config=ttnn.L1_MEMORY_CONFIG,
                    use_pack_untilize=False,
                )
                weights_1b2 = ttnn.embedding(batch_ids_1b, weights_1SB2)
                weights_11b2 = ttnn.reshape(weights_1b2, ttnn.Shape([1, 1, b, 2]))
                weights_11b1 = ttnn.split(
                    ttnn.experimental.tensor.tilize_with_zero_padding(weights_11b2), split_size=1, dim=3
                )[0]
                print("done weight slicing", ttnn.to_torch(weights_11b1))

                # MLP
                results_11bD = expert_i_HD(input_i_11bH) * weights_11b1
                print("done expert MLP")
                results_b1SD = ttnn.experimental.tensor.untilize_with_unpadding(
                    results_11bD,
                    output_tensor_start=ttnn.Shape([0, 0, 0, 0]),
                    output_tensor_end=ttnn.Shape([0, 0, b - 1, 4095]),
                    output_mem_config=ttnn.L1_MEMORY_CONFIG,
                    use_pack_untilize=False,
                )
                # results_b1SD = ttnn.to_layout(results_11bD, layout=ttnn.ROW_MAJOR_LAYOUT)
                print("done expert MLP to layout", results_b1SD)
                results_b1SD = ttnn.permute(results_b1SD, (2, 1, 0, 3))
                print("done expert MLP permute", results_b1SD)

                # create output tensor with results_bO at batch positions batch_ids_b
                if True:
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
                    print("done zero tensor creation")
                    batch_ids_1b = ttnn.reshape(batch_ids_1b, ttnn.Shape([1, 1, 1, b]))
                    output_i_B1SD = ttnn.experimental.tensor.indexed_fill(batch_ids_1b, output_i_B1SD, results_b1SD)
                print("done output tensor creation", output_i_B1SD)

            output_B1SD.append(output_i_B1SD)

            print(f"finished device {i}, time: {time.time() - start_time} ")
        # all gather
        print(f"started ALL GATHER, time: {time.time() - start_time} ")
        output_B1SD_gathered = ttnn.experimental.tensor.all_gather(output_B1SD, dim=2, num_links=1)
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
