# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


class TtMixtralMLP(torch.nn.Module):
    def __init__(self, device, dtype):
        super().__init__()

        self.device = device
        self.dtype = dtype

        self.w1 = ttnn.from_torch(
            torch.randn(4096, 14336), dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT
        )
        self.w2 = ttnn.from_torch(
            torch.randn(14336, 4096), dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT
        )
        self.w3 = ttnn.from_torch(
            torch.randn(4096, 14336), dtype=ttnn.bfloat8_b, device=device, layout=ttnn.TILE_LAYOUT
        )
        self.get_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """
        w1 -> gate_proj
        w2 -> down_proj
        w3 -> up_proj
        HF reference: self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        """
        w1_out = ttnn.linear(
            x,
            self.w1,
            activation="silu",
            core_grid=ttnn.CoreGrid(y=7, x=8),
            use_1d_systolic_array=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.get_compute_kernel_config,
        )

        w3_out = ttnn.matmul(
            x,
            self.w3,
            core_grid=ttnn.CoreGrid(y=7, x=8),
            use_1d_systolic_array=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.get_compute_kernel_config,
        )
        w2_in = ttnn.mul(w1_out, w3_out)
        w2_out = ttnn.matmul(
            w2_in,
            self.w2,
            core_grid=ttnn.CoreGrid(y=7, x=8),
            use_1d_systolic_array=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.get_compute_kernel_config,
        )

        return w2_out


def top_2(gate_logits_1SB8, top_2_mask, expert_mask, ones_1118, ones_11B1):
    # get the highest value and position
    weights_ex0_1SB1 = ttnn.experimental.tensor.max(gate_logits_1SB8, dim=3)
    exp_0_repeated = ttnn.matmul(
        weights_ex0_1SB1, ones_1118, core_grid=ttnn.CoreGrid(y=1, x=8), use_1d_systolic_array=True
    )
    cond0 = ttnn.eq(gate_logits_1SB8, exp_0_repeated)

    # mask out the maximum value
    gate_logits_1SB8_masked = ttnn.where(cond0, top_2_mask, gate_logits_1SB8)

    # get the second highest value and position
    weights_ex1_1SB1 = ttnn.experimental.tensor.max(gate_logits_1SB8_masked, dim=3)
    exp_1_repeated = ttnn.matmul(
        weights_ex1_1SB1, ones_1118, core_grid=ttnn.CoreGrid(y=1, x=8), use_1d_systolic_array=True
    )
    cond1 = ttnn.eq(gate_logits_1SB8, exp_1_repeated)

    # calculate the softmax
    weights_exp = ttnn.exp(weights_ex1_1SB1 - weights_ex0_1SB1)
    weights_1SB1_pre_softmax = ttnn.reciprocal(ones_11B1 + weights_exp)

    # select whether a batch for was selected first or second for the i-th head
    cond0 = ttnn.matmul(cond0, expert_mask, core_grid=ttnn.CoreGrid(y=1, x=8), use_1d_systolic_array=True)
    cond1 = ttnn.matmul(cond1, expert_mask, core_grid=ttnn.CoreGrid(y=1, x=8), use_1d_systolic_array=True)

    # calculate the weight
    weights_1SB1 = cond0 * weights_1SB1_pre_softmax - cond1 * (weights_1SB1_pre_softmax - ones_11B1)

    return weights_1SB1


class TtMoeLayer(torch.nn.Module):
    def __init__(self, devices, dtype):
        super().__init__()
        self.devices = devices
        self.experts = [TtMixtralMLP(device, dtype) for device in devices]
        self.dtype = dtype

        self.gates_H8 = [
            ttnn.from_torch(torch.randn(1, 1, 4096, 8), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            for device in devices
        ]
        self.num_devices = len(devices)

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

        self.ones_1118 = [
            ttnn.from_torch(torch.ones(1, 1, 1, 8), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            for device in self.devices
        ]

        self.ones_11B1 = [
            ttnn.from_torch(
                torch.ones(1, 1, 32, 1),
                device=device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            for device in self.devices
        ]
        reduce_mask_torch = torch.zeros(1, 1, 32, 32 * len(self.devices))
        for i in range(32):
            reduce_mask_torch[:, :, i, range(i, 32 * len(self.devices), 32)] = 1
        self.reduce_mask = [
            ttnn.from_torch(reduce_mask_torch, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            for device in self.devices
        ]
        self.compute_kernel = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, inputs):
        """
        inputs: (seq_len, 1, batch, hidden_dim)

        Tensors are postfixed with 4 characters that represent their 4-D shape:
        B : batch_size (32)
        H : dim (4096)
        S : seq len (1)
        """
        output_11BH = []
        for i in range(len(self.devices)):
            self.devices[i] = self.devices[i]
            input_i_1SBH = inputs[i]
            expert_i_HH = self.experts[i]

            # get logits for the experts
            gate_logits_1SB8 = ttnn.linear(
                input_i_1SBH,
                self.gates_H8[i],
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel,
                use_1d_systolic_array=True,
                core_grid=ttnn.CoreGrid(y=1, x=8),
            )

            # get weights for top-2 experts
            weights_1SB1 = top_2(
                gate_logits_1SB8, self.top_2_mask[i], self.expert_mask[i], self.ones_1118[i], self.ones_11B1[i]
            )

            # MLP and masking
            results_11BH = expert_i_HH(input_i_1SBH) * weights_1SB1

            # output_11BH.append(results_11BH)
        return output_11BH


from models.utility_functions import get_devices_for_t3000

import tt_lib as ttl


def test_moe():
    all_devices = ttl.device.CreateDevices([i for i in range(8)])
    devices = get_devices_for_t3000(all_devices, 8)
    # [0, 7, 6, 1, 2, 5, 4, 3]
    # devices = [devices[4]]
    model = TtMoeLayer(devices, dtype=ttnn.bfloat8_b)
    inputs = [
        ttnn.from_torch(
            torch.randn(1, 1, 32, 4096),
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        for device in devices
    ]
    for iter in range(32 * 50):
        output = model(inputs)
        print("done iter, ", iter)


test_moe()
