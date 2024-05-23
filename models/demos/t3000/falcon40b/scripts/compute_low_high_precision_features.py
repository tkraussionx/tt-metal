# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import matplotlib.pyplot as plt


def main():
    num_layers = 60
    alpha = 20.0
    S = 128
    H = 8192
    num_layers_above_alpha_threshold = num_layers * 0.25  # 25% of layers must have that feature to be considered

    ff1_is_above_alpha_mag = torch.zeros(num_layers, H)
    for i in range(num_layers):
        ff1_in0_torch_mag = torch.load(f"/home/jrock/tt-metal/activations_data/transformer.h.{i}__ff1_in0_torch_mag.pt")
        ff1_is_above_alpha_mag[i, :] = (ff1_in0_torch_mag.squeeze() > alpha).float()

    num_layers_above_alpha = torch.sum(ff1_is_above_alpha_mag, dim=0)
    enough_layeres_above_alpha = num_layers_above_alpha > num_layers_above_alpha_threshold
    indices = torch.nonzero(enough_layeres_above_alpha).squeeze()
    print(indices)

    # Create a line plot
    plt.plot(num_layers_above_alpha)

    # Add title and labels (optional)
    plt.title("Number of Layers Above Alpha")
    plt.xlabel("Feature Index")
    plt.ylabel("Number of Layers")

    plt.savefig("num_layers_above_alpha.png")


if __name__ == "__main__":
    main()
