from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from models.utility_functions import comp_allclose_and_pcc
import torch
import ttnn
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

csv_file_path = "models/experimental/functional_t5/dumps/t5_debugging_plot.csv"


def comp_allclose(golden, calculated, rtol=1e-05, atol=1e-08):
    if golden.dtype != calculated.dtype:
        calculated = calculated.type(golden.dtype)

    atol_delta = torch.max(torch.abs(golden - calculated)).item()
    rtol_delta = torch.max(torch.abs(golden - calculated) / torch.abs(calculated)).item()
    return (atol_delta, rtol_delta)


def write_header():
    # Check if the file exists
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    with open(csv_file_path, "w", newline="") as file:
        writer = csv.writer(file)
        header = ["\t\tComparsion of TORCH and TTNN tensor\n"]
        writer.writerow(header)
        column_names = [
            "Sub-module",
            "PCC",
            "Tolerance",
            "Elements_count",
            "Count(element > tolerance)",
            "Atol",
            "Rtol",
            "Torch_max",
            "Torch_min",
            "Ttnn_max",
            "Ttnn_min",
        ]
        writer.writerow(column_names)  # Write only column names


def append_to_csv(data):
    with open(csv_file_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)


def write_data_to_csv(key, golden_tensor, computed_tensor):
    g_min = torch.min(golden_tensor)
    g_max = torch.max(golden_tensor)
    c_min = torch.min(computed_tensor)
    c_max = torch.max(computed_tensor)
    gt = torch.flatten(golden_tensor)
    ct = torch.flatten(computed_tensor.float())
    dt = torch.abs(gt - ct)
    pcc = check_with_pcc(golden_tensor, computed_tensor)[1]
    atol_delta, rtol_delta = comp_allclose(golden_tensor, computed_tensor)
    tolerance = atol_delta * 0.02  # Fixing tolerance value to 2% of max difference
    num_values_gt_tolerance = (dt > tolerance).sum().item()
    data = [
        f"{key}",
        pcc,
        tolerance,
        len(gt),
        num_values_gt_tolerance,
        atol_delta,
        rtol_delta,
        g_max.item(),
        g_min.item(),
        c_max.item(),
        c_min.item(),
    ]
    append_to_csv(data)


def make_histogram(key, golden_tensor, computed_tensor):
    sliced_g = golden_tensor.flatten().detach().numpy()
    sliced_c = (torch.flatten(computed_tensor.float())).detach().numpy()

    plt.figure(figsize=(15, 9))
    num_bins = 20

    color_x1 = "blue"
    color_x2 = "red"

    plt.hist(
        sliced_g, bins=num_bins, color=color_x1, alpha=0.3, label="torch_tensor"
    )  # alpha controls the transparency
    plt.hist(sliced_c, bins=num_bins, color=color_x2, alpha=0.5, label="tt_tensor")

    plt.xlim([min(min(sliced_g), min(sliced_c)), max(max(sliced_g), max(sliced_c))])
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"tests/ttnn/integration_tests/t5/new_{key}_histogram.png")


def plot_figure(key, golden_tensor, computed_tensor):
    plt.figure(figsize=(10, 6))
    sliced_g = golden_tensor[:, :].flatten().detach().numpy()
    sliced_c = (torch.flatten(computed_tensor[:, :].float())).detach().numpy()
    plt.scatter(range(len(sliced_g)), sliced_g, label="TORCH Tensor", marker="o", color="blue", s=1)
    plt.scatter(range(len(sliced_c)), sliced_c, label="TTNN Tensor", marker="o", color="orange", s=1)
    plt.scatter(
        range(len(sliced_c)), abs(sliced_c - sliced_g), label="Absolute Difference", marker="o", color="red", s=1
    )
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Comparison of TORCH and TTNN Tensors")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"tests/ttnn/integration_tests/t5/{key}.png")


block_index = 0
model = "decoder"


def test_pcc_attention(device):
    write_header()
    tensor_mamtul_1_torch = torch.load(
        "tests/ttnn/integration_tests/t5/t5_torch_outputs/decoder.block."
        + str(block_index)
        + ".layer.0.SelfAttention.matmul_1.pt"
    )[1, :, :, :].float()

    tensor_matmul_1_ttnn = (
        torch.load(
            "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/decoder.block."
            + str(block_index)
            + ".layer.0.SelfAttention.matmul_1.pt"
        )
        .squeeze(0)
        .squeeze(0)
        .float()[1, :, :, :]
    )

    g_max = torch.max(tensor_mamtul_1_torch)
    g_min = torch.min(tensor_mamtul_1_torch)
    c_max = torch.max(tensor_matmul_1_ttnn)
    c_min = torch.min(tensor_matmul_1_ttnn)
    key = "attention_matmul1"
    max_diff = torch.abs(tensor_mamtul_1_torch - tensor_matmul_1_ttnn)
    tolerance = (abs(g_max.item() - c_min.item()) + abs(c_max.item() - g_min.item())) / 20
    num_values_gt_tolerance = (max_diff > tolerance).sum().item()

    plot_figure(key, tensor_mamtul_1_torch, tensor_matmul_1_ttnn)
    make_histogram(key, tensor_mamtul_1_torch, tensor_matmul_1_ttnn)
    write_data_to_csv(key, tensor_mamtul_1_torch, tensor_matmul_1_ttnn)

    tensor_mamtul_2_torch = torch.load(
        "tests/ttnn/integration_tests/t5/t5_torch_outputs/decoder.block."
        + str(block_index)
        + ".layer.0.SelfAttention.matmul_2.pt"
    )[1, :, :, :]

    tensor_matmul_2_ttnn = torch.load(
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/decoder.block."
        + str(block_index)
        + ".layer.0.SelfAttention.matmul_2.pt"
    )[1, :, :, :]
    g_max = torch.max(tensor_mamtul_2_torch)
    g_min = torch.min(tensor_mamtul_2_torch)
    c_max = torch.max(tensor_matmul_2_ttnn)
    c_min = torch.min(tensor_matmul_2_ttnn)

    max_diff = torch.abs(tensor_mamtul_2_torch - tensor_matmul_2_ttnn)
    tolerance = (abs(g_max.item() - c_min.item()) + abs(c_max.item() - g_min.item())) / 20
    num_values_gt_tolerance = (max_diff > tolerance).sum().item()
    key = "attention_matmul2"
    plot_figure(key, tensor_mamtul_2_torch, tensor_matmul_2_ttnn)
    make_histogram(key, tensor_mamtul_2_torch, tensor_matmul_2_ttnn)
    write_data_to_csv(key, tensor_mamtul_2_torch, tensor_matmul_2_ttnn)

    tensor_add_torch = torch.load(
        "tests/ttnn/integration_tests/t5/t5_torch_outputs/decoder.block."
        + str(block_index)
        + ".layer.0.SelfAttention.addd.pt"
    )[1, :, :, :]
    tensor_add_ttnn = torch.load(
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/decoder.block."
        + str(block_index)
        + ".layer.0.SelfAttention.addd.pt"
    )[1, :, :, :]
    g_max = torch.max(tensor_add_torch)
    g_min = torch.min(tensor_add_torch)
    c_max = torch.max(tensor_add_ttnn)
    c_min = torch.min(tensor_add_ttnn)

    max_diff = torch.abs(tensor_add_torch - tensor_add_ttnn)
    tolerance = (abs(g_max.item() - c_min.item()) + abs(c_max.item() - g_min.item())) / 20
    num_values_gt_tolerance = (max_diff > tolerance).sum().item()

    key = "attention_add"
    plot_figure(key, tensor_add_torch, tensor_add_ttnn)
    make_histogram(key, tensor_add_torch, tensor_add_ttnn)
    write_data_to_csv(key, tensor_add_torch, tensor_add_ttnn)
