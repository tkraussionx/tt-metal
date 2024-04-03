from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import comp_allclose, comp_allclose_and_pcc
import torch
import ttnn
import os
import csv

# Add the path where you want the observations csv to be dumped
csv_file_path = "models/experimental/functional_t5/dumps/t5_observations.csv"


def write_header():
    # Check if the file exists
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    with open(csv_file_path, "w", newline="") as file:
        writer = csv.writer(file)
        header = ["\t\tAll close comparison of output after each OP\n"]
        writer.writerow(header)
        column_names = ["Sub-Module", "OP", "Max_Diff"]
        writer.writerow(column_names)  # Write only column names


def append_to_csv(data):
    with open(csv_file_path, "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(data)


write_header()

block_index = 0
model = "decoder"


def test_pcc_layer_norm(device):
    tensor_mean_torch = torch.load(
        "tests/ttnn/integration_tests/t5/t5_torch_outputs/decoder.block."
        + str(block_index)
        + ".layer.0.layer_norm.mean.pt"
    )
    tensor_mean_ttnn = torch.load(
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/decoder.block."
        + str(block_index)
        + ".layer.0.layer_norm.mean.pt"
    )
    print(comp_allclose(tensor_mean_torch, tensor_mean_ttnn))
    diff = torch.abs(tensor_mean_torch - tensor_mean_ttnn)
    max_diff = torch.max(diff)
    data = ["LayerNorm", "mean", max_diff.item()]
    append_to_csv(data)

    tensor_mul_torch = torch.load(
        "tests/ttnn/integration_tests/t5/t5_torch_outputs/decoder.block."
        + str(block_index)
        + ".layer.0.layer_norm.mul.pt"
    )
    tensor_mul_ttnn = torch.load(
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/decoder.block."
        + str(block_index)
        + ".layer.0.layer_norm.mul.pt"
    )
    print(comp_allclose(tensor_mul_torch, tensor_mul_ttnn))
    diff = torch.abs(tensor_mul_torch - tensor_mul_ttnn)
    max_diff = torch.max(diff)
    data = ["LayerNorm", "mul", max_diff.item()]
    append_to_csv(data)

    tensor_pow_torch = torch.load(
        "tests/ttnn/integration_tests/t5/t5_torch_outputs/decoder.block."
        + str(block_index)
        + ".layer.0.layer_norm.pow.pt"
    )
    tensor_pow_ttnn = torch.load(
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/decoder.block."
        + str(block_index)
        + ".layer.0.layer_norm.pow.pt"
    )
    print(comp_allclose(tensor_pow_torch, tensor_pow_ttnn))
    diff = torch.abs(tensor_pow_torch - tensor_pow_ttnn)
    max_diff = torch.max(diff)
    data = ["LayerNorm", "pow", max_diff.item()]
    append_to_csv(data)

    tensor_rsqrt_torch = torch.load(
        "tests/ttnn/integration_tests/t5/t5_torch_outputs/decoder.block."
        + str(block_index)
        + ".layer.0.layer_norm.sqrt.pt"
    )
    tensor_rsqrt_ttnn = torch.load(
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/decoder.block."
        + str(block_index)
        + ".layer.0.layer_norm.sqrt.pt"
    )
    print(comp_allclose(tensor_rsqrt_torch, tensor_rsqrt_ttnn))
    diff = torch.abs(tensor_rsqrt_torch - tensor_rsqrt_ttnn)
    max_diff = torch.max(diff)
    data = ["Layer_Norm", "rsqrt", max_diff.item()]
    append_to_csv(data)


def test_pcc_layer_ff(device):
    tensor_add_torch = torch.load(
        "tests/ttnn/integration_tests/t5/t5_torch_outputs/decoder.block."
        + str(block_index)
        + ".layer.2._layer_ff_add.pt"
    )
    tensor_add_ttnn = torch.load(
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/decoder.block."
        + str(block_index)
        + ".layer.2._layer_ff_add.pt"
    )
    print(comp_allclose(tensor_add_torch, tensor_add_ttnn))
    diff = torch.abs(tensor_add_torch - tensor_add_ttnn)
    max_diff = torch.max(diff)
    data = ["Layer_FF", "add", max_diff.item()]
    append_to_csv(data)


def test_pcc_layer_self_attention(device):
    tensor_add_torch = torch.load(
        "tests/ttnn/integration_tests/t5/t5_torch_outputs/decoder.block."
        + str(block_index)
        + ".layer.0.self_attention_add.pt"
    )
    tensor_add_ttnn = torch.load(
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/decoder.block."
        + str(block_index)
        + ".layer.0.self_attention_add.pt"
    )
    print(comp_allclose(tensor_add_torch[0], tensor_add_ttnn))
    diff = torch.abs(tensor_add_torch[0] - tensor_add_ttnn)
    max_diff = torch.max(diff)
    data = ["Layer_Self_Attention", "add", max_diff.item()]
    append_to_csv(data)


def test_pcc_layer_cross_attention(device):
    tensor_add_torch = torch.load(
        "tests/ttnn/integration_tests/t5/t5_torch_outputs/decoder.block."
        + str(block_index)
        + ".layer.1.cross_attention_add.pt"
    )
    tensor_add_ttnn = torch.load(
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/decoder.block."
        + str(block_index)
        + ".layer.1.cross_attention_add.pt"
    )
    print(comp_allclose(tensor_add_torch[0], tensor_add_ttnn))
    diff = torch.abs(tensor_add_torch[0] - tensor_add_ttnn)
    max_diff = torch.max(diff)
    data = ["Layer_Cross_Attention", "add", max_diff.item()]
    append_to_csv(data)


def test_pcc_dense_act_dense(device):
    tensor_linear_wi_torch = torch.load(
        "tests/ttnn/integration_tests/t5/t5_torch_outputs/decoder.block." + str(block_index) + ".layer.2.wi.pt"
    )
    tensor_linear_wi_ttnn = torch.load(
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/decoder.block."
        + str(block_index)
        + ".layer.2.DenseReluDense.wi.pt"
    )
    print(comp_allclose(tensor_linear_wi_torch, tensor_linear_wi_ttnn))
    diff = torch.abs(tensor_linear_wi_torch - tensor_linear_wi_ttnn)
    max_diff = torch.max(diff)
    data = ["Dense_Act_Dense", "linear_wi", max_diff.item()]
    append_to_csv(data)

    tensor_linear_wo_torch = torch.load(
        "tests/ttnn/integration_tests/t5/t5_torch_outputs/decoder.block." + str(block_index) + ".layer.2.wo.pt"
    )
    tensor_linear_wo_ttnn = torch.load(
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/decoder.block."
        + str(block_index)
        + ".layer.2.DenseReluDense.wo.pt"
    )
    print(comp_allclose(tensor_linear_wo_torch, tensor_linear_wo_ttnn))
    diff = torch.abs(tensor_linear_wo_torch - tensor_linear_wo_ttnn)
    max_diff = torch.max(diff)
    data = ["Dense_Act_Dense", "linear_wo", max_diff.item()]
    append_to_csv(data)

    tensor_relu_torch = torch.load(
        "tests/ttnn/integration_tests/t5/t5_torch_outputs/decoder.block." + str(block_index) + ".layer.2.activation.pt"
    )
    tensor_relu_ttnn = torch.load(
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/decoder.block."
        + str(block_index)
        + ".layer.2.DenseReluDense.activation.pt"
    )
    print(comp_allclose(tensor_relu_torch, tensor_relu_ttnn))
    diff = torch.abs(tensor_relu_torch - tensor_relu_ttnn)
    max_diff = torch.max(diff)
    data = ["Dense_Act_Dense", "relu", max_diff.item()]
    append_to_csv(data)


def test_pcc_attention(device):
    tensor_mamtul_1_torch = torch.load(
        "tests/ttnn/integration_tests/t5/t5_torch_outputs/decoder.block."
        + str(block_index)
        + ".layer.0.SelfAttention.matmul_1.pt"
    )
    tensor_matmul_1_ttnn = torch.load(
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/decoder.block."
        + str(block_index)
        + ".layer.0.SelfAttention.matmul_1.pt"
    )
    print(comp_allclose(tensor_mamtul_1_torch, tensor_matmul_1_ttnn))
    diff = torch.abs(tensor_mamtul_1_torch - tensor_matmul_1_ttnn)
    max_diff = torch.max(diff)
    data = ["Attention", "matmul_1", max_diff.item()]
    append_to_csv(data)

    tensor_mamtul_2_torch = torch.load(
        "tests/ttnn/integration_tests/t5/t5_torch_outputs/decoder.block."
        + str(block_index)
        + ".layer.0.SelfAttention.matmul_2.pt"
    )
    tensor_matmul_2_ttnn = torch.load(
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/decoder.block."
        + str(block_index)
        + ".layer.0.SelfAttention.matmul_2.pt"
    )
    print(comp_allclose(tensor_mamtul_2_torch, tensor_matmul_2_ttnn))
    diff = torch.abs(tensor_mamtul_2_torch - tensor_matmul_2_ttnn)
    max_diff = torch.max(diff)
    data = ["Attention", "matmul_2", max_diff.item()]
    append_to_csv(data)

    tensor_softmax_torch = torch.load(
        "tests/ttnn/integration_tests/t5/t5_torch_outputs/decoder.block."
        + str(block_index)
        + ".layer.0.SelfAttention.softmax.pt"
    )
    tensor_softmax_ttnn = torch.load(
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/decoder.block."
        + str(block_index)
        + ".layer.0.SelfAttention.softmax.pt"
    )
    print(comp_allclose(tensor_softmax_torch, tensor_softmax_ttnn))
    diff = torch.abs(tensor_softmax_torch - tensor_softmax_ttnn)
    max_diff = torch.max(diff)
    data = ["Attention", "softmax", max_diff.item()]
    append_to_csv(data)

    tensor_add_torch = torch.load(
        "tests/ttnn/integration_tests/t5/t5_torch_outputs/decoder.block."
        + str(block_index)
        + ".layer.0.SelfAttention.addd.pt"
    )
    tensor_add_ttnn = torch.load(
        "tests/ttnn/integration_tests/t5/t5_ttnn_outputs/decoder.block."
        + str(block_index)
        + ".layer.0.SelfAttention.addd.pt"
    )
    print(comp_allclose(tensor_add_torch, tensor_add_ttnn))
    diff = torch.abs(tensor_add_torch - tensor_add_ttnn)
    max_diff = torch.max(diff)
    data = ["Attention", "add", max_diff.item()]
    append_to_csv(data)
