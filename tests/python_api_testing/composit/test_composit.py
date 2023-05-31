# TODO: PURGE THIS
import sys
from pathlib import Path

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../../../tt_metal/third_party/composit/src")

import pytest

from contextlib import contextmanager
import math
import operator

import numpy as np
import torch
import transformers


import composit as cnp
import composit.nn
from composit.introspection import class_name
from composit.numpy.core import get_operands
from composit.multidigraph import topological_traversal, compose_all

import flashlight
from model_zoo.bert import (
    create_bert_config,
)

import tt_lib as ttl


@contextmanager
def devices():
    tt_device = ttl.device.CreateDevice(0)
    ttl.device.InitializeDevice(tt_device)
    host = ttl.device.GetHost()
    yield host, tt_device
    ttl.device.CloseDevice(tt_device)


def to_tt_tensor(input_tensor, shape):
    output_tensor = ttl.tensor.Tensor(
        input_tensor,
        ttl.tensor.DataType.BFLOAT16,
    )
    return output_tensor


def from_tt_tensor(input_tensor, shape, dtype=torch.float16):
    output_tensor = torch.tensor(input_tensor.data(), dtype=dtype)
    output_tensor = output_tensor[: math.prod(shape)]
    output_tensor = output_tensor.reshape(*shape)
    output_tensor = output_tensor.detach().numpy()
    return output_tensor


def pad_to_factor(value, factor):
    modulo = value % factor
    if modulo == 0:
        return 0
    padding = factor - modulo
    return padding


def pad_tensor(tensor, factor):
    if len(tensor.shape) == 0:
        tensor = tensor.reshape((1, 1, 1, 1))
    if len(tensor.shape) == 1:
        tensor = tensor.reshape((1, 1, 1, *tensor.shape))
    elif len(tensor.shape) == 2:
        tensor = tensor.reshape((1, 1, *tensor.shape))
    elif len(tensor.shape) == 3:
        tensor = tensor.reshape((1, *tensor.shape))

    *other_dims, height, width = tensor.shape
    height_padding = pad_to_factor(height, factor)
    width_padding = pad_to_factor(width, factor)

    other_paddings = tuple((0, 0) for _ in other_dims)

    if height_padding > 0 or width_padding > 0:
        return np.pad(tensor, other_paddings + ((0, height_padding), (0, width_padding)), mode="constant")
    else:
        return tensor


def dispatch(*, graph, node, input_arrays, host, tt_device):
    attributes = graph.nodes[node]
    instruction = attributes["instruction"]
    instruction_name = class_name(instruction)

    if isinstance(instruction, cnp.core.Constant):
        array = pad_tensor(instruction.array, 32)
        instruction_output = to_tt_tensor(array, array.shape).to(tt_device)
        return instruction_output

    elif instruction_name == "matmul":
        input_arrays = [input_array.to(host).to(ttl.tensor.Layout.TILE).to(tt_device) for input_array in input_arrays]
        if math.prod(input_arrays[1].shape()[:2]) > 1:
            instruction_output = ttl.tensor.bmm(*input_arrays)
        else:
            instruction_output = ttl.tensor.matmul(*input_arrays)
        instruction_output = instruction_output.to(host).to(ttl.tensor.Layout.ROW_MAJOR).to(tt_device)
        return instruction_output

    elif instruction_name in {"add", "subtract", "multiply", "divide"}:
        input_a_shape, input_b_shape = [
            graph.nodes[operand_node]["shapes"][output_index]
            for (operand_node, output_index) in get_operands(graph, node)
        ]

        if input_a_shape == input_b_shape:
            function = {
                "add": ttl.tensor.add,
                "subtract": ttl.tensor.sub,
                "multiply": ttl.tensor.mul,
                "divide": ttl.tensor.mul,  # TODO: decompose divide into reciprocal and multiply
            }[instruction_name]
            return function(*input_arrays)

        math_operation = {
            "add": ttl.tensor.BcastOpMath.ADD,
            "subtract": ttl.tensor.BcastOpMath.SUB,
            "multiply": ttl.tensor.BcastOpMath.MUL,
            "divide": ttl.tensor.BcastOpMath.MUL,  # TODO: decompose divide into reciprocal and multiply
        }[instruction_name]
        if len(input_a_shape) == 0:
            return ttl.tensor.bcast(input_arrays[1], input_arrays[0], math_operation, ttl.tensor.BcastOpDim.HW)
        elif len(input_b_shape) == 0:
            return ttl.tensor.bcast(input_arrays[0], input_arrays[1], math_operation, ttl.tensor.BcastOpDim.HW)

        if input_a_shape[-1] == input_b_shape[-1]:
            instruction_output = ttl.tensor.bcast(*input_arrays, math_operation, ttl.tensor.BcastOpDim.H)
        elif input_a_shape[-2] == input_b_shape[-2]:
            instruction_output = ttl.tensor.bcast(*input_arrays, math_operation, ttl.tensor.BcastOpDim.W)
        else:
            instruction_output = ttl.tensor.bcast(*input_arrays, math_operation, ttl.tensor.BcastOpDim.HW)
        return instruction_output

    elif instruction_name in {"mean", "sum", "max"}:
        (input_shape,) = [
            graph.nodes[operand_node]["shapes"][output_index]
            for (operand_node, output_index) in get_operands(graph, node)
        ]
        output_shape = attributes["shapes"][0]

        reduce_operation = {
            "mean": ttl.tensor.ReduceOpMath.SUM,
            "sum": ttl.tensor.ReduceOpMath.SUM,
            "max": ttl.tensor.ReduceOpMath.MAX,
        }[instruction_name]

        reduce_factor = 1.0
        if input_shape[1] == output_shape[1]:
            if instruction_name == "mean":
                reduce_factor = 1 / input_shape[0]
            instruction_output = ttl.tensor.reduce(
                input_arrays[0], reduce_operation, ttl.tensor.ReduceOpDim.H, reduce_factor
            )
        elif input_shape[0] == output_shape[0]:
            if instruction_name == "mean":
                reduce_factor = 1 / input_shape[1]
            instruction_output = ttl.tensor.reduce(
                input_arrays[0], reduce_operation, ttl.tensor.ReduceOpDim.W, reduce_factor
            )
        else:
            instruction_output = ttl.tensor.reduce(
                input_arrays[0], reduce_operation, ttl.tensor.ReduceOpDim.HW, reduce_factor
            )

        return instruction_output

    elif instruction_name == "transpose":
        instruction_output = ttl.tensor.transpose(input_arrays[0])
        return instruction_output

    elif instruction_name == "reshape":
        array = from_tt_tensor(input_arrays[0].to(host), graph.nodes[node]["shapes"][0])
        array = graph.nodes[node]["instruction"](array)
        array = pad_tensor(array, 32)
        array = to_tt_tensor(array, array.shape).to(tt_device)
        return array

    elif instruction_name in {"exp", "gelu", "relu", "sigmoid", "sqrt", "tanh"}:
        unary_function = getattr(ttl.tensor, instruction_name)
        instruction_output = unary_function(input_arrays[0])
        return instruction_output

    else:
        raise NotImplementedError(f"{instruction_name}")


def evaluate(*outputs, host, tt_device):
    graph = compose_all(*tuple(output.graph for output in outputs))

    cache = {}
    for node in topological_traversal(graph):
        input_arrays = [cache[operand] for operand in get_operands(graph, node)]

        for input_array in input_arrays:
            assert len(input_array.shape()) == 4, f"{node}"

        instruction_output = dispatch(graph=graph, node=node, input_arrays=input_arrays, host=host, tt_device=tt_device)

        if np.isscalar(instruction_output):
            raise RuntimeError(f"Scalars aren't supported on the output!")

        if isinstance(instruction_output, ttl.tensor.Tensor):
            cache[(node, 0)] = instruction_output
        elif isinstance(instruction_output, list):
            for output_index, instruction_output in enumerate(instruction_output):
                cache[(node, output_index)] = instruction_output
        else:
            raise RuntimeError("Unsupported type")

    for output in outputs:
        shape = graph.nodes[output.node]["shapes"][output.output_index]
        tensor = cache[(output.node, output.output_index)]
        cache[(output.node, output.output_index)] = from_tt_tensor(tensor.to(host), shape)

    result = [cache[(output.node, output.output_index)] for output in outputs]
    if len(result) == 1:
        return result[0]
    return result


def test_datacopy():
    np.random.seed(0)
    np_input = np.random.uniform(-0.1, 0.1, (32, 96)).astype(np.float16)

    def model(input_var):
        return input_var

    golden_output = model(np_input)

    input_var = cnp.asarray(np_input)
    output_var = model(input_var)

    with devices() as (host, tt_device):
        output = evaluate(output_var, host=host, tt_device=tt_device)

    assert np.allclose(output, golden_output, atol=1e-2)


def test_matmul():
    def model(input_var, weights_var):
        return input_var @ weights_var

    np.random.seed(0)
    np_input = np.random.uniform(-0.1, 0.1, (32, 96)).astype(np.float16)
    np_weight = np.random.uniform(-0.1, 0.1, (96, 64)).astype(np.float16)
    golden_output = model(np_input, np_weight)

    input_var = cnp.asarray(np_input)
    weights_var = cnp.asarray(np_weight)
    output_var = model(input_var, weights_var)

    with devices() as (host, tt_device):
        output = evaluate(output_var, host=host, tt_device=tt_device)

    assert np.allclose(output, golden_output, atol=1e-2)


@pytest.mark.parametrize("input_b_height", [32, 1])
@pytest.mark.parametrize("input_b_width", [96, 1])
@pytest.mark.parametrize("operation", [operator.add, operator.sub, operator.mul])
def test_binary_operation(input_b_height, input_b_width, operation):
    def model(input_a_var, input_b_var):
        return operation(input_a_var, input_b_var)

    np.random.seed(0)
    np_input_a = np.random.uniform(-0.1, 0.1, (32, 96)).astype(np.float16)
    np_input_b = np.random.uniform(-0.1, 0.1, (input_b_height, input_b_width)).astype(np.float16)
    golden_output = model(np_input_a, np_input_b)

    input_a_var = cnp.asarray(np_input_a)
    input_b_var = cnp.asarray(np_input_b)
    output_var = model(input_a_var, input_b_var)

    with devices() as (host, tt_device):
        output = evaluate(output_var, host=host, tt_device=tt_device)

    assert np.allclose(output, golden_output, atol=1e-2)


@pytest.mark.parametrize("unary_operation", [cnp.exp, cnp.nn.gelu, cnp.nn.relu, cnp.sqrt, cnp.nn.sigmoid, cnp.tanh])
def test_unary_operation(unary_operation):
    np.random.seed(0)
    np_input = np.random.uniform(0.0, 0.1, (32, 96)).astype(np.float16)

    input_var = cnp.asarray(np_input)
    output_var = unary_operation(input_var)
    golden_output = cnp.evaluate(output_var)

    with devices() as (host, tt_device):
        output = evaluate(output_var, host=host, tt_device=tt_device)

    assert np.allclose(output, golden_output, atol=1e-1)


@pytest.mark.parametrize("axes", [(0, 1), 0])
@pytest.mark.parametrize("operation", ["sum", "mean", "max"])
def test_reduce(axes, operation):
    if axes == (0, 1) and operation == "mean":
        pytest.skip()

    def model(input_var, *, np):
        function = getattr(np, operation)
        return function(input_var, axes, keepdims=True)

    np.random.seed(0)
    np_input = np.random.uniform(-0.1, 0.1, (32, 96)).astype(np.float16)
    golden_output = model(np_input, np=np)

    input_var = cnp.asarray(np_input)
    output_var = model(input_var, np=cnp)

    with devices() as (host, tt_device):
        output = evaluate(output_var, host=host, tt_device=tt_device)

    assert np.allclose(output, golden_output, atol=1e-2 if operation == "max" else 1e-1)


def test_linear():
    def model(input_var, weights_var, bias_var):
        return input_var @ weights_var + bias_var

    np.random.seed(0)
    np_input = np.random.uniform(-0.1, 0.1, (32, 96)).astype(np.float16)
    np_weight = np.random.uniform(-0.1, 0.1, (96, 64)).astype(np.float16)
    np_bias = np.random.uniform(-0.1, 0.1, (1, 64)).astype(np.float16)
    golden_output = model(np_input, np_weight, np_bias)

    input_var = cnp.asarray(np_input)
    weights_var = cnp.asarray(np_weight)
    bias_var = cnp.asarray(np_bias)
    output_var = model(input_var, weights_var, bias_var)

    with devices() as (host, tt_device):
        output = evaluate(output_var, host=host, tt_device=tt_device)

    assert np.allclose(output, golden_output, atol=1e-2)


def test_transpose():
    def model(input_var, *, np):
        return np.transpose(input_var, (1, 0))

    np.random.seed(0)
    np_input = np.random.uniform(-0.1, 0.1, (32, 96)).astype(np.float16)
    golden_output = model(np_input, np=np)

    input_var = cnp.asarray(np_input)
    output_var = model(input_var, np=cnp)

    with devices() as (host, tt_device):
        output = evaluate(output_var, host=host, tt_device=tt_device)

    assert np.allclose(output, golden_output, atol=1e-2)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_encoders", [1])
@pytest.mark.parametrize("sequence_size", [32])
@pytest.mark.parametrize("num_attention_heads", [4])
@pytest.mark.parametrize("head_size", [32])
@pytest.mark.parametrize("vocab_size", [10])
def test_bert(
    batch_size,
    num_encoders,
    sequence_size,
    num_attention_heads,
    head_size,
    vocab_size,
):
    config = create_bert_config(
        num_encoders=num_encoders,
        num_attention_heads=num_attention_heads,
        head_size=head_size,
        vocab_size=vocab_size,
    )

    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_size))
    attention_mask = torch.zeros(batch_size, sequence_size, dtype=torch.float32)
    token_type_ids = torch.zeros(batch_size, sequence_size, dtype=torch.int64)

    with flashlight.tracer.trace(run_torch=True):
        transformers_model = transformers.models.bert.modeling_bert.BertModel(config)
        flashlight_output = transformers_model(input_ids, attention_mask, token_type_ids=token_type_ids)[
            "last_hidden_state"
        ]

    for node in topological_traversal(flashlight_output.graph):
        attributes = flashlight_output.graph.nodes[node]
        operation = attributes["operation"]
        print(f"operation: {operation}")

    # with devices() as (host, tt_device):
    #     composit_output = evaluate(flashlight_output.lazy_tensor, host=host, tt_device=tt_device)

    # TODO: add intermediate comparison in evaluate
    # assert np.allclose(composit_output, flashlight_output.detach().numpy(), atol=1e-3)


input_0 = torch.zeros()
input_1 = ...

embedding_88888888888888 = torch.embedding(input_0, input_1)


def test_whisper():
    """
    Original example from Huggingface documentation:

    Example:
        ```python
        >>> import torch
        >>> from transformers import AutoFeatureExtractor, WhisperModel
        >>> from datasets import load_dataset

        >>> model = WhisperModel.from_pretrained("openai/whisper-base")
        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")
        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features
        >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id
        >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
        >>> list(last_hidden_state.shape)
        [1, 2, 512]
        ```
    """

    from datasets import load_dataset
    from transformers import WhisperModel, AutoFeatureExtractor

    model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
    model.eval()

    feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
    # original from HF example should be: seq_len = 3000, when max_source_positions=1500
    input_features = inputs.input_features

    decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id

    with torch.no_grad():
        with flashlight.tracer.trace(run_torch=True):
            last_hidden_state = model(input_features=input_features, decoder_input_ids=decoder_input_ids)
