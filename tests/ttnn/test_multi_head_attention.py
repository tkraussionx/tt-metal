import ttnn
import torch
import torch.nn.functional as F
import pytest
from tests.ttnn.utils_for_testing import assert_with_pcc
import tt_lib as ttl


def multi_head_attention(
    hidden_states,
    attention_mask,
    query_weight,
    query_bias,
    key_weight,
    key_bias,
    value_weight,
    value_bias,
    output_weight,
    output_bias,
    *,
    head_size,
):
    ignored, batch_size, sequence_size, hidden_size = hidden_states.shape()
    num_heads = hidden_size // head_size
    query = hidden_states @ query_weight
    query = query + query_bias
    print(query.shape())
    query = ttnn.reshape(query, (batch_size, sequence_size, num_heads, head_size))
    print(query.shape())
    query = ttnn.permute(query, (0, 2, 1, 3))
    key = hidden_states @ key_weight
    key = key + key_bias
    key = ttnn.reshape(key, (batch_size, sequence_size, num_heads, head_size))
    key = ttnn.permute(key, (0, 2, 3, 1))
    value = hidden_states @ value_weight
    value = value + value_bias
    value = ttnn.reshape(value, (batch_size, sequence_size, num_heads, head_size))
    value = ttnn.permute(value, (0, 2, 1, 3))
    print(f"query shape: {query.shape()}")
    print(f"key shape: {key.shape()}")
    print(f"value shape: {value.shape()}")
    attention_scores = query @ key
    print(f"attention_scores shape: {attention_scores.shape()}")
    attention_scores = attention_scores * (1 / (head_size**0.5))
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask
    attention_probs = ttnn.softmax(attention_scores, dim=-1)
    print(f"attention_probs shape: {attention_probs.shape()}")
    context_layer = attention_probs @ value
    print(f"context_layer shape before permute: {context_layer.shape()}")
    context_layer = ttnn.permute(context_layer, (0, 2, 1, 3))
    print(f"context_layer shape after permute: {context_layer.shape()}")
    context_layer = ttnn.reshape(context_layer, (batch_size, sequence_size, hidden_size))
    self_output = context_layer @ output_weight
    self_output = self_output + output_bias
    return self_output


def pytorch_multi_head_attention(
    hidden_states,
    attention_mask,
    query_weight,
    query_bias,
    key_weight,
    key_bias,
    value_weight,
    value_bias,
    output_weight,
    output_bias,
    *,
    head_size,
):
    ignored, batch_size, sequence_size, hidden_size = hidden_states.shape
    num_heads = hidden_size // head_size
    query = hidden_states @ query_weight
    query = query + query_bias
    query = query.view(batch_size, sequence_size, num_heads, head_size)
    query = query.permute(0, 2, 1, 3)
    key = hidden_states @ key_weight
    key = key + key_bias
    key = key.view(batch_size, sequence_size, num_heads, head_size)
    key = key.permute(0, 2, 3, 1)
    value = hidden_states @ value_weight
    value = value + value_bias
    value = value.view(batch_size, sequence_size, num_heads, head_size)
    value = value.permute(0, 2, 1, 3)
    print(f"query shape: {query.shape}")
    print(f"key shape: {key.shape}")
    print(f"value shape: {value.shape}")
    attention_scores = query @ key
    print(f"attention_scores shape: {attention_scores.shape}")
    attention_scores = attention_scores / (head_size**0.5)
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask
    attention_probs = F.softmax(attention_scores, dim=-1)
    print(f"attention_probs shape: {attention_probs.shape}")
    context_layer = attention_probs @ value
    print(f"context_layer shape before permute: {context_layer.shape}")
    context_layer = context_layer.permute(0, 2, 1, 3)
    print(f"context_layer shape after permute: {context_layer.shape}")
    context_layer = context_layer.reshape(batch_size, sequence_size, hidden_size)
    self_output = context_layer @ output_weight
    self_output = self_output + output_bias
    return self_output


# Note that our reshape requires the width and height to both be multiples of 32
# so the number of heads must be 32
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("sequence_size", [2 * 32])
@pytest.mark.parametrize("num_heads", [4])
@pytest.mark.parametrize("head_size", [32])
def test_multi_head_attention(device, batch_size, sequence_size, num_heads, head_size):
    hidden_size = num_heads * head_size
    hidden_states = ttnn.random(shape=(1, batch_size, sequence_size, hidden_size))
    torch_hidden_states = ttnn.to_torch(hidden_states)

    attention_mask = ttnn.zeros(shape=(1, 1, 1, sequence_size), dtype=ttl.tensor.DataType.BFLOAT16)
    # attention_mask[:, :, ::2, :] = -1e9
    torch_attention_mask = ttnn.to_torch(attention_mask)

    query_weight = ttnn.random(shape=(1, 1, hidden_size, hidden_size))
    torch_query_weight = ttnn.to_torch(query_weight)
    query_bias = ttnn.random(shape=(1, 1, 1, hidden_size))
    torch_query_bias = ttnn.to_torch(query_bias)
    key_weight = ttnn.random(shape=(1, 1, hidden_size, hidden_size))
    torch_key_weight = ttnn.to_torch(key_weight)
    key_bias = ttnn.random(shape=(1, 1, 1, hidden_size))
    torch_key_bias = ttnn.to_torch(key_bias)
    value_weight = ttnn.random(shape=(1, 1, hidden_size, hidden_size))
    torch_value_weight = ttnn.to_torch(value_weight)
    value_bias = ttnn.random(shape=(1, 1, 1, hidden_size))
    torch_value_bias = ttnn.to_torch(value_bias)
    output_weight = ttnn.random(shape=(1, 1, hidden_size, hidden_size))
    torch_output_weight = ttnn.to_torch(output_weight)
    output_bias = ttnn.random(shape=(1, 1, 1, hidden_size))
    torch_output_bias = ttnn.to_torch(output_bias)

    torch_output = pytorch_multi_head_attention(
        torch_hidden_states,
        torch_attention_mask,
        torch_query_weight,
        torch_query_bias,
        torch_key_weight,
        torch_key_bias,
        torch_value_weight,
        torch_value_bias,
        torch_output_weight,
        torch_output_bias,
        head_size=head_size,
    )

    assert torch_output.shape == (
        1,
        batch_size,
        sequence_size,
        hidden_size,
    ), f"Expected output shape to be {batch_size, sequence_size, hidden_size}, got {torch_output.shape}"

    tt_output = multi_head_attention(
        hidden_states,
        attention_mask,
        query_weight,
        query_bias,
        key_weight,
        key_bias,
        value_weight,
        value_bias,
        output_weight,
        output_bias,
        head_size=head_size,
    )

    assert tt_output.shape() == [
        1,
        batch_size,
        sequence_size,
        hidden_size,
    ], f"Expected output shape to be {batch_size, sequence_size, hidden_size}, got {tt_output.shape()}"

    assert_with_pcc(torch_output, ttnn.to_torch(tt_output), 0.9)
