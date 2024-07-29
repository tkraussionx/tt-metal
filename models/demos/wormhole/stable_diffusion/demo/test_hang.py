import ttnn


def test_hang(device, use_program_cache):
    hidden_states = ttnn.load_tensor("hidden_states.bin", device=device)
    norm1_input_mask = ttnn.load_tensor("norm1_input_mask.bin", device=device)
    norm1_weight = ttnn.load_tensor("norm1_weight.bin", device=device)
    norm1_bias = ttnn.load_tensor("norm1_bias.bin", device=device)
    for i in range(10000):
        print(i)
        output = ttnn.group_norm(
            hidden_states,
            num_groups=32,
            input_mask=norm1_input_mask,
            weight=norm1_weight,
            bias=norm1_bias,
            epsilon=1e-05,
            memory_config=ttnn.get_memory_config(hidden_states),
            core_grid=ttnn.CoreGrid(x=8, y=8),
            dtype=ttnn.bfloat8_b,
        )
    ttnn.synchronize_device(device)
