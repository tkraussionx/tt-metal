import pickle
import ttnn


def convert_to_numeric_dict(input_dict):
    # Mapping for dtypes and layouts to integers
    dtype_mapping = {
        "<DataType.BFLOAT16: 0>": 0,
        # Add other data types as needed
    }

    layout_mapping = {
        "<Layout.ROW_MAJOR: 0>": 0,
        "<Layout.TILE: 1>": 1,
        # Add other layouts as needed
    }

    broadcast_mapping = {"None": 0, "h": 1, "w": 2, "hw": 3}

    memory_config_mapping = {
        "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt)": 0,
        "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)": 1,
    }

    # New dictionary for the numeric values
    numeric_dict = {}

    for key, value in input_dict.items():
        # Convert DEVICE KERNEL DURATION [ns] directly to int
        if key == "DEVICE KERNEL DURATION [ns]":
            numeric_dict[key] = int(value)
        # Convert data types using the mapping
        elif value in dtype_mapping:
            numeric_dict[key] = dtype_mapping[value]
        # Convert layouts using the mapping
        elif value in layout_mapping:
            numeric_dict[key] = layout_mapping[value]
        elif value in broadcast_mapping:
            numeric_dict[key] = broadcast_mapping[value]
        # Convert other numerical string values to int
        elif value in memory_config_mapping:
            numeric_dict[key] = memory_config_mapping[value]
        elif "input_shape" in key:
            shape_values = list(map(int, value.strip("()").split(", ")))
            numeric_dict["height"] = shape_values[-2]
            numeric_dict["width"] = shape_values[-1]
        elif "memory_layout=TensorMemoryLayout::BLOCK_SHARDED" in value:
            numeric_dict[key] = 3
        # This is a hack needed because of bad splitting, needs to be fixed.
        elif value[0] == ":":
            continue
        else:
            numeric_dict[key] = int(value) if value.isdigit() else value

    return numeric_dict


string_mapping = {
    ttnn.bfloat16: "<DataType.BFLOAT16: 0>",
    ttnn.TILE_LAYOUT: "<Layout.TILE: 1>",
    ttnn.ROW_MAJOR_LAYOUT: "<Layout.ROW_MAJOR: 0>",
    ttnn.DRAM_MEMORY_CONFIG: "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt)",
    ttnn.L1_MEMORY_CONFIG: "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)",
}


def parse_dictionary(config_dict):
    return_dict = {}
    for key, value in config_dict.items():
        if value in string_mapping:
            return_dict[key] = string_mapping[value]
        else:
            return_dict[key] = str(value)
    return return_dict


def check_tree(config_dict, module_name):
    X = []
    results = [config_dict]
    for entry in results:
        X.append(list(entry.values()))  # Features

    with open(
        f"tests/sweep_framework/perf_modeling/models_and_edges/{module_name}_decision_tree_model.pkl", "rb"
    ) as model_file:
        loaded_model = pickle.load(model_file)
        X_test = X
        print(X_test)
        with open(
            f"tests/sweep_framework/perf_modeling/models_and_edges/{module_name}_bin_edges.pkl", "rb"
        ) as bin_edges_file:
            bin_edges = pickle.load(bin_edges_file)
            print(f"Bin edges: {bin_edges}")  # Optional: Show bin ranges
            y_pred = loaded_model.predict(X_test)

            return y_pred


def kernel_time_value(config_dict, module_name):
    config_dict = parse_dictionary(config_dict)
    config_dict = convert_to_numeric_dict(config_dict)
    return check_tree(config_dict, module_name)


if __name__ == "__main__":
    config_dict = {
        "input_shape": (1, 1, 64, 64),
        "input_a_dtype": ttnn.bfloat16,
        "input_a_layout": ttnn.TILE_LAYOUT,
        "input_a_memory_config": ttnn.DRAM_MEMORY_CONFIG,
        "output_memory_config": ttnn.DRAM_MEMORY_CONFIG,
    }

    module_name = "softmax"

    print(kernel_time_value(config_dict, module_name))
