import pickle
import ttnn
from utility_functions import convert_to_numeric_dict

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
