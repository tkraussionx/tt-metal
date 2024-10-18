import re
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np


def convert_to_numeric_dict(input_dict):
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

    memory_config_mapping = {  # Currently, all block sharded memory configs are under num 3, but that needs to be changed to accomodate differend sharding patterns.
        "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::L1,shard_spec=std::nullopt)": 0,
        "MemoryConfig(memory_layout=TensorMemoryLayout::INTERLEAVED,buffer_type=BufferType::DRAM,shard_spec=std::nullopt)": 1,
    }

    # New dictionary for the numeric values
    numeric_dict = {}

    for key, value in input_dict.items():
        if key == "DEVICE KERNEL DURATION [ns]":
            numeric_dict[key] = int(value)
        elif value in dtype_mapping:
            numeric_dict[key] = dtype_mapping[value]
        elif value in layout_mapping:
            numeric_dict[key] = layout_mapping[value]
        elif value in broadcast_mapping:
            numeric_dict[key] = broadcast_mapping[value]
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


# Function to clean and extract the configuration string into a dictionary of strings
def parse_configuration_string(config_str):
    cleaned_str = config_str.replace("Configuration: ", "").strip().replace("'", '"')

    cleaned_str = cleaned_str[1:-1]

    cleaned_str = cleaned_str.replace('""', '"')

    pattern = r'"(.*?)":\s*("[^"]*"|\{.*?\}|\(.*?\)|[^\s,]+)'

    config_dict = {}

    # Find all key-value pairs in the configuration string
    for match in re.finditer(pattern, cleaned_str):
        key = match.group(1).strip()
        value = match.group(2).strip()
        value = value.strip("<>")  # Removing <...> from complex types

        value = value.strip('"')

        config_dict[key] = value

    return config_dict


def modify_dictionary(input_dict):
    modified_dict = {}

    for key, value in input_dict.items():
        if "batch_sizes" in key or "height" in key or "width" in key or "DEVICE KERNEL DURATION [ns]" in key:
            new_key = key.replace("'", "")
            new_key = new_key.replace('"', "")
            new_key = new_key.replace("{", "")
            new_value = "".join([char for char in value if char.isdigit()])
            modified_dict[new_key] = new_value
        else:
            new_key = key.replace("'", "")
            new_key = new_key.replace('"', "")
            new_value = value.replace("'", "")
            new_value = new_value.replace('"', "")
            if new_value[-1] == "}":
                new_value = new_value[:-1]
            modified_dict[new_key] = new_value

    return modified_dict


# Function to process the text file and create a single dictionary per line
def process_txt_file(file_path):
    results = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()

            # Match and extract CSV Row, Configuration, and Device Kernel Duration
            match = re.search(r"CSV Row: (.*)Configuration: (.*)Device Kernel Duration: (\d+)", line)

            if match:
                # Extract CSV Row, Configuration, and Device Kernel Duration
                csv_row_str = match.group(1)
                config_str = match.group(2)
                device_kernel_duration = match.group(3)

                # Parse the configuration string into a dictionary
                config_dict = parse_configuration_string(config_str)

                # Add 'DEVICE KERNEL DURATION [ns]' to the dictionary
                config_dict["DEVICE KERNEL DURATION [ns]"] = device_kernel_duration

                config_dict = modify_dictionary(config_dict)

                # Append the config_dict to results
                results.append(config_dict)

    return results
