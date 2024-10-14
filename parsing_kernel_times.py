import re
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
import pickle
import pandas as pd


def convert_to_numeric_dict(input_dict):
    # Mapping for dtypes and layouts to integers
    dtype_mapping = {
        "<DataType.BFLOAT16: 0>": 0,
        # Add other data types as needed
    }

    layout_mapping = {
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
        else:
            numeric_dict[key] = int(value) if value.isdigit() else value

    return numeric_dict


# Function to clean and extract the configuration string into a dictionary of strings
def parse_configuration_string(config_str):
    # Clean the configuration string
    cleaned_str = config_str.replace("Configuration: ", "").strip().replace("'", '"')
    # Remove unnecessary outer braces and quotes
    cleaned_str = cleaned_str[1:-1]  # Remove the outer {}

    # Fix the double quotes issue
    cleaned_str = cleaned_str.replace('""', '"')  # Change "" to "

    pattern = r'"(.*?)":\s*("[^"]*"|\{.*?\}|\(.*?\)|[^\s,]+)'

    config_dict = {}

    # Find all key-value pairs in the configuration string
    for match in re.finditer(pattern, cleaned_str):
        key = match.group(1).strip()  # Extract key
        value = match.group(2).strip()  # Extract value
        value = value.strip("<>")  # Removing <...> from complex types

        # Remove surrounding whitespace and additional quotes
        value = value.strip('"')

        # Add key-value pair to dictionary, all values as strings
        config_dict[key] = value

    return config_dict


def modify_dictionary(input_dict):
    # Create a new dictionary to store the modified keys and values
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


file_path = "config_and_times_test.txt"  # Replace with the actual path to your .txt file
results = process_txt_file(file_path)


def make_a_decision_tree(results):
    X = []
    y = []

    for entry in results:
        target_duration = entry.pop("DEVICE KERNEL DURATION [ns]")
        X.append(list(entry.values()))  # Features
        y.append(target_duration)  # Target

    X = np.array(X)
    y = np.array(y)

    y_binned, bin_edges = pd.cut(y, bins=3, labels=False, retbins=True)  # Create bins
    print(f"Bin edges: {bin_edges}")  # Optional: Show bin ranges

    X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(max_depth=6, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy score: {accuracy}")

    # Plot the decision tree
    plt.figure(figsize=(50, 50))  # Set figure size
    plot_tree(model, feature_names=list(results[0].keys()), filled=True, rounded=True)
    plt.title("Decision Tree for DEVICE KERNEL DURATION")
    plt.savefig("decision_tree_plot.png", format="png")

    with open("add_decision_tree_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    with open("bin_edges.pkl", "wb") as bin_edges_file:
        pickle.dump(bin_edges, bin_edges_file)


# Print the results to verify
numeric_results = []
for result in results:
    # print(result)
    numeric_result = convert_to_numeric_dict(result)
    numeric_results.append(numeric_result)
    # print("pp=", numeric_result)
    # print("-" * 80)

make_a_decision_tree(numeric_results)
