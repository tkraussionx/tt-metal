import csv
import re
import argparse
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np


# Function to parse the second file as a dictionary with string values
def parse_configurations_as_strings(config_file):
    configurations = []
    with open(config_file, "r") as file:
        for line in file:
            line = line.strip()
            # Match the dictionary part after the assignment (if present)
            match = re.search(r"=\s*(\{.*\})", line)
            if match:
                raw_pairs = match.group(1)
                config_dict = {}
                # Split on commas but avoid splitting inside parentheses or brackets
                pairs = re.split(r",\s*(?![^()]*\))", raw_pairs)
                for pair in pairs:
                    key, value = map(str.strip, pair.split(":", 1))
                    config_dict[key] = value
                configurations.append(config_dict)
    return configurations


# Function to match CSV rows with configurations
def match_configs_with_csv(csv_file, config_file):
    configurations = parse_configurations_as_strings(config_file)
    matches = []

    with open(csv_file, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if i < len(configurations):  # Ensure we have enough configurations
                device_kernel_duration = int(float(row["DEVICE KERNEL DURATION [ns]"]))
                config = configurations[i]
                matches.append({"csv_row": row, "config": config, "device_kernel_duration": device_kernel_duration})

    return matches


# Print or save the matched configurations and CSV rows
def print_matches(matches, output_file_path):
    with open(output_file_path, "w") as output_file:
        for match in matches:
            output_file.write(f"CSV Row: {match['csv_row']}")
            output_file.write(f"Configuration: {match['config']}")
            output_file.write(f", Device Kernel Duration: {match['device_kernel_duration']}")
            output_file.write("\n")


def convert_to_numeric_dict(input_dict):
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


def make_a_decision_tree(results, print_tree=False, output_tree_file=None):
    X = []
    y = []

    for entry in results:
        target_duration = entry.pop("DEVICE KERNEL DURATION [ns]")
        X.append(list(entry.values()))  # Features
        y.append(target_duration)  # Target

    X = np.array(X)
    y = np.array(y)

    y_binned, bin_edges = pd.cut(y, bins=3, labels=False, retbins=True)  # Create bins
    bin_edges[-1] = 20000
    print(f"Bin edges: {bin_edges}")  # Optional: Show bin ranges

    X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(max_depth=5, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy score: {accuracy}")

    # Plot the decision tree
    if print_tree:
        plt.figure(figsize=(50, 50))  # Set figure size
        plot_tree(model, feature_names=list(results[0].keys()), filled=True, rounded=True)
        plt.title("Decision Tree for DEVICE KERNEL DURATION")
        plt.savefig(output_tree_file, format="png")

    with open("add_decision_tree_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    with open("bin_edges.pkl", "wb") as bin_edges_file:
        pickle.dump(bin_edges, bin_edges_file)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--merge-config-files",
        action="store_true",
        help="Does the script first merge config generated files or just uses the existing output file",
    )
    parser.add_argument("--csv", type=str, required=False, help="Path of the .csv file with kernel times")
    parser.add_argument("--config", type=str, required=False, help="Path of the .txt file with configurations")
    parser.add_argument(
        "--output", type=str, required=False, help="Path of the output file of times and configurations"
    )
    parser.add_argument(
        "--make-tree", action="store_true", help="Does the script make the decision tree from the given configs."
    )
    parser.add_argument("--print-tree", action="store_true", help="Print the decision tree in the output file")
    parser.add_argument("--output-tree-file", type=str, required=False, help="Output file for the decision tree")

    args = parser.parse_args()

    merge_config_files = args.merge_config_files
    csv_file = args.csv
    config_file = args.config
    output_file = args.output
    make_tree = args.make_tree
    print_tree = args.print_tree
    output_tree_file = args.output_tree_file

    if merge_config_files:
        if not csv_file or not config_file:
            raise Exception("Please provide the config files.")

    if make_tree:
        if print_tree:
            if not output_tree_file:
                raise Exception("Please provide the file to output the decision tree diagram to.")

    if merge_config_files:
        # Match configurations with CSV rows
        matches = match_configs_with_csv(csv_file, config_file)

        # Print the matches
        print_matches(matches, output_file)

    if make_tree:
        # Process the output file
        results = process_txt_file(output_file)
        numeric_results = []
        for result in results:
            numeric_result = convert_to_numeric_dict(result)
            numeric_results.append(numeric_result)

        make_a_decision_tree(numeric_results, print_tree, output_tree_file)


if __name__ == "__main__":
    main()
