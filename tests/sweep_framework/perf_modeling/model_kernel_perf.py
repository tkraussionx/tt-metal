import csv
import re
import argparse
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pickle
import numpy as np
from utility_functions import process_txt_file, convert_to_numeric_dict


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


def make_a_decision_tree(results, module_name, tree_depth=6, output_tree_file=None):
    X = []
    y = []

    for entry in results:
        target_duration = entry.pop("DEVICE KERNEL DURATION [ns]")
        X.append(list(entry.values()))  # Features
        y.append(target_duration)  # Target
    X = np.array(X)
    y = np.array(y)

    # This needs to be done on a per-op basis, based on manual testing, this is for add
    bin_edges = [0, 5000, 10000, 50000, 100000]
    y_binned = np.digitize(y, bin_edges, right=False) - 1
    # for i in range(len(y)):
    #     print(y[i], y_binned[i])
    # y_binned, bin_edges = pd.cut(y, bins=3, labels=False, retbins=True)  # Create bins

    print(f"Bin edges: {bin_edges}")  # Optional: Show bin ranges
    # print("y=", y)
    # print("y_binned=", y_binned)
    X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=0.2, random_state=42)

    model = DecisionTreeClassifier(max_depth=tree_depth, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy score: {accuracy}")

    # Plot the decision tree
    if output_tree_file:
        plt.figure(figsize=(50, 50))  # Set figure size
        plot_tree(model, feature_names=list(results[0].keys()), filled=True, rounded=True)
        plt.title("Decision Tree for DEVICE KERNEL DURATION")
        plt.savefig(output_tree_file, format="png")

    with open(
        f"tests/sweep_framework/perf_modeling/models_and_edges/{module_name}_decision_tree_model.pkl", "wb"
    ) as model_file:
        pickle.dump(model, model_file)
    with open(
        f"tests/sweep_framework/perf_modeling/models_and_edges/{module_name}_bin_edges.pkl", "wb"
    ) as bin_edges_file:
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
    parser.add_argument("--output-tree-file", type=str, required=False, help="Output file for the decision tree")
    parser.add_argument("--module-name", type=str, required=False, help="Name of the module for the decision tree")
    parser.add_argument("--tree-depth", type=int, default=6, help="Set the depth of the decision tree")

    args = parser.parse_args()

    merge_config_files = args.merge_config_files
    csv_file = args.csv
    config_file = args.config
    output_file = args.output
    make_tree = args.make_tree
    output_tree_file = args.output_tree_file
    module_name = args.module_name
    tree_depth = args.tree_depth

    if merge_config_files:
        if not csv_file or not config_file:
            raise Exception("Please provide the config files.")

    if make_tree:
        if not module_name:
            raise Exception("Please provide the module name")

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

        make_a_decision_tree(numeric_results, module_name, tree_depth, output_tree_file)


if __name__ == "__main__":
    main()
