import argparse
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from utility_functions import process_txt_file, convert_to_numeric_dict


def check_tree(results, module_name, print_diff):
    X = []
    y = []

    for entry in results:
        target_duration = entry.pop("DEVICE KERNEL DURATION [ns]")
        X.append(list(entry.values()))  # Features
        y.append(target_duration)  # Target

    with open(
        f"tests/sweep_framework/perf_modeling/models_and_edges/{module_name}_decision_tree_model.pkl", "rb"
    ) as model_file:
        loaded_model = pickle.load(model_file)
        X_test = X
        with open(
            f"tests/sweep_framework/perf_modeling/models_and_edges/{module_name}_bin_edges.pkl", "rb"
        ) as bin_edges_file:
            bin_edges = pickle.load(bin_edges_file)
            print(f"Bin edges: {bin_edges}")  # Optional: Show bin ranges
            y_binned = np.digitize(y, bin_edges, right=False) - 1
            y_pred = loaded_model.predict(X_test)

            if print_diff:
                for i in range(len(y_binned)):
                    if y_binned[i] != y_pred[i]:
                        for j in range(len(X[i])):
                            print(list(results[0].keys())[j], ":", X[i][j])
                        print(y_binned[i], y_pred[i], y[i])

            accuracy = accuracy_score(y_binned, y_pred)
            print(f"accuracy score: {accuracy}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--file-path", type=str, required=True, help="Path to the .txt with merged configs and times.")
    parser.add_argument("--module-name", type=str, required=True, help="Name of the module.")
    parser.add_argument("--print-diff", action="store_true", help="Print bins that differ")

    args = parser.parse_args()

    file_path = args.file_path
    module_name = args.module_name
    print_diff = args.print_diff

    results = process_txt_file(file_path)

    numeric_results = []

    for result in results:
        numeric_result = convert_to_numeric_dict(result)
        numeric_results.append(numeric_result)

    check_tree(numeric_results, module_name, print_diff)


if __name__ == "__main__":
    main()
