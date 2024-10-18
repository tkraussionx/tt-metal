import os
import pandas as pd
import numpy as np


def average_csv_in_subfolders(root_folder, output_file):
    # Initialize a list to store DataFrames
    dataframes = []

    # Traverse the root folder and its subfolders
    for foldername, subfolders, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".csv") and filename.startswith("ops_perf_results"):
                filepath = os.path.join(foldername, filename)
                print(f"Processing file: {filepath}")
                # Read the CSV file and append it to the list
                df = pd.read_csv(filepath)
                dataframes.append(df)

    # Check if there are any DataFrames loaded
    if dataframes:
        # Ensure all files have the same shape
        num_rows = dataframes[0].shape[0]
        num_cols = dataframes[0].shape[1]
        for df in dataframes:
            if df.shape != (num_rows, num_cols):
                print("Error: Not all CSV files have the same structure and number of rows.")
                return

        # Stack the DataFrames and calculate the mean along the third dimension (row-wise and column-wise)
        stacked_df = np.stack([df.values for df in dataframes], axis=2)
        avg_values = np.mean(stacked_df, axis=2)

        # Create a DataFrame from the average values
        avg_df = pd.DataFrame(avg_values, columns=dataframes[0].columns)

        # Write the averaged DataFrame to a new CSV file
        avg_df.to_csv(output_file, index=False)
        print(f"Averaged CSV file saved to: {output_file}")
    else:
        print("No CSV files found in the specified folder.")


# Usage
root_folder = "generated/profiler/reports"  # Replace with the root folder path
output_file = "averaged_output_for_testing.csv"  # Replace with your desired output file name
average_csv_in_subfolders(root_folder, output_file)
