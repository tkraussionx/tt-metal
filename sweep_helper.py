import csv
import re

# File paths
csv_file_path = "generated/profiler/reports/2024_10_11_14_37_12/ops_perf_results_2024_10_11_14_37_12.csv"
config_file_path = "helper_add.txt"

# Regular expression to extract relevant fields from the dictionary string
pattern = re.compile(
    r"'height': (\d+), 'width': (\d+), 'broadcast': ([\w']+), 'input_a_dtype': <DataType\.(\w+): \d+>, 'input_b_dtype': <DataType\.(\w+): \d+>"
)


# Parse the second file to extract configurations
def parse_configurations(config_file):
    configurations = []
    with open(config_file, "r") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                height, width, broadcast, input_a_dtype, input_b_dtype = match.groups()
                configurations.append(
                    {
                        "height": int(height),
                        "width": int(width),
                        "broadcast": broadcast.strip("'"),
                        "input_a_dtype": input_a_dtype,
                        "input_b_dtype": input_b_dtype,
                    }
                )
    return configurations


# Parse the CSV file and match with the configurations
def match_configs_with_csv(csv_file, config_file):
    configurations = parse_configurations(config_file)
    matches = []

    with open(csv_file, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if i < len(configurations):  # Ensure that there are matching configurations
                device_kernel_duration = int(row["DEVICE KERNEL DURATION [ns]"])
                config = configurations[i]
                matches.append({"csv_row": row, "config": config, "device_kernel_duration": device_kernel_duration})

    return matches


# Print or save the matched configurations and CSV rows
def print_matches(matches):
    for match in matches:
        print(f"CSV Row: {match['csv_row']}")
        print(f"Configuration: {match['config']}")
        print(f"Device Kernel Duration: {match['device_kernel_duration']}")
        print("-" * 80)


# Match configurations with CSV rows
matches = match_configs_with_csv(csv_file_path, config_file_path)

# Print the matches
print_matches(matches)
