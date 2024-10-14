import csv
import re

# File paths
csv_file = "generated/profiler/reports/2024_10_14_14_53_16/ops_perf_results_2024_10_14_14_53_16.csv"
config_file = "helper_add.txt"


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
                device_kernel_duration = int(row["DEVICE KERNEL DURATION [ns]"])
                config = configurations[i]
                matches.append({"csv_row": row, "config": config, "device_kernel_duration": device_kernel_duration})

    return matches


# Print or save the matched configurations and CSV rows
def print_matches(matches):
    output_file_path = "config_and_times_test.txt"
    with open(output_file_path, "w") as output_file:
        for match in matches:
            output_file.write(f"CSV Row: {match['csv_row']}")
            output_file.write(f"Configuration: {match['config']}")
            output_file.write(f", Device Kernel Duration: {match['device_kernel_duration']}")
            output_file.write("\n")


# Match configurations with CSV rows
matches = match_configs_with_csv(csv_file, config_file)

# Print the matches
print_matches(matches)
