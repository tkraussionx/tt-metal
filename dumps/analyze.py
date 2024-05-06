import re

priority = "/home/jnie/tt-metal/dumps/test_result_with_priority.txt"
no_priority = "/home/jnie/tt-metal/dumps/test_result_no_priority.txt"

pattern = r"(\d+\.\d+|\d+) seconds"

test_mode_dram = "BFLOAT16-DRAM"
test_mode_l1 = "BFLOAT16-L1"
priority_runtimes = {test_mode_l1: [], test_mode_dram: []}

no_priority_runtimes = {test_mode_l1: [], test_mode_dram: []}


with open(priority, "r") as file:
    lines = file.readlines()
    for line in lines:
        runtime = re.search(pattern, line).group(1)
        if test_mode_dram in line:
            priority_runtimes[test_mode_dram].append(float(runtime))
        elif test_mode_l1 in line:
            priority_runtimes[test_mode_l1].append(float(runtime))
        else:
            assert False

with open(no_priority, "r") as file:
    lines = file.readlines()
    for line in lines:
        runtime = re.search(pattern, line).group(1)
        if test_mode_dram in line:
            no_priority_runtimes[test_mode_dram].append(float(runtime))
        elif test_mode_l1 in line:
            no_priority_runtimes[test_mode_l1].append(float(runtime))
        else:
            assert False


print(
    f"""
Average runtime with priority {test_mode_l1}: {sum(priority_runtimes[test_mode_l1]) / len(priority_runtimes[test_mode_l1])}
Average runtime with priority {test_mode_dram}: {sum(priority_runtimes[test_mode_dram]) / len(priority_runtimes[test_mode_dram])}
"""
)

print(
    f"""
Average runtime without priority {test_mode_l1}: {sum(no_priority_runtimes[test_mode_l1]) / len(no_priority_runtimes[test_mode_l1])}
Average runtime without priority {test_mode_dram}: {sum(no_priority_runtimes[test_mode_dram]) / len(no_priority_runtimes[test_mode_dram])}
"""
)
