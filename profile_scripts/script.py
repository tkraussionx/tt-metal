import sys
import argparse
import copy
import math

def calculate_standard_deviation(numbers):
    n = len(numbers)
    mean = sum(numbers) / n
    squared_differences = [(x - mean) ** 2 for x in numbers]
    mean_of_squared_diff = sum(squared_differences) / n
    standard_deviation = math.sqrt(mean_of_squared_diff)
    return standard_deviation

def profile_issue_barrier(file_name):
    f = open(file_name, "r")
    lines = f.readlines()
    issue = []
    barrier = []
    buffer = 0
    transaction = 0
    for line in lines:
        if "Buffer" in line:
            if len(issue) > 0:
                issue_avg = sum(issue)/len(issue)
                barrier_avg = sum(barrier)/len(barrier)
                print("Buffer: {} Transaction: {} issue: {:.2f} barrier: {:.2f}".format(buffer, transaction, issue_avg, barrier_avg))
                issue = []
                barrier = []
            buffer = int(line.split()[1])
            transaction = int(line.split()[-1])
        elif "issue" in line:
            lst = line.split()
            issue.append(int(lst[1]))
            barrier.append(int(lst[-1]))
    issue_avg = sum(issue)/len(issue)
    barrier_avg = sum(barrier)/len(barrier)
    print("Buffer: {} Transaction: {} issue: {:.2f} barrier: {:.2f}".format(buffer, transaction, issue_avg, barrier_avg))

def profile_fine_grain(file_name):
    f = open(file_name, "r")
    lines = f.readlines()
    dic = {5:[], 6:[]}
    for i in range(11, 18):
        dic[i] = []
    buffer = 0
    transaction = 0
    for line in lines:
        if "Buffer" in line:
            if len(dic[11]) > 0:
                print("Buffer: {} Transaction: {}".format(buffer, transaction), end=" ")
                avg = sum(dic[5])/len(dic[5])
                print("{}: {:.2f}".format(5, avg), end=" ")
                for i in range(11, 18):
                    avg = sum(dic[i])/len(dic[i])
                    print("{}: {:.2f}".format(i, avg), end=" ")
                    dic[i] = []
                avg = sum(dic[6])/len(dic[6])
                print("{}: {:.2f}".format(6, avg), end=" ")
                print()
            buffer = int(line.split()[1])
            transaction = int(line.split()[-1])
        elif "11:" in line:
            lst = line.split()
            dic[5].append(int(lst[1]))
            dic[6].append(int(lst[-1]))
            for i in range(11, 18):
                dic[i].append(int(lst[(i-9)*2+1]))
    print("Buffer: {} Transaction: {}".format(buffer, transaction), end=" ")
    avg = sum(dic[5])/len(dic[5])
    print("{}: {:.2f}".format(5, avg), end=" ")
    for i in range(11, 18):
        avg = sum(dic[i])/len(dic[i])
        print("{}: {:.2f}".format(i, avg), end=" ")
    avg = sum(dic[6])/len(dic[6])
    print("{}: {:.2f}".format(6, avg), end=" ")
    print()



def print_dram_rw(dic, read_write_bar):
    if read_write_bar:
        print("Read Speed")
    else:
        print("Write Speed")
    for transaction_power in range(6, 14):
        print(2**transaction_power, end=" ")
    print()
    for buffer_power in range(13, 20):
        print(2**buffer_power, end=" ")
        for transaction_power in range(6, 14):
            buffer = 2**buffer_power
            transaction = 2**transaction_power
            for tup in dic.keys():
                if buffer == tup[0] and transaction == tup[1]:
                    if read_write_bar:
                        print("{:.2f}".format(dic[tup][0]), end=" ")
                    else:
                        print("{:.2f}".format(dic[tup][1]), end=" ")
        print()

def profile_riscv_rw_dram(file_name):
    dic = {}
    f = open(file_name, "r")
    lines = f.readlines()
    for line in lines:
        lst = line.split()
        if "Test arguments" in line:
            transaction = int(lst[-1])
            buffer = int(lst[-7][:-1])
        elif "Read speed GB/s" in line:
            read = float(lst[-1])
        elif "Write speed GB/s" in line:
            write = float(lst[-1])
        elif "Test " in line:
            dic[(buffer, transaction)] = (read, write)
    print_dram_rw(dic, 1)
    print_dram_rw(dic, 0)

def print_tensix_bandwidth(dic):
    for transaction_power in range(6, 18):
        print(2**transaction_power, end=" ")
    print()
    for buffer_power in range(6, 18):
        print(2**buffer_power, end=" ")
        for transaction_power in range(6, buffer_power+1):
            buffer = 2**buffer_power
            transaction = 2**transaction_power
            for tup in dic.keys():
                if buffer == tup[0] and transaction == tup[1]:
                        print("{:.2f}".format(dic[tup]), end=" ")
        print()

def profile_riscv_tensix(file_name, read_write_bar):
    if read_write_bar:
        marker = "Read"
    else:
        marker = "Write"
    print(marker)
    dic = {}
    f = open(file_name, "r")
    lines = f.readlines()
    for line in lines:
        lst = line.split()
        if "Test arguments" in line:
            transaction = int(lst[-7][:-1])
            buffer = int(lst[-13][:-1])
        elif marker + " speed GB/s" in line:
            time = float(lst[-1])
            dic[(buffer, transaction)] = time
    print_tensix_bandwidth(dic)

def print_tensix_issue_barrier(file_name):
    dic_issue = {}
    dic_barrier = {}
    dic_noc_util = {}
    f = open(file_name, "r")
    lines = f.readlines()
    noc_util_flag = False
    for line in lines:
        lst = line.split()
        if "Buffer" in line:
            if len(lst) > 8:
                noc_util_flag = True
            transaction = int(lst[3])
            buffer = int(lst[1])
            issue = float(lst[5])
            barrier = float(lst[7])
            dic_issue[(buffer, transaction)] = issue
            dic_barrier[(buffer, transaction)] = barrier
            if noc_util_flag:
                noc_util = float(lst[9]) * 100
                dic_noc_util[(buffer, transaction)] = noc_util
        elif "write" in line:
            print("read")
            print("issue")
            print_tensix_bandwidth(dic_issue)
            print("barrier")
            print_tensix_bandwidth(dic_barrier)
            if noc_util_flag:
                print("noc_util")
                print_tensix_bandwidth(dic_noc_util)
            dic = {}
    print("write")
    print("issue")
    print_tensix_bandwidth(dic_issue)
    print("barrier")
    print_tensix_bandwidth(dic_barrier)
    if noc_util_flag:
        print("noc_util")
        print_tensix_bandwidth(dic_noc_util)

def profile_tensix_constant_flit(file_name):
    dic = {}
    f = open(file_name, "r")
    lines = f.readlines()
    for line in lines:
        lst = line.split()
        if "Buffer" in line:
            buffer = int(lst[1])
            transaction = int(lst[3])
            issue = float(lst[5])
            barrier = float(lst[7])
            if transaction not in dic.keys():
                dic[transaction] = []
            dic[transaction].append(barrier)
        elif "write" in line:
            print("Read")
            for transaction in dic.keys():
                print("Transaction:", transaction, calculate_standard_deviation(dic[transaction]))
            dic = {}
    print("Write")
    for transaction in dic.keys():
        print("Transaction:", transaction, calculate_standard_deviation(dic[transaction]))

def profile_noc_utilization(file_name, read_or_write):
    print(read_or_write)
    read_write_flag = False
    noc_util_flag = False
    buffer_flag = False
    f = open(file_name, "r")
    lines = f.readlines()
    for line in lines:
        lst = line.split()

        if "non_NIU_programming" in line:
            non_NIU_programming = lst[-1]

        if "read" in line:
            if read_or_write == "read":
                read_write_flag = True
            else:
                read_write_flag = False
        elif "write" in line:
            if read_or_write == "write":
                read_write_flag = True
            else:
                read_write_flag = False

        if "noc_util" in line:
            noc_util_flag = True
        elif "issue" in line or "barrier" in line:
            noc_util_flag = False

        if lst[0] == "131072":
            buffer_flag = True
        elif lst[0] != "131072":
            buffer_flag = False

        if read_write_flag and noc_util_flag and buffer_flag:
            print(non_NIU_programming, line[:-1])


def get_args():
    parser = argparse.ArgumentParser('Profile raw results.')
    parser.add_argument("--file-name", help="file to profile")
    parser.add_argument("--profile-target", choices=["Tensix2Tensix", "DRAM2Tensix", "Tensix2Tensix_Issue_Barrier", "Tensix2Tensix_Fine_Grain", "Print_Tensix2Tensix_Issue_Barrier", "Profile_Tensix2Tensix_Constant_Flit", "Profile_NOC_Utilization"], help="profile target choice")
    parser.add_argument("--read-or-write", choices=["read", "write"], help="read or write choice")
    args = parser.parse_args()
    return args

args = get_args()
file_name = args.file_name
if args.profile_target == "Tensix2Tensix":
    if args.read_or_write == "read":
        profile_riscv_tensix(file_name, 1)
    elif args.read_or_write == "write":
        profile_riscv_tensix(file_name, 0)
elif args.profile_target == "DRAM2Tensix":
    profile_riscv_rw_dram(file_name)
elif args.profile_target == "Tensix2Tensix_Issue_Barrier":
    profile_issue_barrier(file_name)
elif args.profile_target == "Tensix2Tensix_Fine_Grain":
    profile_fine_grain(file_name)
elif args.profile_target == "Print_Tensix2Tensix_Issue_Barrier":
    print_tensix_issue_barrier(file_name)
elif args.profile_target == "Profile_Tensix2Tensix_Constant_Flit":
    profile_tensix_constant_flit(file_name)
elif args.profile_target == "Profile_NOC_Utilization":
    profile_noc_utilization(file_name, args.read_or_write)
