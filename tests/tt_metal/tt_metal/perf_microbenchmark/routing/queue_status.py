#!/usr/bin/env python3

# SPDX-FileCopyrightText: (c) 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time
import sys
import re
import subprocess
import argparse
import re

input_queue_status_start = 0xABCDEF00
output_queue_status_start = 0xABCDEF01


def mmio_read_xy(chip, x, y, noc_id, reg_addr):
    return chip.pci_read_xy(x, y, noc_id, reg_addr)


def print_base_queue_status(vals, index):
    print(f"   Queue ID: {vals[index]}")
    index += 1
    print(f"   Queue addr: 0x{vals[index]:x}")
    index += 1
    print(f"   Queue size: 0x{vals[index]:x}")
    index += 1
    print(f"   Remote x: {vals[index]}")
    index += 1
    print(f"   Remote y: {vals[index]}")
    index += 1
    print(f"   Remote queue id: {vals[index]}")
    index += 1
    print(f"   Local write pointer: 0x{vals[index]:x}")
    index += 1
    print(f"   Local read pointer sent: 0x{vals[index]:x}")
    index += 1
    print(f"   Local read pointer cleared: 0x{vals[index]:x}")
    index += 1
    print(f"   Remote update network type: {vals[index]}")
    index += 1
    return index


def print_input_queue_status(vals, index):
    index += 1
    print(f"     Curr packet valid: {vals[index]}")
    index += 1
    print(f"     Curr packet tag: 0x{vals[index]:x}")
    index += 1
    print(f"     Curr packet dest: 0x{vals[index]:x}")
    index += 1
    print(f"     Curr packet size words: 0x{vals[index]:x}")
    index += 1
    print(f"     Curr packet words sent: 0x{vals[index]:x}")
    index += 1
    return index


def print_output_queue_status(vals, index):
    index += 1
    print(f"     Max num words to forward: {vals[index]}")
    index += 2
    print(f"     Curr index: {vals[index]}")
    index += 1
    for i in range(2):
        index += 1
        print(f"     Index {i} total words in flight: {vals[index]}")
        index += 1
        for j in range(4):
            index += 1
            print(f"       Input queue {j} words in flight: {vals[index]}")
            index += 1
    return index


def main():
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", type=str)
    parser.add_argument("--x", type=int)
    parser.add_argument("--y", type=int)
    parser.add_argument("--len", type=int)

    args = parser.parse_args()

    if args.addr is not None:
        args.addr = int(args.addr, base=16)
    else:
        args.addr = 0
    if args.x is None:
        args.x = 1
    if args.y is None:
        args.y = 1
    if args.len is None:
        args.len = 1

    cmd = f'./tt_metal/third_party/umd/device/bin/silicon/wormhole/tt-script --interface pci:0 ./device/bin/silicon/wormhole/ttx_status.py   --args "x={args.x},y={args.y},addr=0x{args.addr:x},burst={args.len}"'
    output = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    output_lines = output.stdout.split("\n")
    vals = []
    curr_addr = args.addr
    for line in output_lines:
        if re.match(r"^\d+:", line):
            line_parts = line.split()
            val = int(line_parts[4], 16)
            vals.append(val)
            addr = int(line_parts[2], 16)
            if addr != curr_addr:
                print(f"Error: Expected address: {curr_addr:08x}, got: {addr:08x}")
                return -1
            curr_addr += 4

    # print (vals)
    # return 0

    index = 0
    num_words = 0
    while index < len(vals):
        val = vals[index]
        curr_addr = index * 4 + args.addr
        if index == 16:
            num_words = val << 32
        if index == 17:
            num_words |= val
            print(f"Total number of words: {num_words}")
        if val == input_queue_status_start:
            print(f"Input Queue status found at: {curr_addr:08x}")
            index += 1
            index = print_base_queue_status(vals, index)
            index = print_input_queue_status(vals, index)
        elif val == output_queue_status_start:
            print(f"Output Queue status found at: {curr_addr:08x}")
            index += 1
            index = print_base_queue_status(vals, index)
            index = print_output_queue_status(vals, index)
        else:
            index += 1

    return 0


main()
