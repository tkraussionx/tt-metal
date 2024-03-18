# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import argparse


def analyze_perf_csv(csv_file, layers, show_all):
    df = pd.read_csv(csv_file)
    df = df[df["DEVICE FW DURATION [ns]"] != "-"]
    df["DEVICE FW DURATION [ns]"] = df["DEVICE FW DURATION [ns]"].astype(int)
    df[
        [
            "INPUT_0_W",
            "INPUT_0_Z",
            "INPUT_0_Y",
            "INPUT_0_X",
            "INPUT_1_W",
            "INPUT_1_Z",
            "INPUT_1_Y",
            "INPUT_1_X",
            "OUTPUT_0_W",
            "OUTPUT_0_Z",
            "OUTPUT_0_Y",
            "OUTPUT_0_X",
        ]
    ] = (
        df[
            [
                "INPUT_0_W",
                "INPUT_0_Z",
                "INPUT_0_Y",
                "INPUT_0_X",
                "INPUT_1_W",
                "INPUT_1_Z",
                "INPUT_1_Y",
                "INPUT_1_X",
                "OUTPUT_0_W",
                "OUTPUT_0_Z",
                "OUTPUT_0_Y",
                "OUTPUT_0_X",
            ]
        ]
        .replace("-", 1)
        .astype(int)
    )
    sorted_df = df.sort_values(by="DEVICE FW DURATION [ns]", ascending=False)
    sum_duration = df["DEVICE FW DURATION [ns]"].sum()

    matmul_rows = sorted_df[sorted_df["OP CODE"].str.contains("tt::operations::primary::Matmul")]
    matmul_rows.loc[:, "bytes"] = (
        matmul_rows["INPUT_1_W"] * matmul_rows["INPUT_1_Z"] * matmul_rows["INPUT_1_Y"] * matmul_rows["INPUT_1_X"]
    )
    matmul_rows.loc[:, "flops"] = 2 * matmul_rows["INPUT_0_Y"] * matmul_rows["INPUT_0_X"] * matmul_rows["OUTPUT_0_X"]
    matmul_rows["GB/s"] = matmul_rows["bytes"] / matmul_rows["DEVICE FW DURATION [ns]"]
    matmul_rows["TFLOP/s"] = matmul_rows["flops"] / matmul_rows["DEVICE FW DURATION [ns]"] / 1000
    matmul_rows["% DRAM (240)"] = 100 * matmul_rows["GB/s"] / 240  # Peak expected WH bandwidth
    matmul_rows["% FPU (82)"] = 100 * matmul_rows["TFLOP/s"] / 82  # Peak theoretical FP16 FPU performance
    matmul_rows["% TIME"] = 100 * matmul_rows["DEVICE FW DURATION [ns]"] / sum_duration
    matmul_rows["% TIME SUM"] = matmul_rows["% TIME"].cumsum()
    matmul_sum_duration = matmul_rows["DEVICE FW DURATION [ns]"].sum()

    # shorten some column names
    matmul_rows.rename(columns={"DEVICE FW DURATION [ns]": "DURATION [ns]"}, inplace=True)
    sorted_df.rename(columns={"DEVICE FW DURATION [ns]": "DURATION [ns]"}, inplace=True)

    selected_columns = [
        "OP CODE",
        "% TIME",
        "% TIME SUM",
        "% DRAM (240)",
        "% FPU (82)",
        "DURATION [ns]",
        "GB/s",
        "TFLOP/s",
        "CORE COUNT",
        "INPUT_0_Y",
        "INPUT_0_X",
        "INPUT_1_Y",
        "INPUT_1_X",
        "OUTPUT_0_Y",
        "OUTPUT_0_X",
    ]
    print(matmul_rows[selected_columns])

    if show_all:
        selected_columns = [
            "OP CODE",
            "% TIME",
            "% TIME SUM",
            "DURATION [ns]",
            "CORE COUNT",
            "INPUT_0_Y",
            "INPUT_0_X",
            "INPUT_1_Y",
            "INPUT_1_X",
            "OUTPUT_0_Y",
            "OUTPUT_0_X",
        ]
        sorted_df["% TIME"] = 100 * sorted_df["DURATION [ns]"] / sum_duration
        sorted_df["% TIME SUM"] = sorted_df["% TIME"].cumsum()
        print()
        print(sorted_df[selected_columns])

    if layers:
        tokens_per_sec_user = 1000000000 / sum_duration / layers
        tokens_per_sec = 32 * tokens_per_sec_user
        print(f"Layer ms: {sum_duration / 1000000:.1f} ({matmul_sum_duration / sum_duration:.1%} matmul)")
        print(f"Tokens/sec/user: {tokens_per_sec_user:.1f}")
        print(f"Tokens/sec: {tokens_per_sec:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze perf CSV file")
    parser.add_argument(
        "-a", "--all", action="store_true", help="List ops in the CSV file - by default only matmul ops are shown."
    )
    parser.add_argument("-l", "--layers", type=int, help="Number of layers to extrapolate perf results up to.")
    parser.add_argument(
        "csv_file", type=str, help="Path to the perf CSV file from tt-metal for a single decoder layer."
    )
    args = parser.parse_args()

    analyze_perf_csv(args.csv_file, layers=args.layers, show_all=args.all)


if __name__ == "__main__":
    main()
