import pandas as pd
import argparse


def analyze_perf_csv(csv_file):
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
    tokens_per_sec_user = 1000000000 / sum_duration / 32
    tokens_per_sec = 32 * tokens_per_sec_user

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
    matmul_sum_duration = matmul_rows["DEVICE FW DURATION [ns]"].sum()

    selected_columns = [
        "OP CODE",
        "% TIME",
        "% DRAM (240)",
        "% FPU (82)",
        "DEVICE FW DURATION [ns]",
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

    print(f"Layer ms: {sum_duration / 1000000:.1f} ({matmul_sum_duration / sum_duration:.1%} matmul)")
    print(f"Tokens/sec/user: {tokens_per_sec_user:.1f}")
    print(f"Tokens/sec: {tokens_per_sec:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze perf CSV file")
    parser.add_argument(
        "csv_file", type=str, help="Path to the perf CSV file from tt-metal for a single decoder layer."
    )
    args = parser.parse_args()

    analyze_perf_csv(args.csv_file)


if __name__ == "__main__":
    main()
