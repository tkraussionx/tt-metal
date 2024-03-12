# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import argparse
import requests
import tempfile
import pathlib
import zipfile
import pandas as pd
from loguru import logger
from dataclasses import dataclass
from tabulate import tabulate
import os
import shutil
from datetime import datetime
import numpy as np

max_runs_to_capture = 20


@dataclass
class PerfResults:
    perf_type: str
    file_name: str
    branch: str
    commit_hash: str
    date: datetime
    df: pd.DataFrame


def get_list_of_e2e_device_runs_gs(branch):
    params = {"per_page": max_runs_to_capture, "branch": branch}
    url = "https://api.github.com/repos/tenstorrent-metal/tt-metal/actions/workflows/perf-models.yaml/runs"
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        runs = response.json()
    else:
        raise RuntimeError(f"Error fetching e2e workflow runs: {response.status_code}:{response.text}")

    return runs


def get_list_of_e2e_metal_run_benchmarks_gs(branch):
    params = {"per_page": max_runs_to_capture, "branch": branch}
    url = (
        "https://api.github.com/repos/tenstorrent-metal/tt-metal/actions/workflows/metal-run-microbenchmarks.yaml/runs"
    )
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        runs = response.json()
    else:
        raise RuntimeError(f"Error fetching e2e workflow runs: {response.status_code}:{response.text}")

    return runs


def get_list_of_perf_device_runs_gs(branch):
    params = {"per_page": max_runs_to_capture, "branch": branch}
    url = "https://api.github.com/repos/tenstorrent-metal/tt-metal/actions/workflows/perf-device-models.yaml/runs"
    headers = {"Accept": "application/vnd.github.v3+json"}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        runs = response.json()
    else:
        raise RuntimeError(f"Error fetching device workflow runs: {response.status_code}:{response.text}")

    return runs


def download_artifacts(token, artifacts_url, temp_dir_path, directory_index):
    response = requests.get(artifacts_url)
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    if response.status_code == 200:
        artifacts_data = response.json()
        if artifacts_data["artifacts"]:
            artifact = artifacts_data["artifacts"][0]
            artifact_download_url = artifact["archive_download_url"]
            artifact_response = requests.get(artifact_download_url, headers=headers)
            if artifact_response.status_code == 200:
                (temp_dir_path / str(directory_index)).mkdir(parents=True, exist_ok=True)
                artifact_zip = temp_dir_path / str(directory_index) / "artifact.zip"
                with open(artifact_zip, "wb") as file:
                    file.write(artifact_response.content)
                logger.info(f"{artifacts_url} downloaded successfully.")
                return True
            else:
                raise RuntimeError("Failed to download the artifact.")
        else:
            print(f"No artifacts found.  Is there a run in progress for {artifacts_url} ?")
    else:
        raise RuntimeError(f"Failed to fetch artifacts list. {response.status_code}:{response.text}")
    return False


def read_csv_from_zip(zip_file, file_name, date, perf_type):
    date = datetime.fromisoformat(date.rstrip("Z"))
    with zip_file.open(file_name) as fin:
        data = fin.read().splitlines(True)
        branch = data[0]
        commit_hash = data[1]

    with zip_file.open(file_name) as fin:
        df = pd.read_csv(fin, skiprows=[0, 1])
        df.columns = [col.strip() for col in df.columns]
        for col in df.columns:
            df[col] = df[col].apply(lambda x: str(x).replace("\t", "    ").replace("\n", " "))
            # df.replace(-np.inf, np.nan, inplace=True)

        return PerfResults(perf_type, file_name, branch, commit_hash, date, df)


def trim_column(texte, longueur):
    if len(texte) > longueur:
        return texte[-longueur + 3 :]
    return texte


def delete_directory_contents(dir_path):
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


def collect_perf_results(temp_dir, perf_type):
    temp_dir = pathlib.Path(temp_dir)
    subdirectories = sorted((item for item in temp_dir.iterdir() if item.is_dir()), key=lambda x: int(x.name))

    perf_results = []

    commit_hash = ""
    for subdir in subdirectories:
        recent_zip = subdir / "artifact.zip"
        commit_hash_file = subdir / "commit_hash.txt"
        with open(commit_hash_file, "r") as file:
            commit_hash = file.read()
        date_file = subdir / "date.txt"
        with open(date_file, "r") as file:
            date = file.read()

        with zipfile.ZipFile(recent_zip, "r") as zip1:
            zip1_files = set(zip1.namelist())
            for file_name in zip1_files:
                test_name = pathlib.Path(file_name).stem
                if file_name.endswith(".csv"):
                    perf_results.append(read_csv_from_zip(zip1, file_name, date, perf_type))

    return perf_results


def build_summary_rst_file(directory_for_rst_pages, summary_df):
    directory_for_rst_pages = pathlib.Path(directory_for_rst_pages)
    rst_table = tabulate(summary_df, headers="keys", tablefmt="rst")
    rst_page_name = directory_for_rst_pages / f"perf_summary.rst"
    with open(rst_page_name, "w") as f:
        f.writelines(f".. _ttnn.perf_summary:\n")
        f.writelines("\n")
        f.writelines(f"Lastest performance metrics\n")
        f.writelines("====================================================================\n")
        f.write(rst_table)


def download_from_pipeline(runs, temp_dir, token):
    """
    Download the results of the sweeps from the GitHub pipeline.

    :param token: Provide your GitHub token.
    """

    if len(runs["workflow_runs"]) < 3:
        # Note that if the run is in progress, there will not be any artifacts available yet on the most recent run.
        raise RuntimeError("We need at least three runs to compare the changes in the sweep tests")

    total_expected_runs = len(runs["workflow_runs"])
    if runs["workflow_runs"][0]["status"] == "completed":
        most_recent_run_index = 0
    else:  # a run is in progress so we just use the prior two for the first comparison
        most_recent_run_index = 1

    directory_index = 0
    temp_dir_path = pathlib.Path(temp_dir)
    for i in range(most_recent_run_index, total_expected_runs):
        most_recent_run = runs["workflow_runs"][i]
        most_recent_artifact_url = most_recent_run["artifacts_url"]
        commit_hash = most_recent_run["head_sha"]
        date = most_recent_run["created_at"]
        if download_artifacts(token, most_recent_artifact_url, temp_dir_path, directory_index):
            commit_hash_file = temp_dir_path / str(directory_index) / "commit_hash.txt"
            with open(commit_hash_file, "w") as file:
                file.write(commit_hash)
            date_file = temp_dir_path / str(directory_index) / "date.txt"
            with open(date_file, "w") as file:
                file.write(date)
            directory_index = directory_index + 1

    return directory_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token")
    parser.add_argument("--dir")
    token = parser.parse_args().token
    directory_for_rst_pages = parser.parse_args().dir

    branch = "main"
    device_runs = get_list_of_perf_device_runs_gs(branch)
    e2e_runs = get_list_of_e2e_device_runs_gs(branch)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_device_path = pathlib.Path(temp_dir) / "device"
        temp_e2e_path = pathlib.Path(temp_dir) / "e2e"

        download_from_pipeline(device_runs, temp_device_path, token)
        download_from_pipeline(e2e_runs, temp_e2e_path, token)

        e2e_perf_results = collect_perf_results(temp_e2e_path, "e2e")
        for item in e2e_perf_results:
            item.df["E2E Hash"] = item.commit_hash.replace("hash: ", "")
            item.df["E2E Date"] = item.date
        e2e_latest_perf = pd.concat([i.df for i in e2e_perf_results])
        e2e_latest_perf["Throughput GS (Batch*inf/sec)"] = pd.to_numeric(
            e2e_latest_perf["Throughput GS (Batch*inf/sec)"], errors="coerce"
        )
        max_e2e_times = e2e_latest_perf.groupby(["Model", "Batch"])["Throughput GS (Batch*inf/sec)"].transform("max")
        max_e2e_times = e2e_latest_perf[e2e_latest_perf["Throughput GS (Batch*inf/sec)"] == max_e2e_times]
        max_e2e_times = max_e2e_times.drop_duplicates(subset=["Model", "Batch", "Throughput GS (Batch*inf/sec)"])
        max_e2e_times = max_e2e_times[["Model", "Batch", "Throughput GS (Batch*inf/sec)", "E2E Hash", "E2E Date"]]
        max_e2e_times.rename(
            columns={"Throughput GS (Batch*inf/sec)": "Best Throughput GS (Batch*inf/sec)"}, inplace=True
        )
        max_e2e_times.reset_index(drop=True, inplace=True)
        e2e_latest_perf = e2e_latest_perf.drop_duplicates(subset=["Model", "Batch"], keep="last")
        e2e_latest_perf = e2e_latest_perf.merge(
            max_e2e_times, on=["Model", "Batch"], how="left", suffixes=("", " Best")
        )

        device_perf_results = collect_perf_results(temp_device_path, "device")
        for item in device_perf_results:
            item.df["Device Hash"] = item.commit_hash.replace("hash: ", "")
            item.df["Device Date"] = item.date
        device_latest_perf = pd.concat([i.df for i in device_perf_results])
        device_latest_perf["MAX DEVICE FW SAMPLES/S"] = pd.to_numeric(
            device_latest_perf["MAX DEVICE FW SAMPLES/S"], errors="coerce"
        )
        device_latest_perf["AVG DEVICE FW SAMPLES/S"] = pd.to_numeric(
            device_latest_perf["AVG DEVICE FW SAMPLES/S"], errors="coerce"
        )
        max_device_times = device_latest_perf.groupby(["Model", "Batch"])["MAX DEVICE FW SAMPLES/S"].transform("max")
        max_device_times = device_latest_perf[device_latest_perf["MAX DEVICE FW SAMPLES/S"] == max_device_times]
        max_device_times = max_device_times.drop_duplicates(subset=["Model", "Batch", "MAX DEVICE FW SAMPLES/S"])
        max_device_times = max_device_times[["Model", "Batch", "MAX DEVICE FW SAMPLES/S", "Device Hash", "Device Date"]]
        max_device_times.rename(columns={"MAX DEVICE FW SAMPLES/S": "Best DEVICE FW SAMPLES/S"}, inplace=True)
        max_device_times.reset_index(drop=True, inplace=True)
        device_latest_perf = device_latest_perf.drop_duplicates(subset=["Model", "Batch"], keep="last")
        device_latest_perf = device_latest_perf.merge(
            max_device_times, on=["Model", "Batch"], how="left", suffixes=("", " Best")
        )

        # These are all the fields currently available to put on the report.
        #
        # Model
        # Setting_x
        # Batch
        # First Run (sec)
        # Second Run (sec)
        # Compile Time (sec)
        # Expected Compile Time (sec)
        # Inference Time (sec)
        # Expected Inference Time (sec)
        # Throughput (Batch*inf/sec)
        # Inference Time CPU (sec)
        # Throughput CPU (Batch*inf/sec)
        # E2E Hash
        # E2E Date
        # Inference Time GS (sec)
        # Expected Inference Time GS (sec)
        # Throughput GS (Batch*inf/sec)
        # Best Throughput GS (Batch*inf/sec)
        # E2E Hash Best
        # E2E Date Best
        # Setting_y
        # AVG DEVICE FW SAMPLES/S
        # MIN DEVICE FW SAMPLES/S
        # MAX DEVICE FW SAMPLES/S
        # AVG DEVICE KERNEL SAMPLES/S
        # Lower Threshold AVG DEVICE KERNEL SAMPLES/S
        # Upper Threshold AVG DEVICE KERNEL SAMPLES/S
        # MIN DEVICE KERNEL SAMPLES/S
        # MAX DEVICE KERNEL SAMPLES/S
        # AVG DEVICE BRISC KERNEL SAMPLES/S
        # MIN DEVICE BRISC KERNEL SAMPLES/S
        # MAX DEVICE BRISC KERNEL SAMPLES/S
        # Device Hash
        # Device Date
        # Best DEVICE FW SAMPLES/S
        # Device Hash Best
        # Device Date Best

        merged_df = pd.merge(e2e_latest_perf, device_latest_perf, on=["Model", "Batch"], how="left")
        merged_df = merged_df[
            [
                "Model",
                "Batch",
                "Best Throughput GS (Batch*inf/sec)",
                "Best DEVICE FW SAMPLES/S",
                "E2E Hash",
                "Throughput CPU (Batch*inf/sec)",
                "AVG DEVICE FW SAMPLES/S",
            ]
        ]

        delete_directory_contents(directory_for_rst_pages)
        build_summary_rst_file(directory_for_rst_pages, merged_df)


if __name__ == "__main__":
    main()
