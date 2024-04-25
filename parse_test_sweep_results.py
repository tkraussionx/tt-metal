import sys
import os
import json
from os.path import isfile
from os import listdir

# import mysql.connector
import pandas as pd
import json
from datetime import datetime
import csv
import argparse


def to_int(input_str, default_value):
    try:
        x = int(input_str)
    except ValueError:
        x = default_value

    return x


def parse_profiler_results(csv_location):
    with open(csv_location) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        profile_results = []
        profile_result = None
        i = 0
        header = ""
        counter = 0

        for row in reader:
            if i == 0:
                i += 1
                host_duration_index = row.index("HOST DURATION [ns]") if "HOST DURATION [ns]" in row else -1
                device_duration_index = row.index("DEVICE FW DURATION [ns]") if "DEVICE FW DURATION [ns]" in row else -1
                compile_duration_index = (
                    row.index("CompileProgram Average [ns]") if "CompileProgram Average [ns]" in row else -1
                )
                header = ", ".join(row)
                continue

            if "test_sweep_separator" in row[0]:
                counter += 1
                if profile_result is not None:
                    profile_results.append(profile_result)

                profile_result = {
                    "host_duration_ns": 0,
                    "device_duration_ns": 0,
                    "compile_duration_ns": 0,
                    "details": [],
                }
            else:
                if profile_result is None:
                    return None, header

                profile_result["host_duration_ns"] += (
                    to_int(row[host_duration_index], 0) if host_duration_index != -1 else 0
                )
                profile_result["device_duration_ns"] += (
                    to_int(row[device_duration_index], 0) if device_duration_index != -1 else 0
                )
                profile_result["compile_duration_ns"] += (
                    to_int(row[compile_duration_index], 0) if compile_duration_index != -1 else 0
                )
                profile_result["details"].append(row)

    if counter > len(profile_results):
        profile_results.append(profile_result)

    print("Parsed profiler logs")
    print(f"Expected number: {counter}")
    print(f"Actual number: {len(profile_results)}")

    return profile_results, header


# UPDATE DATABASE ========================================================
def insert_op_test_sweeps(mycursor, test_name, exec_time_loc, args):
    # default parameters
    test_name = test_name
    pcie_slot = args.pcie_slot
    date = datetime.now()
    machine_name = args.machine_name
    git_commit_hash = args.git_commit_hash
    arch_name = args.arch_name

    try:
        with open(exec_time_loc) as f:
            for line in f:
                execution_time = line.split(" ")[0]
                execution_status = line.split(" ")[1]
                break
    except:
        execution_time = 0
        execution_status = "Not measured"

    # insert data into op_test_sweep table
    sql_insert_into_tests = "INSERT INTO op_test_sweeps (test_name, date, execution_time, execution_status, git_commit_hash, machine_name, arch_name, pcie_slot, error, status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    values = (
        test_name,
        date,
        execution_time,
        execution_status,
        git_commit_hash,
        machine_name,
        arch_name,
        pcie_slot,
        "",
        "",
    )
    mycursor.execute(sql_insert_into_tests, values)
    print(mycursor.rowcount, "record inserted.")

    # get latest row_id for the test
    latest_test_id = mycursor.lastrowid
    print(latest_test_id, "latest id record inserted.")

    return latest_test_id


def insert_error_output(mycursor, msg, latest_test_id):
    sql_update_table = """UPDATE op_test_sweeps SET error=%s WHERE test_sweep_id=%s"""
    values = (msg, latest_test_id)
    mycursor.execute(sql_update_table, values)
    print("Updated test sweep with error msg")


def update_test_sweep_status(mycursor, status_value, sweep_id):
    sql_update_table = """UPDATE op_test_sweeps SET status=%s WHERE test_sweep_id=%s"""
    values = (status_value, sweep_id)
    mycursor.execute(sql_update_table, values)
    print("Updated test sweep with status value")


def insert_op_test_units(mycursor, filename, latest_test_id, csv_row_id, profiler_results, header):
    # read csv file
    reader = csv.DictReader(open(filename))

    # if no records in csv
    if len(profiler_results) > 1:
        status_value = "Pass"
    else:
        status_value = ""

    if profiler_results is not None:
        for row in reader:
            if csv_row_id >= len(profiler_results):
                print(f"Incorrect ROW ID: {csv_row_id}")
                break

            sql_insert_into_subtests = "INSERT INTO op_test_units (input_shapes, args, data_seed, env_vars, status, test_output, pass_fail, op_test_sweep_id, host_duration, device_duration, compile_duration, profile_details, header_columns) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            values = (
                row["input_shapes"],
                row["args"],
                row["data_seed"],
                row["env_vars"],
                row["status"],
                row["test_output"],
                row["pass/fail"],
                latest_test_id,
                profiler_results[csv_row_id]["host_duration_ns"],
                profiler_results[csv_row_id]["device_duration_ns"],
                profiler_results[csv_row_id]["compile_duration_ns"],
                json.dumps(profiler_results[csv_row_id]["details"]),
                header,
            )
            mycursor.execute(sql_insert_into_subtests, values)

            if row["pass/fail"] == "fail":
                status_value = "Fail"

            csv_row_id += 1
    else:
        for row in reader:
            sql_insert_into_subtests = "INSERT INTO op_test_units (input_shapes, args, data_seed, env_vars, status, test_output, pass_fail, op_test_sweep_id, host_duration, device_duration, compile_duration, profile_details, header_columns) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            values = (
                row["input_shapes"],
                row["args"],
                row["data_seed"],
                row["env_vars"],
                row["status"],
                row["test_output"],
                row["pass/fail"],
                latest_test_id,
                0,
                0,
                0,
                "",
                "",
            )
            mycursor.execute(sql_insert_into_subtests, values)

            if row["pass/fail"] == "fail":
                status_value = "Fail"

    print("Finished inserting records into mysql")
    update_test_sweep_status(mycursor, status_value, latest_test_id)

    return csv_row_id


if __name__ == "__main__":
    # get parameters
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-folder",
        help="Test sweeps results folder",
        # required=True,
    )
    parser.add_argument(
        "-s",
        "--pcie-slot",
        default=0,
        type=int,
        help="Virtual PCIE slot of GS device to run on",
    )
    parser.add_argument(
        "-m",
        "--machine-name",
        help="Hostname for the tests execution",
        # required=True,
    )
    parser.add_argument(
        "-gch",
        "--git-commit-hash",
        help="Git commit hash value",
        # required=True,
    )
    parser.add_argument(
        "-a",
        "--arch-name",
        help="Architecture name",
        # required=True,
    )

    args = parser.parse_args()

    # connect to database
    # mydb = mysql.connector.connect(
    #     host="localhost", user="ara", password="password", database="ara"
    # )
    # mycursor = mydb.cursor()

    input_folder = args.input_folder
    test_subdirectories = [os.path.join(input_folder, x) for x in os.listdir(input_folder)]

    for test_subdirectory in test_subdirectories:
        print(f"Running test: {test_subdirectory}")
        test_name = test_subdirectory.split("/")[-1]
        profiler_dir_path = test_subdirectory + "/profile"
        exec_time_loc = test_subdirectory + f"/total_time.txt"
        print(f"Execution time pth: {exec_time_loc}")

        profiler_csv_loc = ""
        for dir_path, dir_names, file_names in os.walk(profiler_dir_path):
            for file in file_names:
                if file.endswith(".csv") and file.startswith("ops_perf"):
                    profiler_csv_loc = os.path.join(dir_path, file)

        print(f"Profiler result pth: {profiler_csv_loc}")

        # write test sweep general info
        # try:
        #     latest_test_id = insert_op_test_sweeps(mycursor, test_name, exec_time_loc, args)
        # except Exception as e:
        #     msg = f"{e}"
        #     print(f"{msg}")
        # else:

        try:
            # write test sweep details
            files = [
                os.path.join(test_subdirectory, f)
                for f in listdir(test_subdirectory)
                if isfile(os.path.join(test_subdirectory, f))
            ]
        except Exception as e:
            msg = f"{e}"
            print(f"{msg}")
            insert_error_output(mycursor, msg, latest_test_id)
        else:
            if profiler_csv_loc != "":
                try:
                    profiler_results, header = parse_profiler_results(profiler_csv_loc)
                except Exception as e:
                    msg = f"{e}"
                    print(f"{msg}")
                    insert_error_output(mycursor, msg, latest_test_id)
                # else:
                #     try:
                #         csv_row_id = 0
                #         for filename in files:
                #             if filename.endswith("_sweep.csv") or filename.endswith("_sweep_tile.csv") or filename.endswith("_sweep_rm.csv"):
                #                 csv_row_id = insert_op_test_units(mycursor, filename, latest_test_id, csv_row_id, profiler_results, header)
                #     except Exception as e:
                #         msg = f"{e}"
                #         print(f"{msg}")
                #         insert_error_output(mycursor, msg, latest_test_id)
            else:
                try:
                    csv_row_id = 0
                    for filename in files:
                        if (
                            filename.endswith("_sweep.csv")
                            or filename.endswith("_sweep_tile.csv")
                            or filename.endswith("_sweep_rm.csv")
                        ):
                            print("insert_op_test_units")
                            # csv_row_id = insert_op_test_units(mycursor, filename, latest_test_id, csv_row_id, None, None)
                except Exception as e:
                    msg = f"{e}"
                    print(f"{msg}")
                    print("insert_error_output")
                    # insert_error_output(mycursor, msg, latest_test_id)

    # mydb.commit()
