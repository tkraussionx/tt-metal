#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os
import sys
import subprocess
from io import StringIO
import contextlib
from loguru import logger

import pandas as pd
from sklearn.linear_model import TheilSenRegressor, LinearRegression
from sklearn.datasets import make_regression

from tt_metal.tools.profiler.common import (
    TT_METAL_HOME,
    PROFILER_DEVICE_SIDE_LOG,
    PROFILER_BIN_DIR,
    TRACY_FILE_NAME,
    TRACY_CSVEXPROT_TOOL,
)

RUN_TYPE = "runs"

lsCmd = subprocess.run(
    f"cd {TT_METAL_HOME / RUN_TYPE}; ls",
    shell=True,
    check=True,
    capture_output=True,
)

runsList = lsCmd.stdout.decode().split("\n")[:-1]

runData = {}

pd.options.display.float_format = "{:,.2f}".format
pd.options.display.max_rows = 2550

syncDataAll = pd.DataFrame()
for run in runsList:
    tracyCsv = ""
    logsFolder = TT_METAL_HOME / RUN_TYPE / run / ".logs"
    tracyFile = logsFolder / TRACY_FILE_NAME
    deviceCsvFile = logsFolder / PROFILER_DEVICE_SIDE_LOG
    syncInfoFile = logsFolder / "sync_device_info.csv"

    filesExist = True
    for file in [tracyFile, deviceCsvFile, syncInfoFile]:
        filesExist &= os.path.isfile(file)

    testPoints = 2
    if filesExist:
        syncinfoDF = pd.read_csv(syncInfoFile)
        if syncDataAll.empty:
            syncDataAll = syncinfoDF.copy()
        else:
            syncDataAll = pd.concat([syncDataAll, syncinfoDF.copy()])

syncDataAll["host_tracy_global [ns]"] = (syncDataAll["host_tracy"] + syncDataAll["host_start"]) * syncDataAll[
    "tracy_ratio"
].mean()

syncDataAll.reset_index()

# print(syncDataAll)
# print(syncDataAll.diff())


syncDataAll.to_excel("output.xlsx")
# X = [[x] for x in list(syncDataAll["host_tracy_global [ns]"])]
# y = list(syncDataAll["device"])
X = [[x] for x in list(syncDataAll.iloc[:250]["host_real"])]
y = list(syncDataAll.iloc[:250]["device"])
# X = [[syncDataAll.iloc[260]["host_tracy_global [ns]"]], [syncDataAll.iloc[700]["host_tracy_global [ns]"]]]
# y = [syncDataAll.iloc[260]["device"], syncDataAll.iloc[700]["device"]]
with contextlib.redirect_stderr(StringIO()) as f:
    # reg = TheilSenRegressor(random_state=0).fit(X, y)
    reg = LinearRegression().fit(X, y)

frequency = reg.coef_[0]
delayCycle = reg.intercept_

print(frequency, delayCycle)

for run in runsList:
    tracyCsv = ""
    logsFolder = TT_METAL_HOME / RUN_TYPE / run / ".logs"
    tracyFile = logsFolder / TRACY_FILE_NAME
    deviceCsvFile = logsFolder / PROFILER_DEVICE_SIDE_LOG
    syncInfoFile = logsFolder / "sync_device_info.csv"

    filesExist = True
    for file in [tracyFile, deviceCsvFile, syncInfoFile]:
        filesExist &= os.path.isfile(file)

    testPoints = 2
    if filesExist:
        tracyDF = pd.DataFrame()
        for zoneName in ["CQ-CONSUMER-MAIN", "HWCommandQueue_finish"]:
            csvExportCmd = subprocess.run(
                f"{PROFILER_BIN_DIR / TRACY_CSVEXPROT_TOOL} -u -f {zoneName} {tracyFile}",
                shell=True,
                check=True,
                capture_output=True,
            )

            tracyCsv = csvExportCmd.stdout.decode()
            if not tracyDF.empty:
                tracyDF = pd.concat([tracyDF, pd.read_csv(StringIO(tracyCsv))])
            else:
                tracyDF = pd.read_csv(StringIO(tracyCsv))

        tracyDF = tracyDF.reset_index()
        tracyDF = tracyDF.drop(list(range(0, testPoints * 2, 2)))  # remove CONSUMER_MAIN that is not relevant
        tracyDF = tracyDF.reset_index()
        tracyDF = tracyDF.drop(columns=["level_0", "index"])
        tracyDF["ns_end_time"] = tracyDF["ns_since_start"] + tracyDF["exec_time_ns"]
        # customSyncData["device_delta"] = customSyncDataDiff["device"]

        # host_delta_mean = customSyncData["host_real_delta"].mean()
        # # print (customSyncData[["device","host_real", "device_delta", "host_real_delta"]])
        # margin = 1/1e5
        # customSyncData =  customSyncData.loc[(customSyncData["host_real_delta"] < host_delta_mean * (1+margin)) & (customSyncData["host_real_delta"] > host_delta_mean * (1-margin))]
        # print (f"Count: {customSyncData['write_overhead'].count()}")

        syncinfoDF = pd.read_csv(syncInfoFile)
        hostStart = syncinfoDF.iloc[0]["host_start"] * syncinfoDF.iloc[0]["tracy_ratio"]
        tracyBaseTime_ns = syncinfoDF.iloc[0]["tracy_base_time"] * syncinfoDF.iloc[0]["tracy_ratio"]
        frequencyRec = syncinfoDF.iloc[0]["frequency"]
        delayRecCycle = syncinfoDF.iloc[0]["delay"]

        # frequency = frequencyRec
        # delayCycle = delayRecCycle
        print(frequencyRec, delayRecCycle)

        deviceDF = pd.read_csv(deviceCsvFile, skiprows=[0])
        deviceDF = deviceDF.loc[(deviceDF[" zone name"] == "CQ-CONSUMER-MAIN") & (deviceDF[" zone phase"] == "end")]
        deviceDF = deviceDF.reset_index()
        deviceDF = deviceDF.drop(list(range(0, testPoints * 2, 2)))  # remove CONSUMER_MAIN that is not relevant
        deviceDF = deviceDF.reset_index()

        deviceDF = deviceDF[[" time[cycles since reset]"]]
        deviceDF["time[ns since reset]"] = deviceDF[" time[cycles since reset]"] / frequency
        deviceDF["time[cycles host synced]"] = deviceDF[" time[cycles since reset]"] - delayCycle
        deviceDF["time[ns host synced]"] = deviceDF["time[cycles host synced]"] / frequency
        deviceDF["corrected device time [ns]"] = round(deviceDF["time[ns host synced]"] + hostStart - tracyBaseTime_ns)

        testDF = deviceDF[["corrected device time [ns]"]].copy()
        testDF["tracy device time [ns]"] = tracyDF.loc[(tracyDF["name"] == "CQ-CONSUMER-MAIN")]["ns_end_time"] * 1.0
        testDF["tracy host time [ns]"] = (
            tracyDF.loc[(tracyDF["name"] == "HWCommandQueue_finish")].reset_index()["ns_end_time"] * 1.0
        )
        testDF["original host device diff [ns]"] = testDF["tracy host time [ns]"] - testDF["tracy device time [ns]"]
        testDF["new host device diff [ns]"] = testDF["tracy host time [ns]"] - testDF["corrected device time [ns]"]

        # print(deviceDF)
        print(testDF)
