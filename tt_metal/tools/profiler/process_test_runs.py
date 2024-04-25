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
from sklearn.linear_model import TheilSenRegressor
from sklearn.datasets import make_regression

from tt_metal.tools.profiler.common import (
    TT_METAL_HOME,
    PROFILER_DEVICE_SIDE_LOG,
    PROFILER_BIN_DIR,
    TRACY_FILE_NAME,
    TRACY_CSVEXPROT_TOOL,
)

RUN_TYPE = "runs_400ms_1_1_core"

lsCmd = subprocess.run(
    f"cd {TT_METAL_HOME / RUN_TYPE}; ls",
    shell=True,
    check=True,
    capture_output=True,
)

runsList = lsCmd.stdout.decode().split("\n")[:-1]

runData = {}

pd.options.display.float_format = "{:,.2f}".format
pd.options.display.max_rows = 255
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

        syncinfoDF = pd.read_csv(syncInfoFile)
        frequencyRec = syncinfoDF.iloc[0]["frequency"]
        delayRecCycle = syncinfoDF.iloc[0]["delay"]
        hostStart = syncinfoDF.iloc[0]["host_start"] * syncinfoDF.iloc[0]["tracy_ratio"]

        # customSyncData = syncinfoDF.copy()
        # customSyncDataDiff = customSyncData.diff()
        # customSyncData["host_real_delta"] = customSyncDataDiff["host_real"]
        # customSyncData["device_delta"] = customSyncDataDiff["device"]

        # host_delta_mean = customSyncData["host_real_delta"].mean()
        # # print (customSyncData[["device","host_real", "device_delta", "host_real_delta"]])
        # margin = 1/1e5
        # customSyncData =  customSyncData.loc[(customSyncData["host_real_delta"] < host_delta_mean * (1+margin)) & (customSyncData["host_real_delta"] > host_delta_mean * (1-margin))]
        # print (f"Count: {customSyncData['write_overhead'].count()}")

        # X = [[x] for x in list(customSyncData["host_real"])]
        # y = list(customSyncData["device"])
        # reg = None
        # with contextlib.redirect_stderr(StringIO()) as f:
        # reg = TheilSenRegressor(random_state=0).fit(X, y)

        deviceDF = pd.read_csv(deviceCsvFile, skiprows=[0])
        deviceDF = deviceDF.loc[(deviceDF[" zone name"] == "CQ-CONSUMER-MAIN") & (deviceDF[" zone phase"] == "end")]
        deviceDF = deviceDF.reset_index()
        deviceDF = deviceDF.drop(list(range(0, testPoints * 2, 2)))  # remove CONSUMER_MAIN that is not relevant
        deviceDF = deviceDF.reset_index()

        tracyBaseTime_ns = 0
        trials = 0
        diff = 0
        while trials < 1:
            trials += 1
            deviceDF = deviceDF[[" time[cycles since reset]"]]
            deviceDF["time[ns since reset]"] = deviceDF[" time[cycles since reset]"] / frequencyRec
            deviceDF["time[cycles host synced]"] = deviceDF[" time[cycles since reset]"] - delayRecCycle
            deviceDF["time[ns host synced]"] = deviceDF["time[cycles host synced]"] / frequencyRec

            if not tracyBaseTime_ns:
                for i in range(testPoints):
                    baseTime = round(tracyDF.iloc[i]["ns_end_time"] - deviceDF.iloc[i]["time[ns host synced]"])
                    if tracyBaseTime_ns:
                        assert (
                            baseTime < tracyBaseTime_ns + 2 and baseTime > tracyBaseTime_ns - 2
                        ), f"{baseTime}, {tracyBaseTime_ns}"
                    else:
                        tracyBaseTime_ns = baseTime

            deviceDF["corrected device time [ns]"] = round(deviceDF["time[ns host synced]"] + tracyBaseTime_ns)
            testDF = deviceDF[["corrected device time [ns]"]].copy()
            testDF["tracy device time [ns]"] = tracyDF.loc[(tracyDF["name"] == "CQ-CONSUMER-MAIN")]["ns_end_time"] * 1.0
            testDF["tracy host time [ns]"] = (
                tracyDF.loc[(tracyDF["name"] == "HWCommandQueue_finish")].reset_index()["ns_end_time"] * 1.0
            )
            testDF["original host device diff [ns]"] = testDF["tracy host time [ns]"] - testDF["tracy device time [ns]"]
            testDF["new host device diff [ns]"] = testDF["tracy host time [ns]"] - testDF["corrected device time [ns]"]

            # print(testDF["new host device diff [ns]"])
            # print (frequencyRec, delayRecCycle)
            # if not diff:
            # diff = round(testDF.iloc[0][4] - testDF.iloc[1][4])
            # frequencyRec = reg.coef_[0]
            # delayRecCycle = reg.intercept_
            # else:
            # # print(diff, round(testDF.iloc[0][4] - testDF.iloc[1][4]))
            # # print(abs(diff) - abs(round(testDF.iloc[0][4] - testDF.iloc[1][4])))
            # if abs(diff) < 100:
            # break

            # if diff < 0:
            # frequencyRec -= 1/1e9
            # if diff > 0:
            # frequencyRec += 1/1e9
