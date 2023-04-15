#!/usr/bin/env python3

import os
import csv
from pathlib import Path
import json

from rich import print
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

from process_device_log import import_log_run_stats,generate_plots, import_host_log_run_stats
import plot_setup

OPS_LOGS_DIR = "/tmp/tt_perf/ops"

HOST_SIDE_LOG = "profile_log_host.csv"
DEVICE_SIDE_LOG = "profile_log_device.csv"
HOST_SIDE_LOG = "profile_log_host.csv"
OPS_CSV = "logs/profile_log_ops.csv"
MERGED_DEVICE_SIDE_LOG = "logs/profile_log_device_merged.csv"

OPS_CSV_HEADER = [
    "OP NAME",
    "CALL COUNT",
    "HOST Start TS",
    "HOST End TS",
    "HOST Duration [ns]",
    "DEVICE Start Cycle",
    "DEVICE End Cycle",
    "DEVICE Duration [ns]",
    "Device to Host Utilization %",
    "Parallelization Strategy",
    "Core Count"]

HOST_SIDE_STATS = ["Count", "Average"]
HOST_FUNCSTION_HEADER_FORMAT = "{} {}"

ttMetalFunctionsSet = set()
def append_detail_host_time_data (opCandidatePath, call_count, timeDataDict):
    hostLogPath = os.path.join(opCandidatePath, f"{call_count}",HOST_SIDE_LOG)
    if os.path.isfile(hostLogPath):
        hostData = import_host_log_run_stats(hostLogPath)
        for functionName, calls in hostData.items():
            ttMetalFunctionsSet.add(functionName)
            for stat in HOST_SIDE_STATS:
                assert stat in calls["stats"].keys()
                functionKey = HOST_FUNCSTION_HEADER_FORMAT.format(functionName, stat)
                timeDataDict[functionKey] = int(calls["stats"][stat])


def append_device_time_data (opCandidatePath, call_count, timeDataDict):
    deviceLogPath = os.path.join(opCandidatePath, f"{call_count}",DEVICE_SIDE_LOG)
    if os.path.isfile(deviceLogPath):
        setup=plot_setup.default_setup()
        setup.deviceInputLog = deviceLogPath
        setup.timerAnalysis = {}

        devicesData = import_log_run_stats(setup)
        # with open(deviceLogPath, "r") as deviceCSV:
            # deviceCSVData = deviceCSV.read().strip()


        start_ID, start_ts, start_risc, start_core = devicesData['devices'][0]['cores']['DEVICE']['riscs']['TENSIX']['timeseries'][0]
        end_ID, end_ts, end_risc, end_core = devicesData['devices'][0]['cores']['DEVICE']['riscs']['TENSIX']['timeseries'][-1]

        cores = list(devicesData['devices'][0]['cores'].keys())
        cores.remove("DEVICE")

        delta_time = end_ts - start_ts
        delta_time_ns = delta_time / setup.coreFreq

        # assert delta_time_ns < timeDataDict["HOST Duration [ns]"]

        timeDataDict ["DEVICE Start Cycle"] = start_ts
        timeDataDict ["DEVICE End Cycle"] = end_ts
        timeDataDict ["DEVICE Duration [ns]"] = int(delta_time_ns)
        timeDataDict ["Core Count"] = len(cores)
        timeDataDict ["Device to Host Utilization %"] = int(100* (delta_time_ns / timeDataDict["HOST Duration [ns]"]))
    else:
        timeDataDict ["DEVICE Start Cycle"] = "-"
        timeDataDict ["DEVICE End Cycle"] = "-"
        timeDataDict ["DEVICE Duration [ns]"] = "-"
        timeDataDict ["Core Count"] = "-"
        timeDataDict ["Device to Host Utilization %"] = "-"

minTime = 0
maxDiff = 0
def parse_ops_logs():
    global minTime,maxDiff
    ops = {}

    paths = sorted(Path(OPS_LOGS_DIR).iterdir(), key=os.path.getmtime, reverse=True)

    for opCandidate in paths:
        opCandidatePath = os.path.join(OPS_LOGS_DIR, opCandidate)
        if os.path.isdir(opCandidatePath):
            op = opCandidate
            opLogPath = os.path.join(opCandidatePath, HOST_SIDE_LOG)
            with open(opLogPath, "r") as csvFile:
                csvReader = csv.reader(csvFile, delimiter=",")
                for lineCount,row in enumerate(csvReader):
                    if lineCount > 0:
                        op_name = row[1].strip()
                        start_ts = int(row[2].strip())
                        end_ts = int(row[3].strip())
                        delta_time = int(row[4].strip())

                        metadata = row[0].strip().split("-")
                        assert len(metadata) > 1

                        call_count = int(metadata[0])
                        parallel_strategy = metadata[1]

                        if minTime == 0:
                            minTime = start_ts
                        elif minTime > start_ts:
                            minTime = start_ts

                        if maxDiff < delta_time:
                            maxDiff = delta_time

                        if op_name in ["eltwise_unary", "eltwise_binary"]:
                            assert len(metadata) > 2
                            op_name = metadata[2].lower()

                        timeDataDict = {
                            "CALL COUNT" : call_count,
                            "HOST Start TS" : start_ts,
                            "HOST End TS" : end_ts,
                            "HOST Duration [ns]" : delta_time,
                            "Parallelization Strategy" : parallel_strategy
                        }

                        append_device_time_data (opCandidatePath, call_count, timeDataDict)
                        append_detail_host_time_data (opCandidatePath, call_count, timeDataDict)

                        if op_name in ops.keys():
                            ops[op_name].append(timeDataDict)
                        else:
                            ops[op_name] =[timeDataDict]
    return ops

preFig = go.Figure()
def run_dashbaord_webapp():

    global preFig
    curveDict = {}
    curveNumber = 0
    fig = go.Figure()
    for op, opCalls in ops.items():
        xVals = []
        yVals = []
        Xs = []
        Ys = []
        Cs = []
        Ss = []
        diffs = []
        names = []
        print (op)
        for opCall in opCalls:
            s = opCall['HOST Start TS'] - minTime
            e = opCall['HOST End TS'] - minTime
            c = opCall['CALL COUNT']
            diff = opCall['HOST Duration [ns]']
            ps = opCall['Parallelization Strategy']
            m = (s+e)//2
            xVals += [None,s,e,e,s,s]
            yVals += [None,0,0,1,1,0]
            Xs += [m]
            Ys += [0.5]
            Cs += [c]
            diffs += [diff/1e9]
            names += [op]
            Ss += [ps]

            curveDict [curveNumber] = {
                "op" : op,
                "callCount" : c
            }
            curveNumber += 1

        fig.add_trace(go.Scatter(
            x=xVals,
            y=yVals,
            name = op,
            hoverinfo = "none",
            mode='none',
            fill="toself")
        )

        fig.add_trace(
            go.Scatter(
                x=Xs,
                y=Ys,
                name = "",
                customdata = np.stack((names,Cs,diffs, Ss), axis=-1),
                hovertemplate="<br>".join(["Op: %{customdata[0]}", "Call: %{customdata[1]}", "Duration: %{customdata[2]:.3f} s", "Meta: %{customdata[3]}"]),
                mode='markers',
                marker_size = 60,
                hoverlabel=dict(
                    bgcolor="white",
                ),
                hoverinfo = "x",
                showlegend = False,
                opacity = 0,
            )
        )
        fig.update_layout(
            xaxis=dict(
                range=[-1e7, maxDiff + 1e7],
                rangeslider=dict(
                    visible=True,
                ),
            )
        )

    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    app = Dash(__name__, external_stylesheets=external_stylesheets)
    app.layout = html.Div(
        [
            html.H5(f"OPs:", id="text"),
            dcc.Graph(figure=fig, id="plot"),
            dcc.Graph(figure=go.Figure(), id="plot-2"),
        ]
    )
    @app.callback(
        Output('text', 'children'),
        [Input('plot', 'hoverData')])
    def display_hover_data(hoverData):
        data = {}
        try:
            data = hoverData["points"][0]["curveNumber"]
        except TypeError:
            data = {}
        except NameError:
            data = {}
        except KeyError:
            data = {}
        return json.dumps(hoverData, indent=2)

    @app.callback(
        Output('plot-2', 'figure'),
        [Input('plot', 'hoverData')])
    def plot_device_data(hoverData):
        global preFig
        fig = preFig
        if hoverData and "points" in hoverData.keys():
            if len(hoverData["points"]) > 0:
                if "customdata" in hoverData["points"][0].keys():
                    op = hoverData["points"][0]["customdata"][0]
                    if op in ["exp","surecipb","gelua", "relu", "sqrt", "sigmoid", "log", "tanh"]:
                        op = "eltwise_binary"
                    if op in ["add","mul","sub"]:
                        op = "eltwise_binary"
                    callCount = hoverData["points"][0]["customdata"][1]
                    filePath = f"{OPS_LOGS_DIR}/{op}/{callCount}/{DEVICE_SIDE_LOG}"
                    setup=plot_setup.default_setup()
                    setup.deviceInputLog = filePath
                    setup.timerAnalysis = {}

                    devicesData = import_log_run_stats(setup)
                    figs = generate_plots(devicesData,setup)
                    for fig in figs.values():
                        preFig = fig

        return fig

    app.run_server(host="0.0.0.0", debug=True)

def print_ops_csv(ops):

    with open(OPS_CSV, "w") as opsCSV:

        opsWriter = csv.writer(opsCSV, delimiter=",")
        hostFunctions = []
        for functionName in sorted(ttMetalFunctionsSet):
            for stat in HOST_SIDE_STATS:
                functionKey = HOST_FUNCSTION_HEADER_FORMAT.format(functionName, stat)
                if "Count" not in functionKey:
                    hostFunctions.append(f"{functionKey} [ns]")
                else:
                    hostFunctions.append(functionKey)
        opsWriter.writerow(OPS_CSV_HEADER + hostFunctions)

        for op, opCalls in ops.items():
            for opCall in opCalls:
                opsROW = [op]
                for item in OPS_CSV_HEADER:
                    if item != "OP NAME":
                        assert item in opCall.keys(), item
                        opsROW.append(opCall[item])
                for functionName in sorted(ttMetalFunctionsSet):
                    for stat in HOST_SIDE_STATS:
                        functionKey = HOST_FUNCSTION_HEADER_FORMAT.format(functionName, stat)
                        if functionKey in opCall.keys():
                            opsROW.append(opCall[functionKey])
                        else:
                            opsROW.append("0")
                opsWriter.writerow(opsROW)

    #run_dashbaord_webapp()

if __name__ == "__main__":
    ops =  parse_ops_logs()
    print(ttMetalFunctionsSet)
    print_ops_csv(ops)
