#!/usr/bin/env python3

import os
import sys
import csv

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from rich import print
import plotly.express as px
import pandas as pd
import numpy as np

import plot_setup


CYCLE_COUNT_TO_MILISECS = 1.2e6
BASE_HEIGHT = 200
PER_CORE_HEIGHT = 90

REARRANGED_TIME_CSV = "device_arranged_timestamps.csv"
DEVICE_STATS_TXT = "device_stats.txt"
DEVICE_PERF_HTML = "device_perf.html"
DEVICE_TIME_CSV = "logs/profile_log_device.csv"

DEVICE_PERF_RESULTS = "device_perf_results.tar"

def coreCompare(coreStr):
    x = int(coreStr.split(",")[0])
    y = int(coreStr.split(",")[1])
    return x + y * 100


def generate_analysis_table(analysisData):
    stats = set()
    for analysis in analysisData.keys():
        for stat in analysisData[analysis].keys():
            stats.add(stat)

    stats = sorted(stats)
    return html.Table(
        # Header
        [html.Tr([html.Th("Type")] + [html.Th(f"{stat} [cycles]") for stat in stats])]
        +
        # Body
        [
            html.Tr(
                [html.Td(f"{analysis}")]
                + [html.Td(f"{analysisData[analysis][stat]:.0f}")\
                   if stat in analysisData[analysis].keys() else html.Td("-") for stat in stats]
            )
            for analysis in analysisData.keys()
        ]
    )


def return_available_timer(riscTimers, timerID):
    if timerID in riscTimers.keys():
        return riscTimers[timerID]
    else:
        return ""


def print_arranged_csv(timerVals, timerIDLabels, pcie_slot, freq_text):
    with open(REARRANGED_TIME_CSV, "w") as timersCSV:
        header = ["core_x", "core_y"]
        timeWriter = csv.writer(timersCSV, delimiter=",")

        timeWriter.writerow(["Clock Frequency [GHz]", freq_text])
        timeWriter.writerow(["PCIe slot",pcie_slot])
        timeWriter.writerow(
            ["core x", "core y"]
            + [f"BRISC {timerIDLabel[1]}" for timerIDLabel in timerIDLabels]
            + [f"NCRISC {timerIDLabel[1]}" for timerIDLabel in timerIDLabels]
        )
        for core in sorted(timerVals.keys(), key=coreCompare):
            coreSplit = core.split(",")
            core_x = coreSplit[0].strip()
            core_y = coreSplit[1].strip()
            timeWriter.writerow(
                [core_x, core_y]
                + [
                    return_available_timer(timerVals[core]["BRISC"], timerIDLabel[0])
                    for timerIDLabel in timerIDLabels
                ]
                + [
                    return_available_timer(timerVals[core]["NCRISC"], timerIDLabel[0])
                    for timerIDLabel in timerIDLabels
                ]
            )

def analyze_stats(timerStats, timerStatsCores):
    FW_START_VARIANCE_THRESHOLD = 1e3
    if int(timerStats["FW start"]["Max"])  > FW_START_VARIANCE_THRESHOLD:
        print(f"NOTE: Variance on FW starts seems too high at : {timerStats['FW start']['Max']} [cycles]")
        print(f"Please reboot the host to make sure the device is not in a bad reset state")

def print_stats_outfile(timerStats, timerStatsCores):
    original_stdout = sys.stdout
    with open(DEVICE_STATS_TXT, "w") as statsFile:
        sys.stdout = statsFile
        print_stats(timerStats, timerStatsCores)
        sys.stdout = original_stdout


def print_stats(timerStats, timerStatsCores):

    numberWidth = 12
    sampleCores = list(timerStatsCores.keys())
    durationTypes = set()
    for coreDurations in timerStatsCores.values():
        for durationType in coreDurations.keys():
            durationTypes.add(durationType)
    for duration in sorted(durationTypes):
        print()
        print(f"=================== {duration} ===================")
        for stat in sorted(timerStats[duration].keys()):
            print(f"{stat:>12} [cycles] = {timerStats[duration][stat]:>13,.0f}")
        print()
        for core_y in range(-3, 11):
            # Print row number
            if core_y > -1 and core_y < 5:
                print(f"{core_y:>2}|| ", end="")
            elif core_y > 5:
                print(f"{core_y-1:>2}|| ", end="")
            else:
                print(f"{' ':>4} ", end="")

            for core_x in range(-1, 12):
                if core_x > -1:
                    if core_y == -3:
                        print(f"{core_x:>{numberWidth}}", end="")
                    elif core_y == -2:
                        print(f"{'=':=>{numberWidth}}", end="")
                    elif core_y == -1:
                        if core_x in [0, 3, 6, 9]:
                            print(f"{f'DRAM{int(core_x/3)}':>{numberWidth}}", end="")
                        else:
                            print(f"{'---':>{numberWidth}}", end="")
                    elif core_y != 5:
                        core = f"{core_x},{core_y}"
                        if core_y > 5:
                            core = f"{core_x},{core_y-1}"
                        if (
                            core in timerStatsCores.keys()
                            and duration in timerStatsCores[core].keys()
                        ):
                            print(
                                f"{timerStatsCores[core][duration]:>{numberWidth},}",
                                end="",
                            )
                        else:
                            print(f"{'X':>{numberWidth}}", end="")
                    else:
                        if core_x in [0, 3, 6, 9]:
                            print(
                                f"{f'DRAM{4 + int(core_x/3)}':>{numberWidth}}", end=""
                            )
                        else:
                            print(f"{'---':>{numberWidth}}", end="")

                else:
                    if core_y == 1:
                        print("ARC", end="")
                    elif core_y == 3:
                        print("PCI", end="")
                    elif core_y > -1:
                        print("---", end="")
                    else:
                        print("   ", end="")

            print()
        print()
        print()
        print()
    for duration in timerStats.keys():
        if duration not in durationTypes:
            print(f"=================== {duration} ===================")
            for stat in sorted(timerStats[duration].keys()):
                print(f"{stat:>12} [cycles] = {timerStats[duration][stat]:>13,.0f}")


def print_help():
    print(
        "Please choose a plot setup class that matches your test kernel profile data."
    )
    print("e.g. : process_device_log.py test_add_two_ints")
    print("Or run default by providing no args")
    print("e.g. : process_device_log.py")


def import_device_profile_log (logPath):
    deviceData = {}
    with open(logPath) as csvFile:
        csvReader = csv.reader(csvFile, delimiter=",")
        for lineCount,row in enumerate(csvReader):
            if lineCount > 1:
                chipID = int(row[0])
                core = (int(row[1]), int(row[2]))
                risc = row[3].strip()
                timerID = int(row[4])
                timeData = int(row[5])

                if chipID in deviceData.keys():
                    if core in deviceData[chipID].keys():
                        if risc in deviceData[chipID][core].keys():
                            deviceData[chipID][core][risc]["timeSeries"].append(
                                (timerID, timeData)
                            )
                        else:
                            deviceData[chipID][core][risc] = {
                                "timeSeries": [(timerID, timeData)]
                            }
                    else:
                        deviceData[chipID][core] = {
                            risc: {"timeSeries": [(timerID, timeData)]}
                        }
                else:
                    deviceData[chipID] = {
                        core: {risc: {"timeSeries": [(timerID, timeData)]}}
                    }
    return deviceData


def main(args):
    if len(args) == 1:
        try:
            setup = getattr(plot_setup, args[0])()
            try:
                setup.timerAnalysis.update(setup.timerAnalysisBase)
            except Exception:
                setup.timerAnalysis = setup.timerAnalysisBase
        except Exception:
            print_help()
            return
    elif len(args) == 0:
        try:
            setup = getattr(plot_setup, "test_base")()
            setup.timerAnalysis = setup.timerAnalysisBase
        except Exception:
            print_help()
            return
    else:
        print_help()
        return

    deviceData = import_device_profile_log(DEVICE_TIME_CSV)

    ganttData = []
    for device in deviceData.keys():
        for core in deviceData[device].keys():
            for risc in deviceData[device][core].keys():
                timeSeries = deviceData[device][core][risc]['timeSeries']
                timeSeries.sort(key=lambda x: x[1]) # Sort on timestamp
                for startTime, endTime in zip(timeSeries[:-1],timeSeries[1:]):
                    ganttData.append(
                        dict(core=core, risc=risc, start=startTime[1], end=endTime[1], duration=(startTime[0],endTime[0])))

    df = pd.DataFrame(ganttData)
    df['delta'] = df['end'] - df['start']
    minCycle = min(df.start)
    df['start'] = df['start'] - minCycle
    df['end'] = df['end'] - minCycle

    yVals = sorted(list(set(df.core.tolist())), key=lambda x: x[1]*100 + x[0], reverse=True)

    def return_row_data_tuple(row):
        return (
            row.start,
            row.end,
            row.delta,
        )

    def add_x_val_to_core (xVal, xVals, core):
        for yVal in yVals:
            xValAdd = 0
            if core == yVal:
                xValAdd = xVal
            xVals.append(xValAdd)
        return xVals

    devicePlotData = {}
    for index, row in df.iterrows():
        if row.core in devicePlotData.keys():
            if row.risc in devicePlotData[row.core].keys():
                if row.duration in devicePlotData[row.core][row.risc]["data"].keys():
                    devicePlotData[row.core][row.risc]["data"][row.duration].append(return_row_data_tuple(row))
                else:
                    devicePlotData[row.core][row.risc]["data"][row.duration] = [return_row_data_tuple(row)]
                devicePlotData[row.core][row.risc]["order"].append((row.duration,
                                                                   len(devicePlotData[row.core][row.risc]["data"][row.duration])-1))
            else:
                if row.start != 0:
                    devicePlotData[row.core][row.risc] = {
                        "data": {(-1,0):[(0,row.start,row.start)],row.duration:[return_row_data_tuple(row)]},
                        "order":[((-1,0),0),(row.duration, 0)]
                    }
                else:
                    devicePlotData[row.core][row.risc] = {
                        "data": {row.duration:[return_row_data_tuple(row)]},
                        "order":[(row.duration, 0)]
                    }
        else:
            if row.start != 0:
                devicePlotData[row.core] = {
                    row.risc:{
                        "data": {(-1,0):[(0,row.start,row.start)],row.duration:[return_row_data_tuple(row)]},
                        "order":[((-1,0),0),(row.duration, 0)]
                    }
                }
            else:
                devicePlotData[row.core] = {
                    row.risc:{
                        "data": {row.duration:[return_row_data_tuple(row)]},
                        "order":[(row.duration, 0)]
                    }
                }

    # print(devicePlotData)

    riscs = ['BRISC','NCRISC']
    riscColors = {
        'BRISC': "rgba(255,0,100,{})",
        'NCRISC': "rgba(100,0,255,{})",
    }

    xValsDict = {risc:[] for risc in riscs}

    for core in devicePlotData.keys():
        for risc in devicePlotData[core].keys():
            for durationType in devicePlotData[core][risc]["order"]:
                    xValsDict[risc].append(((durationType,core), []))

    for risc in riscs:
        for xVals in xValsDict[risc]:
            xVal = 0
            core = xVals[0][1]
            duration = xVals[0][0][0]
            durationIndex = xVals[0][0][1]
            if risc in devicePlotData[core].keys():
                xVal = devicePlotData[core][risc]["data"][duration][durationIndex][2]

            add_x_val_to_core(xVal, xVals[1], core)

    print(xValsDict)

    layout = go.Layout(xaxis=dict(title="Cycle count"), yaxis=dict(title="Cores"), hoverlabel=dict(bgcolor="black"))
    fig = go.Figure(layout=layout)

    fig.add_trace(
        go.Bar(
            y=[yVals, [" "] * len(yVals)],
            x=[0] * len(yVals),
            orientation="h",
            showlegend=False,
            marker=dict(color="rgba(255, 255, 255, 0.0)"),
        )
    )
    for risc in riscs:
        for xVals in xValsDict[risc]:
            duration = xVals[0][0][0]
            if duration == (4,1) or duration == (-1,0):
                color = "rgba(255,255,255,0.0)"
            else:
                color = riscColors[risc].format((30*duration[1] - 10*duration[0])/100)

            showlegend = False
            # if combo[2] == "blank":
                # showlegend = False

            fig.add_trace(
                go.Bar(
                    y=[yVals, [risc] * len(yVals)],
                    x=xVals[1],
                    orientation="h",
                    name=f"{duration}",
                    showlegend=showlegend,
                    marker=dict(color=color),
                    customdata=[duration for i in range(len(xVals[1]))],
                    hovertemplate="<br>".join([
                        "duration type: %{customdata}",
                        "duration: %{x}",
                    ])
                )
            )
    fig.add_trace(
        go.Bar(
            y=[yVals, [""] * len(yVals)],
            x=[0] * len(yVals),
            orientation="h",
            showlegend=False,
            marker=dict(color="rgba(255, 255, 255, 0.0)"),
        )
    )

    fig.update_layout(
        barmode="stack", height=BASE_HEIGHT + PER_CORE_HEIGHT * len(yVals)
    )

    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    app = Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div(
        [
            html.H1("Device Performance"),
            html.Br(),
            html.H3("Stats Table"),
            # generate_analysis_table(timerStats),
            dcc.Graph(figure=fig),
        ]
    )

    app.run_server(host="0.0.0.0", debug=True)


if __name__ == "__main__":
    main(sys.argv[1:])
