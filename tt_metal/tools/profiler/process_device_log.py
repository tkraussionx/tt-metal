#!/usr/bin/env python3

import os
import sys
import csv

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from rich import print
import plotly.express as px
import pandas as pd
import seaborn as sns

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
                devicePlotData[row.core][row.risc] = {
                    "data": {(0,1):[(0,row.start,row.start)],row.duration:[return_row_data_tuple(row)]},
                    "order":[((0,1),0),(row.duration, 0)]
                }
        else:
            devicePlotData[row.core] = {
                row.risc:{
                    "data": {(0,1):[(0,row.start,row.start)],row.duration:[return_row_data_tuple(row)]},
                    "order":[((0,1),0),(row.duration, 0)]
                }
            }

    riscs = ['BRISC', 'NCRISC']

    xValsDict = {risc:[] for risc in riscs}
    traces = {risc:[] for risc in riscs}

    coreOrderTrav = {core:{risc:0 for risc in devicePlotData[core].keys()} for core in devicePlotData.keys()}
    for risc in riscs:
        ordering = True
        traceToAdd = None
        discardedTraces = set()
        while ordering:

            ordering = False
            addTrace = True
            for core in devicePlotData.keys():
                if risc in devicePlotData[core].keys():
                    if coreOrderTrav[core][risc] < len(devicePlotData[core][risc]["order"]):
                        ordering = True
                        trace = devicePlotData[core][risc]["order"][coreOrderTrav[core][risc]]
                        if traceToAdd:
                            if core not in traceToAdd[1]:
                                if traceToAdd[0] == trace:
                                    traceToAdd[1].add(core)
                                else:
                                    #Let see if any trace in the future is the candidate for this core
                                    for i in range (coreOrderTrav[core][risc]+1, len(devicePlotData[core][risc]["order"])):
                                        futureTrace = devicePlotData[core][risc]["order"][i]
                                        if futureTrace == traceToAdd[0] and traceToAdd[0] not in discardedTraces:
                                            #Pick a better candidate
                                            discardedTraces.add(traceToAdd[0])
                                            traceToAdd = (trace,set([core]))
                                            addTrace = False
                                            break
                                    if addTrace == False:
                                        break
                        else:
                            #Pick a new candidate
                            traceToAdd = (trace,set([core]))
                            addTrace = False
                            break

            if addTrace and traceToAdd:
                if traceToAdd[0] in discardedTraces:
                    discardedTraces.remove(traceToAdd[0])
                traces[risc].append(traceToAdd)
                for core in traceToAdd[1]:
                    if risc in devicePlotData[core].keys():
                        coreOrderTrav[core][risc] += 1
                traceToAdd = None

    print(traces)
    for risc in traces.keys():
        for trace in traces[risc]:
            xVals = []
            traceType = trace[0]
            cores = trace[1]
            for core in yVals:
                xVal = 0
                if core in cores:
                    xVal = devicePlotData[core][risc]["data"][traceType[0]][traceType[1]][2]
                xVals.append(xVal)
            xValsDict[risc].append((traceType,xVals))

    layout = go.Layout(xaxis=dict(title="Cycle count"), yaxis=dict(title="Cores"))
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
    riscColors = {
        'BRISC': "light:b",
        'NCRISC': "light:r",
    }
    for risc in riscs:
        durations = []
        for xVals in xValsDict[risc]:
            duration = xVals[0][0]
            if duration not in durations:
                durations.append(duration)

        colors = sns.color_palette(riscColors[risc],len(durations) + 1).as_hex()
        colorMap = {duration:color for duration,color in zip(durations,colors)}
        colorMap [(4,1)] = "rgba(255, 255, 255, 0.0)"
        colorMap [(0,1)] = "rgba(255, 255, 255, 0.0)"
        colorMap [(1,2)] = colors[-1]
        colorMap [(3,4)] = colors[-1]

        for xVals in xValsDict[risc]:
            duration = xVals[0][0]
            color = colorMap[duration]

            showlegend = False

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
                        "%{customdata}",
                        "%{x} cycles",
                    ]),
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
        barmode="stack",
        height=BASE_HEIGHT + PER_CORE_HEIGHT * len(yVals)
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
