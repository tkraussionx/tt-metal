#!/usr/bin/env python3

import os
import sys
import csv

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
# from rich import print
import pandas as pd
import seaborn as sns

import plot_setup


CYCLE_COUNT_TO_MILISECS = 1.2e6
BASE_HEIGHT = 200
PER_CORE_HEIGHT = 90

CORE_FREQ = 1.2

REARRANGED_TIME_CSV = "device_arranged_timestamps.csv"
DEVICE_STATS_TXT = "device_stats.txt"
DEVICE_PERF_HTML = "device_perf.html"
DEVICE_TIME_CSV = "logs/profile_log_device.csv"

DEVICE_PERF_RESULTS = "device_perf_results.tar"

def coreCompare(core):
    if type(core) == str:
        return (1<<64)-1
    x = core[0]
    y = core[1]
    return x + y * 100


def generate_analysis_table(analysisData, setup):
    stats = setup.displayStats
    return html.Div([
        html.H6("Stats Table"),
        html.Table(
            # Header
            [html.Tr([html.Th("Type")] + [html.Th(f"{stat} [cycles]") for stat in stats])]
            +
            # Body
            [
                html.Tr(
                    [html.Td(f"{analysis}")]
                    + [html.Td(f"{analysisData[analysis]['stats'][stat]:.0f}")\
                       if stat in analysisData[analysis]['stats'].keys() else html.Td("-") for stat in stats]
                )
                for analysis in analysisData.keys()
            ]
        )])

# Note if multiple instances are present, all are returned space delimited
# Further analysis has to be done on the excel side
def return_available_timer(timeseries, desiredTimerID):
    res = ""
    for timerID, timestamp,*metaData in timeseries:
        if timerID == desiredTimerID:
            if res:
                res = f"{res} {timestamp}"
            else:
                res = f"{timestamp}"
    return res


def print_arranged_csv(deviceData, timerIDLabels, freq_text = CORE_FREQ):
    with open(REARRANGED_TIME_CSV, "w") as timersCSV:
        for chipID, deviceData in deviceData.items():
            header = ["core_x", "core_y"]
            timeWriter = csv.writer(timersCSV, delimiter=",")

            timeWriter.writerow(["Clock Frequency [GHz]", freq_text])
            timeWriter.writerow(["PCIe slot",chipID])
            timeWriter.writerow(
                ["core x", "core y"]
                + [f"BRISC {timerIDLabel[1]}" for timerIDLabel in timerIDLabels]
                + [f"NCRISC {timerIDLabel[1]}" for timerIDLabel in timerIDLabels]
            )
            for core in sorted(deviceData.keys(), key=coreCompare):
                if type(core) == tuple:
                    core_x, core_y = core
                    timeWriter.writerow(
                        [core_x, core_y]
                        + [
                            return_available_timer(deviceData[core]["BRISC"]["timeseries"], timerIDLabel[0])
                            for timerIDLabel in timerIDLabels
                        ]
                        + [
                            return_available_timer(deviceData[core]["NCRISC"]["timeseries"], timerIDLabel[0])
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


def print_stats(deviceData, setup):
    numberWidth = 15
    for device in deviceData.keys():
        for analysis in setup.timerAnalysis.keys():
            if analysis in deviceData[device]["DEVICE"]["TENSIX"]["analysis"].keys():
                stats = deviceData[device]["DEVICE"]["TENSIX"]["analysis"][analysis]["stats"]
                print()
                print(f"=================== {analysis} ===================")
                if stats["Count"] > 1:
                    for stat in setup.displayStats:
                        print(f"{stat:>12} [cycles] = {stats[stat]:>10,.0f}")
                else:
                    print(f"{'Duration':>12} [cycles] = {stats['Max']:>10,.0f}")
                print()
                if setup.timerAnalysis[analysis]["across"] in ["risc","core"]:
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
                                    core = (core_x,core_y)
                                    if core_y > 5:
                                        core = (core_x,core_y-1)
                                    if core in deviceData[device].keys():
                                        for risc in deviceData[device][core].keys():
                                            if analysis in deviceData[device][core][risc]["analysis"].keys():
                                                stats = deviceData[device][core][risc]["analysis"][analysis]["stats"]
                                                plusMinus = (stats['Max']-stats['Min'])
                                                median = stats['Median']
                                                tmpStr = f"{median:,}"
                                                if plusMinus:
                                                    tmpStr = "{tmpStr}{sign}{plusMinus:,}".format(
                                                        tmpStr=tmpStr,
                                                        sign=u"\u00B1",
                                                        plusMinus=plusMinus)
                                                print(
                                                    f"{tmpStr:>{numberWidth}}",
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
    # for duration in timerStats.keys():
        # if duration not in durationTypes:
            # print(f"=================== {duration} ===================")
            # for stat in sorted(timerStats[duration].keys()):
                # print(f"{stat:>12} [cycles] = {timerStats[duration][stat]:>13,.0f}")


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
                            deviceData[chipID][core][risc]["timeseries"].append(
                                (timerID, timeData)
                            )
                        else:
                            deviceData[chipID][core][risc] = {
                                "timeseries": [(timerID, timeData)]
                            }
                    else:
                        deviceData[chipID][core] = {
                            risc: {"timeseries": [(timerID, timeData)]}
                        }
                else:
                    deviceData[chipID] = {
                        core: {risc: {"timeseries": [(timerID, timeData)]}}
                    }

    #Sort all timeseries and find global min timestamp
    globalMinTimestamp = (1 << 64) - 1
    for device in deviceData.keys():
        for core in deviceData[device].keys():
            for risc in deviceData[device][core].keys():
                deviceData[device][core][risc]['timeseries'].sort(key=lambda x: x[1])
                firstTimeID, firsTimestamp = deviceData[device][core][risc]['timeseries'][0]
                if globalMinTimestamp > firsTimestamp:
                    globalMinTimestamp = firsTimestamp

    #Include global min timestamp in all timeseries
    for device in deviceData.keys():
        for core in deviceData[device].keys():
            for risc in deviceData[device][core].keys():
                deviceData[device][core][risc]['timeseries'].insert(0,(0,globalMinTimestamp))

    return deviceData


def risc_to_core_timeseries(deviceData):
    for device in deviceData.keys():
        for core in deviceData[device].keys():
            tmpTimeseries = []
            for risc,data in deviceData[device][core].items():
                for ts in data["timeseries"]:
                    tmpTimeseries.append(ts + (risc,))

            tmpTimeseries.sort(key= lambda x:x[1])
            deviceData[device][core]["TENSIX"] = {
                "timeseries" : tmpTimeseries
            }

def core_to_device_timeseries(deviceData):
    for device in deviceData.keys():
        tmpTimeseries = {}
        for core,coreData in deviceData[device].items():
            for risc in coreData.keys():
                for ts in coreData[risc]["timeseries"]:
                    if risc in tmpTimeseries.keys():
                        tmpTimeseries[risc]["timeseries"].append(ts + (core,))
                    else:
                        tmpTimeseries[risc]= {"timeseries": [ts + (core,)]}

        for risc in tmpTimeseries.keys():
            tmpTimeseries[risc]["timeseries"].sort(key= lambda x:x[1])

        deviceData[device]["DEVICE"] = tmpTimeseries

def timeseries_to_durations (deviceData):
    for core in deviceData.keys():
        for risc in deviceData[core].keys():
            deviceData[core][risc]["durations"] = {"data":{},"order":[]}
            timeseries = deviceData[core][risc]['timeseries']
            for startData, endData in zip(timeseries[:-1],timeseries[1:]):
                startTimerID, startTime, *startMeta = startData
                endTimerID, endTime, *endMeta = endData
                start = startTimerID
                if startMeta:
                    start =(startTimerID,) + tuple(startMeta)
                end = endTimerID
                if endMeta:
                    end =(endTimerID,) + tuple(endMeta)
                durationType = (start,end)
                if durationType in deviceData[core][risc]["durations"]["data"].keys():
                    deviceData[core][risc]["durations"]["data"][durationType].append((startTime, endTime, endTime-startTime))
                else:
                    deviceData[core][risc]["durations"]["data"][durationType] = [(startTime, endTime, endTime-startTime)]
                deviceData[core][risc]["durations"]["order"].append((durationType,
                                                                   len(deviceData[core][risc]["durations"]["data"][durationType])-1))


def plotData_to_timelineXVals(plotData, plotCores, setup):
    plotRiscs = setup.riscsData.keys()
    xValsDict = {risc:[] for risc in plotRiscs}
    traces = {risc:[] for risc in plotRiscs}

    coreOrderTrav = {core:{risc:0 for risc in plotData[core].keys()} for core in plotCores}
    for risc in plotRiscs:
        ordering = True
        traceToAdd = None
        discardedTraces = set()
        while ordering:
            ordering = False
            addTrace = True
            for core in plotCores:
                assert(core in plotData.keys())
                if risc in plotData[core].keys():
                    if coreOrderTrav[core][risc] < len(plotData[core][risc]["durations"]["order"]):
                        ordering = True
                        trace = plotData[core][risc]["durations"]["order"][coreOrderTrav[core][risc]]
                        # print(trace)
                        if traceToAdd:
                            if core not in traceToAdd[1]:
                                if traceToAdd[0] == trace:
                                    traceToAdd[1].add(core)
                                else:
                                    #Let see if any trace in the future is the candidate for this core
                                    for i in range (coreOrderTrav[core][risc]+1, len(plotData[core][risc]["durations"]["order"])):
                                        futureTrace = plotData[core][risc]["durations"]["order"][i]
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
                    if risc in plotData[core].keys():
                        coreOrderTrav[core][risc] += 1
                traceToAdd = None

    for risc in traces.keys():
        for trace in traces[risc]:
            xVals = []
            traceType = trace[0]
            cores = trace[1]
            for core in plotCores:
                xVal = 0
                if core in cores:
                    xVal = plotData[core][risc]["durations"]["data"][traceType[0]][traceType[1]][2]
                xVals.append(xVal)
            xValsDict[risc].append((traceType,xVals))
    return xValsDict


def timeline_plot(yVals, xValsDict, setup):
    riscsData = setup.riscsData
    timerIDLabels = setup.timerIDLabels

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
    for risc in setup.riscs:
        durations = []
        if risc in xValsDict.keys():
            for xVals in xValsDict[risc]:
                (duration, instance), xValsData = xVals
                if duration not in durations:
                    durations.append(duration)

            colors = sns.color_palette(riscsData[risc]["color"],len(durations) + 1).as_hex()
            colorMap = {duration:color for duration,color in zip(durations,colors)}
            colorMap ["TRANSPARENT"] = "rgba(255, 255, 255, 0.0)"
            colorMap ["DARK"] = colors[-1]

            for xVals in xValsDict[risc]:
                (duration, instance), xValsData = xVals
                startData, endData = duration
                if type(startData) == tuple:
                    (start,*startMeta) = startData
                else:
                    start = startData
                    startMeta = None
                if type(endData) == tuple:
                    (end,*endMeta) = endData
                else:
                    end = endData
                    endMeta = None

                if (start,end) in [(4,1),(0,1)]:
                    color = colorMap["TRANSPARENT"]
                elif (start,end) in [(1,2),(3,4)]:
                    color = colorMap["DARK"]
                else:
                    color = colorMap[duration]

                for timerID, text in timerIDLabels:
                    if start == timerID:
                        start = text
                    if end == timerID:
                        end = text

                startTxt = f"{start}"
                if startMeta:
                    startTxt = f"{start},{startMeta}"
                endTxt = f"{end}"
                if endMeta:
                    endTxt = f"{end},{endMeta}"
                name = f"{startTxt}->{endTxt}"

                showlegend = False

                fig.add_trace(
                    go.Bar(
                        y=[yVals, [risc] * len(yVals)],
                        x=xValsData,
                        orientation="h",
                        name=name,
                        showlegend=showlegend,
                        marker=dict(color=color),
                        customdata=[name for i in range(len(xValsData))],
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

    return fig

def translate_metaData(metaData, core, risc):
    metaRisc = None
    metaCore = None
    if len(metaData) == 2:
        metaRisc, metaCore = metaData
    elif len(metaData) == 1:
        content = metaData[0]
        if type(content) == str:
            metaRisc = content
        elif type(content) == tuple:
            metaCore = content
    if core != "ANY" and metaCore:
        core = metaCore
    if risc != "ANY" and metaRisc:
        risc = metaRisc
    return core, risc

def determine_conditions (timerID, metaData, analysis):
    currCore = analysis["start"]["core"] if "core" in analysis["start"].keys() else None
    currRisc = analysis["start"]["risc"]
    currStart = (timerID,) + translate_metaData(metaData,currCore,currRisc)

    currCore = analysis["end"]["core"] if "core" in analysis["end"].keys() else None
    currRisc = analysis["end"]["risc"]
    currEnd = (timerID,) + translate_metaData(metaData,currCore,currRisc)

    desStart = (
        analysis["start"]["timerID"],
        analysis["start"]["core"] if "core" in analysis["start"].keys() else None,
        analysis["start"]["risc"],
    )
    desEnd = (
        analysis["end"]["timerID"],
        analysis["end"]["core"] if "core" in analysis["end"].keys() else None,
        analysis["end"]["risc"],
    )
    return currStart, currEnd, desStart, desEnd


def timeseries_analysis(timeseries, analysis):
    tmpList = []
    startFound = None
    for timerID, timestamp, *metaData in timeseries:
        currStart, currEnd, desStart, desEnd = determine_conditions (timerID, metaData, analysis)
        if not startFound:
            if currStart == desStart:
                startFound = (timerID,timestamp)
                if analysis["type"] == "first_last":
                    break
        else:
            if currEnd == desEnd:
                startID , startTS = startFound
                tmpList.append(dict(start=startTS, end=timestamp, durationType=(startID,timerID), diff=timestamp-startTS))
                startFound = None

    if startFound and analysis["type"] == "first_last":
        for i in range(len(timeseries)-1,0,-1):
            timerID, timestamp, *metaData = timeseries[i]
            currStart, currEnd, desStart, desEnd = determine_conditions (timerID, metaData, analysis)
            if currEnd == desEnd:
                startID , startTS = startFound
                tmpList.append(dict(start=startTS, end=timestamp, durationType=(startID,timerID), diff=timestamp-startTS))
                startFound = None
                break

    tmpDF = pd.DataFrame(tmpList)
    tmpDict = {}
    if not tmpDF.empty:
        tmpDict = {
            "analysis": analysis,
            "stats" : {
                "Count" : tmpDF.loc[:,'diff'].count(),
                "Average" : tmpDF.loc[:,'diff'].mean(),
                "Max" : tmpDF.loc[:,'diff'].max(),
                "Min" : tmpDF.loc[:,'diff'].min(),
                "Median" : tmpDF.loc[:,'diff'].median(),
                "Sum" : tmpDF.loc[:,'diff'].sum(),
                "First" : tmpDF.loc[0,'diff'],
            }
        }
    return tmpDict

def risc_analysis(name, analysis, deviceData):
    for device in deviceData.keys():
        for core in deviceData[device].keys():
            if core != "DEVICE":
                for risc in deviceData[device][core].keys():
                    if risc == analysis["start"]["risc"]:
                        timeseries = deviceData[device][core][risc]["timeseries"]
                        tmpDict = timeseries_analysis(timeseries,analysis)
                        if "analysis" not in deviceData[device][core][risc].keys():
                            deviceData[device][core][risc]["analysis"]={
                                name:tmpDict
                            }
                        else:
                            deviceData[device][core][risc]["analysis"][name] = tmpDict


def core_analysis(name, analysis, deviceData):
    for device in deviceData.keys():
        for core in deviceData[device].keys():
            if core != "DEVICE":
                risc = "TENSIX"
                assert(risc in deviceData[device][core].keys())
                timeseries = deviceData[device][core][risc]["timeseries"]
                tmpDict = timeseries_analysis(timeseries,analysis)
                if "analysis" not in deviceData[device][core][risc].keys():
                    deviceData[device][core][risc]["analysis"]={
                        name:tmpDict
                    }
                else:
                    deviceData[device][core][risc]["analysis"][name] = tmpDict

def device_analysis(name, analysis, deviceData):
    for device in deviceData.keys():
        core = "DEVICE"
        risc = "TENSIX"
        assert(core in deviceData[device].keys())
        assert(risc in deviceData[device][core].keys())
        timeseries = deviceData[device][core][risc]["timeseries"]
        tmpDict = timeseries_analysis(timeseries,analysis)
        if "analysis" not in deviceData[device][core][risc].keys():
            deviceData[device][core][risc]["analysis"]={
                name:tmpDict
            }
        else:
            deviceData[device][core][risc]["analysis"][name] = tmpDict

def generate_device_level_summary(deviceData):
    for device in deviceData.keys():
        analysisLists = {}
        for core in deviceData[device].keys():
            if core != "DEVICE":
                for risc in deviceData[device][core].keys():
                    for name, analysis in deviceData[device][core][risc]["analysis"].items():
                        if name in analysisLists.keys():
                            analysisLists[name]["statList"].append(analysis['stats'])
                        else:
                            analysisLists[name] = dict(analysis=analysis['analysis'], statList = [analysis['stats']])

        for name, analysisList in analysisLists.items():
            tmpDF = pd.DataFrame(analysisList["statList"])
            tmpDict = {}
            if not tmpDF.empty:
                tmpDict = {
                    "analysis": analysisList["analysis"],
                    "stats" : {
                        "Count" : tmpDF.loc[:,'Count'].sum(),
                        "Average" : tmpDF.loc[:,'Sum'].sum()/tmpDF.loc[:,'Count'].sum(),
                        "Max" : tmpDF.loc[:,'Max'].max(),
                        "Min" : tmpDF.loc[:,'Min'].min(),
                        "Median" : tmpDF.loc[:,'Median'].median(),
                    }
                }
            deviceData[device]['DEVICE']['TENSIX']['analysis'][name] = tmpDict



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
    risc_to_core_timeseries(deviceData)
    core_to_device_timeseries(deviceData)

    for name, analysis in sorted(setup.timerAnalysis.items()):
        if analysis["across"] == "risc":
            risc_analysis(name, analysis, deviceData)
        elif analysis["across"] == "core":
            core_analysis(name, analysis, deviceData)
        elif analysis["across"] == "device":
            device_analysis(name, analysis, deviceData)

    generate_device_level_summary(deviceData)

    print_stats(deviceData, setup)
    print_arranged_csv(deviceData, setup.timerIDLabels)

    timelineFigs = {}
    statTables = {}
    for chipID, deviceData in deviceData.items():
        timeseries_to_durations(deviceData)
        yVals = sorted(deviceData.keys(), key=coreCompare, reverse=True)
        yVals.remove('DEVICE')

        xValsDict = plotData_to_timelineXVals(deviceData, yVals, setup)
        key = f"Chip {chipID}: Cores"
        statTables[key] = generate_analysis_table(deviceData["DEVICE"]["TENSIX"]["analysis"],setup)
        timelineFigs[key] = timeline_plot(yVals, xValsDict, setup)

        xValsDict = plotData_to_timelineXVals(deviceData, ['DEVICE'], setup)
        key = f"Chip {chipID}: Device"
        timelineFigs[key] = timeline_plot(['DEVICE'], xValsDict, setup)

    external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    app = Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div(
        [
            html.H1("Device Performance"),
            html.Br(),
        ] +
        [
            html.Div(
                [
                    html.H5(f"{figure}"),
                    statTables[figure] if figure in statTables.keys() else html.Div([]),
                    dcc.Graph(figure=timelineFigs[figure])
                ]
            ) for figure in sorted(timelineFigs.keys())
        ]
    )

    app.run_server(host="0.0.0.0", debug=True)


if __name__ == "__main__":
    main(sys.argv[1:])
