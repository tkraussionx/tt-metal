#!/bin/bash
mode="performance"
cpus=`nproc`
for i in $(seq 0 $((cpus-1)))
do
    echo $mode | sudo tee  /sys/devices/system/cpu/cpu$i/cpufreq/scaling_governor
done
