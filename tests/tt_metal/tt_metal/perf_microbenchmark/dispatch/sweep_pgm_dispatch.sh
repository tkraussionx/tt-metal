#/bin/bash

if [ "$ARCH_NAME" = "grayskull" ]; then
    echo "Configured core range for grayskull"
    max_x="11"
    max_y="8"
elif [ "$ARCH_NAME" = "wormhole_b0" ]; then
    echo "Configured core range for wormhole_b0"
    max_x="7"
    max_y="6"
elif [ "$ARCH_NAME" = "blackhole" ]; then
    echo "Configured core range for blackhole"
    max_x="12"
    max_y="9"
else
    echo "Unknown arch: $ARCH_NAME"
    exit
fi

# Initialize the string variable
trace_option=""

# Parse command line arguments
for arg in "$@"; do
    case $arg in
        --trace)
            trace_option="-tr"
            shift # Remove --trace from processing
            ;;
        *)
            # Handle other arguments if necessary
            ;;
    esac
done

# brisc only
echo "###" brisc only
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -n -t $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -n -t $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -n -t $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -n -t $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -n -t $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 -n -t $trace_option

# ncrisc only
echo "###" ncrisc only
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -b -t $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -b -t $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -b -t $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -b -t $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -b -t $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -b -t $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 -b -t $trace_option

#trisc only
echo "###" trisc only
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -b -n $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -b -n $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -b -n $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -b -n $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -b -n $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -b -n $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 -b -n $trace_option

#brisc+trisc only
echo "###" brisc+trisc only
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -n $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -n $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -n $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -n $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -n $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 -n $trace_option

#all processors
echo "###" all procesors
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 $trace_option

#all processors, all cores
echo "###" all procesors all cores
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -x $max_x -y $max_y $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -x $max_x -y $max_y $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -x $max_x -y $max_y $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -x $max_x -y $max_y $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x $max_x -y $max_y $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -x $max_x -y $max_y $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 10240 -x $max_x -y $max_y $trace_option

#all processors, all cores, 1 CB
echo "###" all procesors all cores 1cb
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -x $max_x -y $max_y -c 1 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -x $max_x -y $max_y -c 1 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -x $max_x -y $max_y -c 1 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -x $max_x -y $max_y -c 1 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x $max_x -y $max_y -c 1 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -x $max_x -y $max_y -c 1 $trace_option

#all processors, all cores, 32 CB
echo "###" all procesors all cores 32cb
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -x $max_x -y $max_y -c 32 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -x $max_x -y $max_y -c 32 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -x $max_x -y $max_y -c 32 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -x $max_x -y $max_y -c 32 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x $max_x -y $max_y -c 32 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -x $max_x -y $max_y -c 32 $trace_option

#all processors, 1 core, 1 rt arg
echo "###" all procesors 1 core 1 rta
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -a 1 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -a 1 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -a 1 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -a 1 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -a 1 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -a 1 $trace_option

#1 processors, alls core, 128 rt arg
echo "###" all procesors all cores 128 rta
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -x $max_x -y $max_y -a 128 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -n -t -x $max_x -y $max_y -a 128 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -n -t -x $max_x -y $max_y -a 128 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -n -t -x $max_x -y $max_y -a 128 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -n -t -x $max_x -y $max_y -a 128 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -n -t -x $max_x -y $max_y -a 128 $trace_option

#1 processors, alls core, 1 rt arg
echo "###" all procesors all cores 1 rta
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -x $max_x -y $max_y -a 1 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -n -t -x $max_x -y $max_y -a 1 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -n -t -x $max_x -y $max_y -a 1 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -n -t -x $max_x -y $max_y -a 1 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -n -t -x $max_x -y $max_y -a 1 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -n -t -x $max_x -y $max_y -a 1 $trace_option

#all processors, alls core, 1 rt arg
echo "###" all procesors all cores 1 rta
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -x $max_x -y $max_y -a 1 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -x $max_x -y $max_y -a 1 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -x $max_x -y $max_y -a 1 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -x $max_x -y $max_y -a 1 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x $max_x -y $max_y -a 1 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -x $max_x -y $max_y -a 1 $trace_option

#all processors, all cores, 32 args
echo "###" all procesors all cores 32 rta
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -x $max_x -y $max_y -a 32 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -x $max_x -y $max_y -a 32 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -x $max_x -y $max_y -a 32 $trace_optionv
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -x $max_x -y $max_y -a 32 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x $max_x -y $max_y -a 32 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -x $max_x -y $max_y -a 32 $trace_option

#all processors, all cores, 128 args
echo "###" all procesors all cores 128 rta
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -x $max_x -y $max_y -a 128 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -x $max_x -y $max_y -a 128 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -x $max_x -y $max_y -a 128 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -x $max_x -y $max_y -a 128 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x $max_x -y $max_y -a 128 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -x $max_x -y $max_y -a 128 $trace_option

# sems
echo "###" sems 1 core 1 processor
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -S 4 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -n -t -S 4 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -n -t -S 4 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -n -t -S 4 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -n -t -S 4 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -n -t -S 4 $trace_option

echo "###" sems all cores 1 processors
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -n -t -x $max_x -y $max_y -S 4 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -n -t -x $max_x -y $max_y -S 4 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -n -t -x $max_x -y $max_y -S 4 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -n -t -x $max_x -y $max_y -S 4 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -n -t -x $max_x -y $max_y -S 4 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -n -t -x $max_x -y $max_y -S 4 $trace_option

# Worst case
echo "###" worst case
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -x $max_x -y $max_y -S 4 -c 32 -a 128 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -x $max_x -y $max_y -S 4 -c 32 -a 128 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -x $max_x -y $max_y -S 4 -c 32 -a 128 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -x $max_x -y $max_y -S 4 -c 32 -a 128 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x $max_x -y $max_y -S 4 -c 32 -a 128 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -x $max_x -y $max_y -S 4 -c 32 -a 128 $trace_option

# Kernel groups (perhaps even worse)
echo "###" kernel groups
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -x $max_x -y $max_y -kg $max_x $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -x $max_x -y $max_y -kg $max_x $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -x $max_x -y $max_y -kg $max_x $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -x $max_x -y $max_y -kg $max_x $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x $max_x -y $max_y -kg $max_x $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -x $max_x -y $max_y -kg $max_x $trace_option

# Same as above, but w/ 1 slow kernel and 4 fast "shadow kernels" (test worker RB queuing)
echo "###" kernel groups w/ shadow kernels
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -x $max_x -y $max_y -kg $max_x -rs 40000 -nf 4 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -x $max_x -y $max_y -kg $max_x -rs 40000 -nf 4 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -x $max_x -y $max_y -kg $max_x -rs 40000 -nf 4 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -x $max_x -y $max_y -kg $max_x -rs 40000 -nf 4 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x $max_x -y $max_y -kg $max_x -rs 40000 -nf 4 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 8192 -x $max_x -y $max_y -kg $max_x -rs 40000 -nf 4 $trace_option

if [ "$ARCH_NAME" = "wormhole_b0" ]; then
# Dispatch to eth
echo "###" eth dispatch
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -b -n -t +e $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 512 -b -n -t +e $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 1024 -b -n -t +e $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 2048 -b -n -t +e $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -b -n -t +e $trace_option

# The new new new worst case (kernel groups+rtas+eth)
# This column is incomplete and isn't just incrementing sizes...
echo "###" tensix+eth dispatch for 2, then long running kernel
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 256 -x $max_x -y $max_y -kg $max_x +e -ec 2 -a 16 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x $max_x -y $max_y -kg $max_x +e -ec 2 -a 16 $trace_option
build/test/tt_metal/perf_microbenchmark/dispatch/test_pgm_dispatch -w 5000 -s 4096 -x $max_x -y $max_y -kg $max_x +e -ec 2 -a 16 -rs 40000 -nf 4 $trace_option
fi
