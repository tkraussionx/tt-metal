#/usr/bin/env bash
var=0
fail=0
rm -rf test.log
rm -rf test-out.log
while true
do
    ((var++))
    sleep 1
    echo
    echo THIS IS LOOP COUNT : $var >> test.log
    echo THIS IS LOOP COUNT : $var
    #pytest models/demos/resnet/tests/test_perf_device_resnet.py::test_perf_device_bare_metal[20-LoFi-activations_BFLOAT8_B-weights_BFLOAT8_B-batch_20-7170] >> test-out.log
    #./tests/scripts/run_profiler_regressions.sh PROFILER

    pytest models/demos/metal_BERT_large_11/tests/test_perf_device_bert.py &>> test-out.log

    if [ $? -eq 0 ]
    then
        echo "  $var: PASSED" >> test.log
    else
        ((fail++))
        echo "  $var: FAILED:$fail" >> test.log
    fi

done
