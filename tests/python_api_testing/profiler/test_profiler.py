import os
import time

from libs import tt_lib as ttl

# def test_back_to_back_folder():
    # testLocation ="tt_metal/tools/profiler/logs/test_profiler/"
    # os.system(f"rm -rf {testLocation}")
    # ttl.profiler.set_profiler_flag(True)
    # ttl.profiler.set_profiler_location(testLocation)

    # ttl.profiler.set_op_name("first_call")
    # ttl.profiler.start_profiling()
    # time.sleep(1)
    # ttl.profiler.stop_profiling()

    # # try:
    # ttl.profiler.set_op_name("second_call")
    # ttl.profiler.start_profiling()
    # time.sleep(1)
    # ttl.profiler.stop_profiling()
    # # except Exception as e:
        # # print (e)

def one_second_profile (name):
    ttl.profiler.start_profiling(name)

    ttl.profiler.append_meta_data(f"{name}_meta_1")
    ttl.profiler.append_meta_data(f"{name}_meta_2")

    time.sleep(1)

    ttl.profiler.start_profiling(f"{name[:-1]}2")
    ttl.profiler.append_meta_data(f"{name[:-1]}2_meta_1")
    time.sleep(1)
    ttl.profiler.append_meta_data(f"{name[:-1]}2_meta_2")
    ttl.profiler.stop_profiling(f"{name[:-1]}2")

    time.sleep(2)
    ttl.profiler.append_meta_data(f"{name}_meta_3")
    ttl.profiler.stop_profiling(name)

def test_python_only():
    pass

def test_host_CPP_only():
    pass

def test_device():
    pass

def test_meta_data_with_dash():
    pass

def test_make_no_artifact_is_generated_on_no_profiler():
    pass

def test_multiple_additional_data_scenarios():
    pass

def test_nested_profiling():
    testLocation ="tt_metal/tools/profiler/logs/test_profiler/nested_profiling"
    os.system(f"rm -rf {testLocation}")
    ttl.profiler.set_profiler_flag(True)
    ttl.profiler.set_profiler_location(testLocation)

    one_second_profile ("first_level_1")
    time.sleep(0.1)
    one_second_profile ("first_level_1")
    time.sleep(0.1)
    one_second_profile ("second_level_1")
