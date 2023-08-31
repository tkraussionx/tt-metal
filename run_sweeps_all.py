import os
from glob import glob
import subprocess

directory = 'tests/tt_eager/python_api_testing/sweep_tests/test_configs/ci_sweep_tests'
result_folder = "/home/ubuntu/tt-metal/ng-test-sweeps/"

if __name__ == "__main__":
    txt_files = glob(os.path.join(directory, '*.yaml'))
    txt_files.sort()
    do_run = False

    for txt_file in txt_files:
        basename = os.path.splitext(os.path.basename(txt_file))[0]
        command = f'./tt_metal/tools/profiler/profile_this.py -c "python tests/tt_eager/python_api_testing/sweep_tests/run_pytorch_test.py -i {txt_file} -o {result_folder}{basename}" -o {result_folder}{basename}'

        if basename == "pytorch_transpose_wh_test":
            do_run = True

        if do_run:
            print(command)

            subprocess.run(["rm -rf tt_metal/tools/profiler/logs/ops_device"], shell=True, check=True)
            subprocess.run(["rm -rf tt_metal/tools/profiler/logs/ops"], shell=True, check=True)
            subprocess.run([command], shell=True, check=True)
