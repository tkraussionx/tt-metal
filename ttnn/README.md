## TTNN [ttnn]

### A Python library built using the apis from tt_lib to be used in conjuction with PyTorch.  The intent of this effort is to provide a successful onboarding experience for new users.

#### Key differences with the following operations:
* matmul
    * The last two dimensions must be a multiple of 32 if the multiplication is to work on the device.  For example a tensor of (1, 1, 3, 4) would not be successfully multiplied to another tensor.  Instead, the developer is currently expected to adjust the tensor to be viable in a device multiplication.
    * Results from a matmul will not have the exact same precision. Instead of pytorch allclose, we used a pearson correlation coefficient to verify the results.
* reshape
    * The last two dimensions in the reshape must be a multiple of 32 when using the tensor on a device.
    * When converting a tt tensor to a pytorch tensor, the 3rd and 4th dimensions must be a multiple of 32.
    * Using the single argument -1 in reshape is not supported.
* transpose
    * There is no transpose method.  Please use permute instead.
* permute
    * When using the ttnn library, the operation requires the first parameter to to be the tensor and the next to be the new order of dimensions within a parenthesis.

#### Frequently asked questions
* What if my device hangs?
    * Try resetting the board on the command line with: `tt-smi -tr all`
* Is slicing available?
    * Slicing is supported.  At the moment this feature falls back to using pytorch slicing.
    * Example:
        * tensor1 = ttnn.from_torch(torch.randn(3,3))
        * print(tensor1[:1])
* Why are the results from operations like add and matmul not the same precision and require a pearson correlation coefficient comparison?
    * Results for operations are different because the order of floating point operations is different between CPU and the TT device.  A similiar issue would arise when comparing cpu and gpu operations.
* How do I create a tensor of all zeros that is not on device and the height and width do not have to be multiples of 32?
    * Use pytorch to achieve this.
        * tensor = ttnn.from_torch(torch.zeros(3,3))
        * print(tensor)
* How do I dump the logs of operations from the TT device?
    * You can add one or both of these environment variables
        *   `export TT_METAL_LOGGER_TYPES=Op`
        *   `export TT_METAL_LOGGER_LEVEL=DEBUG`
    * For the location of the operations use the following environment variable
        * `export OPERATION_HISTORY_CSV=<filename>`

#### How to debug from python and C++ at the same time within vscode
* `export CONFIG=debug` within your virtual environment and run `make build`
* Allow your process to attach to the debugger.
    * `echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope`
    * When done you can reverse this with... `echo 1 | sudo tee /proc/sys/kernel/yama/ptrace_scope`
* Add a task in launch.json for debugging your python code.  For example, you might have something like this below...
    ```
    {
      "name": "Python: test_matmul",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/build/python_env/bin/pytest",
      "args": [
        "${workspaceFolder}/tests/ttnn/test_matmul.py",
        "-k",
        "test_matmul"
      ],
      "justMyCode": false,
      "stopOnEntry": false,
      "console": "integratedTerminal",
      "cwd": "${workspaceFolder}",
      "internalConsoleOptions": "openOnSessionStart",
      "env": {
        "PYTHONPATH": "${workspaceFolder}"
      }
    },
    ```
* Additionally, add the following to launch.json (note that the name "C++: Attach to Python" will be used to find this task in the next step)
    ```
    {
      "name": "C++: Attach to Python",
      "type": "cppdbg",
      "request": "attach",
      "program": "${workspaceFolder}/build/python_env/bin/python3",
      "MIMode": "gdb",
      "miDebuggerPath": "gdb",
      "processId": "296907"
    },
    ```
* Wherever you want to debug the python code, add the function call update_process_id() from utils_for_testing.py
    * When you run your python test in the vscode debugger (for example, launching the test_matmul above), this method will update the "processId" of the task "C++: Attach to Python" in the .vscode/launch.json
    * It will then wait for you to hit enter.  Look for this prompt in the terminal in vscode.
    * Before hitting enter, run the "C++: Attach to Python" which will now have the processId updated that was being used by the debugger.
    * Make sure you have a breakpoint in the C++ code that you want to debug
    * Finally, hit enter in the terminal to continue debugging your C++ calls.
    * Note, the function update_process_id() is defined in utils_for_testing.py as...
        ```
        def update_process_id():
            print(f"Debugging PID: {os.getpid()}")
            cwd = os.getcwd()
            launch_json_path = os.path.join(cwd, ".vscode", "launch.json")
            with open(launch_json_path, "r") as f:
                launch_data = json.load(f)

            for config in launch_data.get("configurations", []):
                if config.get("name") == "C++: Attach to Python":
                    config["processId"] = str(os.getpid())
                    break

            with open(launch_json_path, "w") as f:
                json.dump(launch_data, f, indent=4)

            input("Press Enter to continue once the debugger is attached...")

        ```
