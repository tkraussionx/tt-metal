# TTNN Visualizer

## Contents
- [1. Visualizer Application Setup](#1-visualizer-application-setup)
- [2. How to use Visualizer](#2-how-to-use-visualizer)
- [3. Visualizer Results](#3-visualizer-results)

## 1. Visualizer Application Setup
To setup the Visualizer Application, follow the following steps:
1. Download the _ttnn_visualizer-0.3.1-py3-none-any.whl_ file from the [releases](https://github.com/tenstorrent/ttnn-visualizer/releases/tag/v0.3.1) page.
2. Install Python and Pip in local computer
   - Install python 3.12 to install visualizer
   - Create a virtual environment and activate the virtual environment using the following commands:
   ```
    python3 -m venv virtual_env
    source virtual_env/bin/activate
   ```
3. Run the command to install .whl file using pip: `pip3 install ttnn_visualizer-0.3.1-py3-none-any.whl`

## 2. How to use Visualizer
- Once the .whl file is installed, Go to the "virtual_env" folder created in the Finder.
- Move to the bin folder and check if the ttnn-visualizer (Unix executable file) is present.
- From the terminal, move to the bin directory and run the command: `./ttnn-visualizer`
- Copy the local host link (similar to http://0.0.0.0:8000) and open in Chrome.
- Once the visualizer application is opened, get the required reports to upload.
- To get the required reports to upload, run the export commands and test the file as given in the [documentation](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/usage.html#visualize-using-web-browser)
- Once the test is ran, the reports folder will be generated in the repository. The folder contains the sql and json files, looks similar to `tt-metal/generated/ttnn/reports/7018730453948114104`.
- Use scp command to get the generated report folder to the local laptop orq download the report folder to local laptop.
- Upload the generated report folder to 'Local Folder' of the Visualizer application and click 'View Report'.

## 3. Visualizer Results
- On clicking the 'View Report', all the list of Operations of the test are shown.
- Execution time and Arguments of the operation are detailed in the drop down of each operation
- Switching to 'Tensors' on top right corner of the applications, all the tensors of the test are shown.
- Details of each tensor are specified in the drop down of respective Tensor.
