import os
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
import asyncio
import aiofiles

# If we have the example requirements and test dictionary isolated:
TEST_SWEEP_TEMPLATE = ChatPromptTemplate.from_template(
    """Given this Python test file:\n
    ---------------------------------------------------\n
    {example_sweep_file}\n
    ----------------------------------------------------\n
    If these:\n
    {example_requirements}\n
    dictionaries coincides with these tests:\n
    {}"""
)

SWEEP_TEMPLATE = ChatPromptTemplate.from_template(
    """Given this Python test file:\n\n
    ---------------------------------------------------\n
    {example_sweep_file}\n
    ---------------------------------------------------\n
    If the specs within parameters coincide with these tests:\n
    {example_md_reqs}\n\n
    Return the edited parameters dictionary with a further populated test_specs list based on the previous example and the following list of tests:\n\n
    {next_md_reqs}"""
)

SWEEP_TEMPLATE_BATCH = ChatPromptTemplate.from_template(
    """Given this Python test file:\n\n
    ---------------------------------------------------\n
    {example_sweep_file}\n
    ---------------------------------------------------\n
    If the specs within parameters coincide with these tests:\n
    {example_md_reqs}\n\n
    Return the further populated test_specs list based on the previous example and the following list of tests:\n\n
    {next_md_reqs}\n\n
    RETURN ONLY THE test_specs PYTHON LIST OF DICTIONARIES"""
)


load_dotenv()
client = OpenAI(
    # Defaults to os.environ.get("OPENAI_API_KEY")
)


def extract_parameters(text: str) -> str:
    """
    Extract parameters dict from the chat completion as a string
    """
    start_idx = text.find("parameters = {")

    if start_idx == -1:
        print("no params found")
        return None  # Return None if 'parameters =' is not found

    stack = []
    end_idx = start_idx

    # start from "parameters = {"
    # Using parentheses leetcode lol
    for i in range(start_idx, len(text)):
        char = text[i]
        if char == "{":
            stack.append("{")
        elif char == "}":
            stack.pop()
            if not stack:
                end_idx = i + 1
                break

    return start_idx, end_idx, text[start_idx:end_idx] if stack == [] else None


def extract_suite_specs(text: str) -> str:
    """
    Extract parameters dict from the chat completion as a string
    """
    start_idx = text.find('"test_specs" = [')

    if start_idx == -1:
        print("no specs found")
        return None  # Return None if 'parameters =' is not found

    stack = []
    start_idx = text[start_idx:].find("[")
    end_idx = start_idx

    # start from "test_specs = ["
    # Using parentheses leetcode lol
    for i in range(start_idx, len(text)):
        char = text[i]
        if char == "[":
            stack.append("]")
        elif char == "[":
            stack.pop()
            if not stack:
                end_idx = i + 1
                break

    return start_idx, end_idx, text[start_idx:end_idx] if stack == [] else None


async def async_generate_sweeps(
    prompt: ChatPromptTemplate,
    example_operation_filepath: str,
    example_requirements_filepath: str,
    new_requirements_filepath: str,
) -> tuple:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        generate_sweep,
        prompt,
        example_operation_filepath,
        example_requirements_filepath,
        new_requirements_filepath,
    )


async def generate_sweep(
    prompt: ChatPromptTemplate,
    example_operation_filepath: str,
    example_requirements_filepath: str,
    new_requirements_filepath: str,
) -> str:
    """
    Generate sweep test parameters given an example operation containing sweep examples and their respective md requirements and the new complete md requirements
    """

    async with aiofiles.open(example_operation_filepath, "r") as input_file:
        file_content = await input_file.read()
    async with aiofiles.open(example_requirements_filepath, "r") as input_file:
        example_req = await input_file.read()
    async with aiofiles.open(new_requirements_filepath, "r") as input_file:
        new_req = await input_file.read()
    try:
        prompt_values = {"example_sweep_file": file_content, "example_md_reqs": example_req, "next_md_reqs": new_req}
        prompt_text = prompt.format_prompt(**prompt_values).to_string()

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt_text}],
            max_tokens=2048,
            temperature=1.0,
            n=1,
            stop=None,
        )
        # print(response)
        text = response.choices[0].message.content
        _, __, params = extract_parameters(text)
    except ValueError as e:
        print(f"An unexpected error ocurred: {e}")

    start, end, _ = extract_parameters(file_content)
    new_file = file_content[0:start] + params + file_content[end + 1 : len(file_content)]

    last_slash_index = example_operation_filepath.rfind("/")

    # Extract the directory path and file name
    directory = example_operation_filepath[: last_slash_index + 1]
    filename = example_operation_filepath[last_slash_index + 1 :]
    new_path = directory + "new_" + filename
    async with aiofiles.open(new_path, "w") as output_file:
        await output_file.write(new_file)


async def generate_sweep_list(
    prompt: ChatPromptTemplate,
    example_operation_filepath: str,
    example_requirements_filepath: str,
    new_requirements_filepath: str,
) -> str:
    """
    Generate sweep test parameters given an example operation containing sweep examples and their respective md requirements and the new complete md requirements
    """

    async with aiofiles.open(example_operation_filepath, "r") as input_file:
        file_content = await input_file.read()
    async with aiofiles.open(example_requirements_filepath, "r") as input_file:
        example_req = await input_file.read()
    async with aiofiles.open(new_requirements_filepath, "r") as input_file:
        new_req = await input_file.read()
    try:
        prompt_values = {"example_sweep_file": file_content, "example_md_reqs": example_req, "next_md_reqs": new_req}
        prompt_text = prompt.format_prompt(**prompt_values).to_string()

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt_text}],
            max_tokens=2048,
            temperature=1.0,
            n=1,
            stop=None,
        )
        # print(response)
        text = response.choices[0].message.content
        _, __, params = extract_parameters(text)
    except ValueError as e:
        print(f"An unexpected error ocurred: {e}")

    start, end, _ = extract_parameters(file_content)
    new_file = file_content[0:start] + params + file_content[end + 1 : len(file_content)]

    last_slash_index = example_operation_filepath.rfind("/")

    # Extract the directory path and file name
    directory = example_operation_filepath[: last_slash_index + 1]
    filename = example_operation_filepath[last_slash_index + 1 :]
    new_path = directory + "new_" + filename
    async with aiofiles.open(new_path, "w") as output_file:
        await output_file.write(new_file)


async def run_sweep_tasks(to_gen, prompt):
    tasks = []

    # Set up a progress bar
    pbar = tqdm(total=len(to_gen), desc="Generating Sweeps", leave=False)

    # Iterate over the dictionary and create tasks
    for key in to_gen:
        # Collect the necessary file paths for each sweep
        operations = to_gen[key]["operations"]
        trace = to_gen[key]["trace"]
        new_trace = to_gen[key]["new_trace"]

        # For each operation file, run generate_sweep in parallel
        for operation_file in operations:
            if trace and new_trace:
                # Create an async task for each generate_sweep call
                tasks.append(asyncio.create_task(generate_sweep(prompt, operation_file, trace, new_trace)))

    # Wait for all tasks to complete and update the progress bar
    for task in asyncio.as_completed(tasks):
        await task
        pbar.update(1)

    pbar.close()


if __name__ == "__main__":
    base_directory = "sweeps/data_movement/test_gen/"
    to_gen = {}
    # Walk through each directory and subdirectory
    for root, dirs, files in os.walk(base_directory):
        directory_name = os.path.basename(root)  # Current directory name
        if directory_name == "__pycache__" or directory_name == "":
            continue
        to_gen[directory_name] = {"operations": [], "new_trace": None, "trace": None}
        for file in files:
            file_path = os.path.join(root, file)

            # Categorize files based on their extension and name
            if file.endswith(".py"):
                to_gen[directory_name]["operations"].append(file_path)
            elif file.endswith(".md"):
                if file.startswith("new_"):
                    to_gen[directory_name]["new_trace"] = file_path
                else:
                    to_gen[directory_name]["trace"] = file_path

    print(to_gen.keys())
    for key in to_gen:
        print(key)
        print(to_gen[key])
    # asyncio.run(run_sweep_tasks(to_gen, SWEEP_TEMPLATE))
    asyncio.run(
        generate_sweep(
            SWEEP_TEMPLATE,
            "sweeps/data_movement/test_gen/test_squeeze/squeeze_pytorch2.py",
            "sweeps/data_movement/test_gen/test_squeeze/squeeze_trace.md",
            "sweeps/data_movement/test_gen/test_squeeze/new_squeeze_trace.md",
        )
    )
