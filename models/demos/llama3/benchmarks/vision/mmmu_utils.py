import re
import ast
import numpy as np
import random
from PIL import Image as PIL_Image

prompt_config_mmmu_pro = {
    "vision": "Write out the multiple-choice question in the image and then solve it. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering.",
    "standard": "Answer the preceding multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering.",
}

prompt_config_mmmu = {
    "multi_choice_example_format": "Answer with the option's letter from the given choices directly.",
    "short_ans_example_format": "Answer the question using a single word or phrase.",
}

# Category Mapping
CAT_SHORT2LONG = {
    "acc": "Accounting",
    "agri": "Agriculture",
    "arch": "Architecture_and_Engineering",
    "art": "Art",
    "art_theory": "Art_Theory",
    "bas_med": "Basic_Medical_Science",
    "bio": "Biology",
    "chem": "Chemistry",
    "cli_med": "Clinical_Medicine",
    "cs": "Computer_Science",
    "design": "Design",
    "diag_med": "Diagnostics_and_Laboratory_Medicine",
    "econ": "Economics",
    "elec": "Electronics",
    "ep": "Energy_and_Power",
    "fin": "Finance",
    "geo": "Geography",
    "his": "History",
    "liter": "Literature",
    "manage": "Manage",
    "mark": "Marketing",
    "mate": "Materials",
    "math": "Math",
    "mech": "Mechanical_Engineering",
    "music": "Music",
    "phar": "Pharmacy",
    "phys": "Physics",
    "psy": "Psychology",
    "pub_health": "Public_Health",
    "socio": "Sociology",
}


def replace_images_tokens(input_string):
    for i in range(1, 8):
        question_text = f"<image {i}>"
        query_text = "[image]"
        if question_text in input_string:
            input_string = input_string.replace(question_text, query_text)
    return input_string


def construct_prompt(doc):
    question = doc["question"]
    parsed_options = parse_options(ast.literal_eval(str(doc["options"])))
    question = f"{question}\n{parsed_options}\n{prompt_config_mmmu_pro['standard']}"
    return question


def mmmu_doc_to_text(doc):
    question = construct_prompt(doc)
    return replace_images_tokens(question)


def origin_mmmu_doc_to_visual(doc):
    visual = []
    for i in range(1, 8):
        if not doc[f"image_{i}"]:
            break
        visual.append(doc[f"image_{i}"])
    return visual


def vision_mmmu_doc_to_visual(doc):
    return [doc["image"]]


def process_prompt(data, setting):
    if setting == "standard":
        prompt = mmmu_doc_to_text(data)
        images = origin_mmmu_doc_to_visual(data)
    elif setting == "vision":
        prompt = prompt_config_mmmu_pro["vision"]
        images = vision_mmmu_doc_to_visual(data)

    return prompt, images


def process_prompt_mmmu(data):
    question = data["question"]
    if data["image"]:
        image = data["image"]
    else:
        image = None

    if data["question_type"] == "multiple-choice":
        question += (
            f"\n{parse_options(ast.literal_eval(data['options']))}\n{prompt_config_mmmu['multi_choice_example_format']}"
        )
    else:
        question += f"\n{prompt_config_mmmu['short_ans_example_format']}"
    return question, image


# Function to parse image paths
def parse_img_path(text):
    matches = re.findall("<img='(.*?)'>", text)
    return matches


# Function to process a single sample
def process_single_sample(data, args):
    question = data["question"]
    o_imgs_paths = []
    for option in data["options"]:
        current_o_imgs_paths = parse_img_path(option)
        for img_path in current_o_imgs_paths:
            o_imgs_paths.append(img_path)

    if len(o_imgs_paths) > 1:
        return {
            "id": data["id"],
            "question": question,
            "options": data["options"],
            "answer": data["answer"],
            "image": None,
            "question_type": data["question_type"],
        }
    else:
        return {
            "id": data["id"],
            "question": question,
            "options": data["options"],
            "answer": data["answer"],
            "image": data["image_1"],
            "question_type": data["question_type"],
        }


# Function to get multiple choice options
def get_multi_choice_info(options):
    options = ast.literal_eval(options)
    start_chr = "A"
    all_choices = []
    index2ans = {}
    for i, option in enumerate(options):
        index2ans[chr(ord(start_chr) + i)] = option
        all_choices.append(chr(ord(start_chr) + i))
    return index2ans, all_choices


# Function to parse the model's response for multiple choice
def parse_multi_choice_response(response, all_choices, index2ans):
    last_answer_pos = response.rfind("Answer:")
    if last_answer_pos != -1:
        answer_str = response[last_answer_pos + len("Answer:") :].strip()
        matching_options = [option for option in all_choices if option in answer_str]
        if len(matching_options) == 1:
            return matching_options[0]

    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "

    candidates = []
    ans_with_brack = False
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice} " in response:
                candidates.append(choice)
    if len(candidates) == 0:
        for choice in all_choices:
            if f"{choice}." in response:
                candidates.append(choice)
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
    if len(candidates) == 0:
        return random.choice(all_choices)
    if len(candidates) > 1:
        start_indexes = []
        for can in candidates:
            index = response.rfind(f"({can})" if ans_with_brack else f" {can} ")
            start_indexes.append(index)
        return candidates[np.argmax(start_indexes)]
    return candidates[0]


def eval_open(gold_i, pred_i):
    """
    Evaluate an open question instance
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L191
    """
    correct = False
    if isinstance(gold_i, list):
        # use float to avoid trivial matches
        norm_answers = []
        for answer in gold_i:
            norm_answers.extend(normalize_str(answer))
    else:
        norm_answers = normalize_str(gold_i)
    for pred in pred_i:  # pred is already normalized in parse response phase
        if isinstance(pred, str):  # if it's a string, then find if ans in the pred_i
            for norm_ans in norm_answers:
                # only see if the string answer in the string pred
                if isinstance(norm_ans, str) and norm_ans in pred:
                    if not correct:
                        correct = True
                    break
        else:  # it's a float number
            if pred in norm_answers:
                if not correct:
                    correct = True
                break
    return correct


def eval_multi_choice(gold_i, pred_i):
    """
    Evaluate a multiple choice instance.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L175
    """
    correct = False
    # only they are exactly the same, we consider it as correct
    if isinstance(gold_i, list):
        for answer in gold_i:
            if answer == pred_i:
                correct = True
                break
    else:  # gold_i is a string
        if gold_i == pred_i:
            correct = True
    return correct
