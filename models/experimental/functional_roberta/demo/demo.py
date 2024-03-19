# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from datasets import load_dataset
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import pytest
import torch
from loguru import logger
import tt_lib
import transformers
import ttnn
import evaluate
from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
    profiler,
)
from models.demos.bert.tt import ttnn_bert
from models.demos.bert.tt import ttnn_optimized_bert

from models.datasets.dataset_squadv2 import squadv2_1K_samples_input, squadv2_answer_decode_batch
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
)

from ttnn.model_preprocessing import *
from transformers import (
    RobertaForQuestionAnswering,
    pipeline,
    RobertaTokenizer,
    AutoTokenizer,
    RobertaForTokenClassification,
)

import evaluate


def load_inputs(input_path, batch):
    with open(input_path) as f:
        input_data = json.load(f)
        assert len(input_data) >= batch, f"Input data needs to have at least {batch} (batch size) entries."

        context = []
        question = []
        for i in range(batch):
            context.append(input_data[i]["context"])
            question.append(input_data[i]["question"])

        return context, question


def load_sentences(filepath):
    final = []
    sentences = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            if line == ("-DOCSTART- -X- -X- O\n") or line == "\n":
                if len(sentences) > 0:
                    final.append(sentences)
                    sentences = []
            else:
                l = line.split(" ")
                sentences.append((l[0], l[3].strip("\n")))
    return final


def load_sentences2(dict):
    sentences_list = []
    labels_list = []
    for d in dict:
        sentence = d["tokens"]
        labels = d["ner_tags"]
        sentences_list.append(sentence)
        labels_list.append(labels)
    return sentences_list, labels_list


def remove_label_prefix(label):
    # Remove 'B-' or 'I-' prefixes from the label
    if label.startswith("B-") or label.startswith("I-"):
        return label[2:]
    else:
        return label


def pre_process_raw_token_data(sample):
    prompt_list = []
    labels_list = []
    for sentence in sample:
        label = []
        prompt = ""
        for pair in sentence:
            l = remove_label_prefix(pair[1])
            label.append(l)
            prompt = prompt + " " + pair[0]
        prompt_list.append(prompt)
        labels_list.append(label)
    return prompt_list, labels_list


def pre_process_raw_token_data2(sen_lst, lbl_list):
    prompt_list = []
    labels_list = []
    for sen, lbl in zip(sen_lst, lbl_list):
        label = []
        prompt = ""
        for word, l in zip(sen, lbl):
            roberta_label = remove_label_prefix(l)
            label.append(roberta_label)
            prompt = prompt + " " + word
        prompt = prompt[1:]
        prompt_list.append(prompt)
        labels_list.append(label)
    return prompt_list, labels_list


def run_roberta_question_and_answering_inference(
    device,
    use_program_cache,
    model_name,
    batch_size,
    sequence_size,
    bert,
    input_path,
):
    disable_persistent_kernel_cache()

    hugging_face_reference_model = RobertaForQuestionAnswering.from_pretrained(model_name)
    hugging_face_reference_model.eval()

    # set up tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    config = hugging_face_reference_model.config
    nlp = pipeline("question-answering", model=hugging_face_reference_model, tokenizer=tokenizer)

    if bert == ttnn_bert:
        tt_model_name = f"ttnn_{model_name}"
    elif bert == ttnn_optimized_bert:
        tt_model_name = f"ttnn_{model_name}_optimized"
    else:
        raise ValueError(f"Unknown bert: {bert}")

    profiler.start(f"preprocessing_parameter")
    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: transformers.RobertaForQuestionAnswering.from_pretrained(
            model_name, torchscript=False
        ).eval(),
        custom_preprocessor=bert.custom_preprocessor,
        device=device,
    )
    profiler.end(f"preprocessing_parameter")

    context, question = load_inputs(input_path, batch_size)

    preprocess_params, _, postprocess_params = nlp._sanitize_parameters()
    preprocess_params["max_seq_len"] = sequence_size
    inputs = nlp._args_parser({"context": context, "question": question})
    preprocessed_inputs = []
    for i in range(batch_size):
        model_input = next(nlp.preprocess(inputs[0][i], **preprocess_params))
        single_input = {
            "example": model_input["example"],
            "inputs": model_input,
        }
        preprocessed_inputs.append(single_input)

    roberta_input = tokenizer.batch_encode_plus(
        zip(question, context),
        max_length=sequence_size,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="pt",
    )

    profiler.start(f"preprocessing_input")

    ttnn_roberta_inputs = bert.preprocess_inputs(
        roberta_input["input_ids"],
        roberta_input["token_type_ids"],
        roberta_input["attention_mask"],
        torch.zeros(1, sequence_size) if bert == ttnn_optimized_bert else None,
        device=device,
    )
    profiler.end(f"preprocessing_input")

    profiler.start(f"inference_time")
    tt_output = bert.bert_for_question_answering(
        config,
        *ttnn_roberta_inputs,
        parameters=parameters,
        name="roberta",
    )
    profiler.end(f"inference_time")

    tt_output = ttnn.to_torch(ttnn.from_device(tt_output)).reshape(batch_size, 1, sequence_size, -1).to(torch.float32)

    tt_start_logits = tt_output[..., :, 0].squeeze(1)
    tt_end_logits = tt_output[..., :, 1].squeeze(1)

    model_answers = {}
    profiler.start("post_processing_output_to_string")
    for i in range(batch_size):
        tt_res = {
            "start": tt_start_logits[i],
            "end": tt_end_logits[i],
            "example": preprocessed_inputs[i]["example"],
            **preprocessed_inputs[i]["inputs"],
        }

        tt_answer = nlp.postprocess([tt_res], **postprocess_params)

        logger.info(f"answer: {tt_answer['answer']}\n")
        model_answers[i] = tt_answer["answer"]

    profiler.end("post_processing_output_to_string")

    measurements = {
        "preprocessing_parameter": profiler.get("preprocessing_parameter"),
        "preprocessing_input": profiler.get("preprocessing_input"),
        "inference_time": profiler.get("inference_time"),
        "post_processing": profiler.get("post_processing_output_to_string"),
    }
    logger.info(f"preprocessing_parameter: {measurements['preprocessing_parameter']} s")
    logger.info(f"preprocessing_input: {measurements['preprocessing_input']} s")
    logger.info(f"inference_time: {measurements['inference_time']} s")
    logger.info(f"post_processing : {measurements['post_processing']} s")

    return measurements


def run_roberta_question_and_answering_inference_squad_v2(
    device,
    use_program_cache,
    model_name,
    batch_size,
    sequence_size,
    bert,
    n_iterations,
):
    disable_persistent_kernel_cache()

    hugging_face_reference_model = RobertaForQuestionAnswering.from_pretrained(model_name)
    hugging_face_reference_model.eval()

    # set up tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    config = hugging_face_reference_model.config

    if bert == ttnn_bert:
        tt_model_name = f"ttnn_{model_name}"
    elif bert == ttnn_optimized_bert:
        tt_model_name = f"ttnn_{model_name}_optimized"
    else:
        raise ValueError(f"Unknown bert: {bert}")

    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: transformers.RobertaForQuestionAnswering.from_pretrained(
            model_name, torchscript=False
        ).eval(),
        custom_preprocessor=bert.custom_preprocessor,
        device=device,
    )

    nlp = pipeline("question-answering", model=hugging_face_reference_model, tokenizer=tokenizer)

    attention_mask = True
    token_type_ids = True
    inputs_squadv2 = squadv2_1K_samples_input(tokenizer, sequence_size, attention_mask, token_type_ids, batch_size)
    squad_metric = evaluate.load("squad_v2")

    with torch.no_grad():
        pred_labels = []
        cpu_pred_labels = []
        true_labels = []
        i = 0
        for batch in inputs_squadv2:
            if i < n_iterations:
                batch_data = batch[0]
                curr_batch_size = batch_data["input_ids"].shape[0]
                ttnn_roberta_inputs = bert.preprocess_inputs(
                    batch_data["input_ids"],
                    batch_data["token_type_ids"],
                    batch_data["attention_mask"],
                    torch.zeros(1, sequence_size) if bert == ttnn_optimized_bert else None,
                    device=device,
                )

                tt_output = bert.bert_for_question_answering(
                    config,
                    *ttnn_roberta_inputs,
                    parameters=parameters,
                    name="roberta",
                )
                tt_output = (
                    ttnn.to_torch(ttnn.from_device(tt_output))
                    .reshape(batch_size, 1, sequence_size, -1)
                    .to(torch.float32)
                )
                cpu_output = hugging_face_reference_model(**batch_data)
                references = batch[1]
                question = batch[2]
                context = batch[3]

                cpu_predictions, tt_predictions = squadv2_answer_decode_batch(
                    hugging_face_reference_model,
                    tokenizer,
                    nlp,
                    references,
                    cpu_output,
                    tt_output,
                    curr_batch_size,
                    question,
                    context,
                )
                pred_labels.extend(tt_predictions)
                cpu_pred_labels.extend(cpu_predictions)
                true_labels.extend(references)

                del tt_output
            i += 1
        eval_score = squad_metric.compute(predictions=pred_labels, references=true_labels)
        cpu_eval_score = squad_metric.compute(predictions=cpu_pred_labels, references=true_labels)
        logger.info(f"\tTT_Eval: exact: {eval_score['exact']} --  F1: {eval_score['f1']}")


def calculate_metrics(pred_ttnn, ref):
    predicted = pred_ttnn  # [['LOC', 'O'], ['LOC', 'O', 'PER']]
    reference = ref  # [['LOC', 'O'], ['LOC', 'O', 'MISC']]

    # Initialize dictionaries to store TP, FP, FN for each entity type
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    # Iterate over each entity type
    for pred_sent, ref_sent in zip(predicted, reference):
        for pred_label, ref_label in zip(pred_sent, ref_sent):
            if pred_label == ref_label:
                tp[pred_label] += 1
            elif pred_label != "O":
                if ref_label != "O":
                    fn[ref_label] += 1
                fp[pred_label] += 1

    # Initialize lists to store precision, recall, and f1-score for each entity type
    entity_types = ["PER", "ORG", "LOC", "MISC"]
    precision = []
    recall = []
    f1 = []

    # Calculate precision, recall, and f1-score for each entity type
    for entity in entity_types:
        precision_entity = tp[entity] / (tp[entity] + fp[entity]) if (tp[entity] + fp[entity]) > 0 else 0
        recall_entity = tp[entity] / (tp[entity] + fn[entity]) if (tp[entity] + fn[entity]) > 0 else 0
        f1_entity = (
            2 * (precision_entity * recall_entity) / (precision_entity + recall_entity)
            if (precision_entity + recall_entity) > 0
            else 0
        )

        precision.append(precision_entity)
        recall.append(recall_entity)
        f1.append(f1_entity)

    # Calculate overall precision, recall, and f1-score
    overall_precision = sum(precision) / len(precision)
    overall_recall = sum(recall) / len(recall)
    overall_f1 = sum(f1) / len(f1)

    # Print precision, recall, and f1-score for each entity type
    print("Entity\tPrecision\tRecall\tF1")
    for i, entity in enumerate(entity_types):
        print(f"{entity}\t{precision[i]:.4f}\t{recall[i]:.4f}\t{f1[i]:.4f}")
    print(f"Overall\t{overall_precision:.4f}\t{overall_recall:.4f}\t{overall_f1:.4f}")


def run_roberta_token_inference(
    device,
    use_program_cache,
    model_name,
    batch_size,
    sequence_size,
    bert,
    input_path,
):
    disable_persistent_kernel_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hugging_face_reference_model = RobertaForTokenClassification.from_pretrained(model_name)
    hugging_face_reference_model.eval()
    config = hugging_face_reference_model.config

    if bert == ttnn_bert:
        tt_model_name = f"ttnn_{model_name}"
    elif bert == ttnn_optimized_bert:
        tt_model_name = f"ttnn_{model_name}_optimized"
    else:
        raise ValueError(f"Unknown bert: {bert}")

    profiler.start(f"preprocessing_parameter")
    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: transformers.RobertaForTokenClassification.from_pretrained(
            model_name, torchscript=False
        ).eval(),
        custom_preprocessor=bert.custom_preprocessor,
        device=device,
    )

    tc_input_path = "models/experimental/functional_roberta/demo/conll2003_validation.txt"
    validate_samples = load_sentences(tc_input_path)
    prompt_list, labels_list = pre_process_raw_token_data(validate_samples)
    inputs = prompt_list[0:batch_size]
    labels_list = labels_list[0:batch_size]

    roberta_input = tokenizer.batch_encode_plus(
        inputs,
        max_length=sequence_size,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors="pt",
    )

    # Get the starting indices of word from tokens list
    word_index_list = []
    for t in roberta_input["input_ids"]:
        tokens = tokenizer.convert_ids_to_tokens(t.squeeze())
        positions = []
        for i, token in enumerate(tokens):
            if token.startswith("Ġ"):
                positions.append(i)
        word_index_list.append(positions)
    word_index_list

    torch_output = hugging_face_reference_model(**roberta_input).logits

    # Save torch predictions for evaluating metrics
    torch_predictions = []
    for output, word_ind in zip(torch_output, word_index_list):
        output = output.squeeze(0)
        predicted_token_class_ids = output.argmax(dim=-1)
        predicted_tokens_classes = [config.id2label[t.item()] for t in predicted_token_class_ids]
        sel_pred_tok = []
        for i in word_ind:
            sel_pred_tok.append(predicted_tokens_classes[i])
        logger.info(f"Torch Predicted:{sel_pred_tok}")
        torch_predictions.append(sel_pred_tok)

    profiler.start(f"preprocessing_input")

    ttnn_roberta_inputs = bert.preprocess_inputs(
        roberta_input["input_ids"],
        roberta_input["token_type_ids"],
        roberta_input["attention_mask"],
        torch.zeros(1, sequence_size) if bert == ttnn_optimized_bert else None,
        device=device,
    )
    profiler.end(f"preprocessing_input")

    profiler.start(f"inference_time")
    tt_output = bert.bert_for_token_classification(
        config,
        *ttnn_roberta_inputs,
        parameters=parameters,
        name="roberta",
    )
    profiler.end(f"inference_time")

    tt_output = ttnn.to_torch(ttnn.from_device(tt_output))

    # Store tt predictions for evaluating metrics
    tt_predictions = []
    for output, word_ind in zip(tt_output, word_index_list):
        output = output.squeeze(0)
        predicted_token_class_ids = output.argmax(dim=-1)
        predicted_tokens_classes = [config.id2label[t.item()] for t in predicted_token_class_ids]
        sel_pred_tok = []
        for i in word_ind:
            sel_pred_tok.append(predicted_tokens_classes[i])
        logger.info(f"Tt Predicted {sel_pred_tok}")
        tt_predictions.append(sel_pred_tok)

    print("\n\n torch pipeline results:")
    calculate_metrics(torch_predictions, labels_list)

    print("\n\n ttnn pipeline results:")
    calculate_metrics(tt_predictions, labels_list)

    return tt_predictions


def run_roberta_token_inference_conll2003(
    device,
    use_program_cache,
    model_name,
    batch_size,
    sequence_size,
    bert,
    n_iterations,
):
    disable_persistent_kernel_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hugging_face_reference_model = RobertaForTokenClassification.from_pretrained(model_name)
    hugging_face_reference_model.eval()
    config = hugging_face_reference_model.config

    if bert == ttnn_bert:
        tt_model_name = f"ttnn_{model_name}"
    elif bert == ttnn_optimized_bert:
        tt_model_name = f"ttnn_{model_name}_optimized"
    else:
        raise ValueError(f"Unknown bert: {bert}")

    profiler.start(f"preprocessing_parameter")
    parameters = preprocess_model_parameters(
        model_name=tt_model_name,
        initialize_model=lambda: transformers.RobertaForTokenClassification.from_pretrained(
            model_name, torchscript=False
        ).eval(),
        custom_preprocessor=bert.custom_preprocessor,
        device=device,
    )

    # Load and preprocess dataset
    dataset = load_dataset("conll2003")
    id_to_label = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-ORG": 3,
        "I-ORG": 4,
        "B-LOC": 5,
        "I-LOC": 6,
        "B-MISC": 7,
        "I-MISC": 8,
    }
    sentence_list, ids_list = load_sentences2(dataset["validation"])
    labels_list = []
    for ids_sublist in ids_list:
        labels_sublist = [list(id_to_label.keys())[list(id_to_label.values()).index(id)] for id in ids_sublist]
        labels_list.append(labels_sublist)
    prompt_list, labels_list = pre_process_raw_token_data2(sentence_list, labels_list)

    # Slice the input as per n_iterations
    inputs = prompt_list[0 : n_iterations * batch_size]
    labels_list = labels_list[0 : n_iterations * batch_size]

    with torch.no_grad():
        cpu_predictions = []
        tt_predictions = []
        for i in range(n_iterations):
            batch_input = inputs[i * batch_size : (i + 1) * batch_size]
            roberta_input = tokenizer.batch_encode_plus(
                batch_input,
                max_length=sequence_size,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors="pt",
            )

            # Get the starting indices of word from tokens list
            word_index_list = []
            for t in roberta_input["input_ids"]:
                tokens = tokenizer.convert_ids_to_tokens(t.squeeze())
                positions = []
                for i, token in enumerate(tokens):
                    if token.startswith("Ġ"):
                        positions.append(i)
                word_index_list.append(positions)
            word_index_list

            # run torch model
            torch_output = hugging_face_reference_model(**roberta_input).logits
            torch_iter_predictions = []
            for output, word_ind in zip(torch_output, word_index_list):
                output = output.squeeze(0)
                predicted_token_class_ids = output.argmax(dim=-1)
                predicted_tokens_classes = [config.id2label[t.item()] for t in predicted_token_class_ids]
                sel_pred_tok = []
                for i in word_ind:
                    sel_pred_tok.append(predicted_tokens_classes[i])
                logger.info(f"Torch Predicted:{sel_pred_tok}")
                torch_iter_predictions.append(sel_pred_tok)

            # Prepare inputs for ttnn
            ttnn_roberta_inputs = bert.preprocess_inputs(
                roberta_input["input_ids"],
                roberta_input["token_type_ids"],
                roberta_input["attention_mask"],
                torch.zeros(1, sequence_size) if bert == ttnn_optimized_bert else None,
                device=device,
            )

            # run the tt model
            tt_output = bert.bert_for_token_classification(
                config,
                *ttnn_roberta_inputs,
                parameters=parameters,
                name="roberta",
            )

            tt_output = ttnn.to_torch(ttnn.from_device(tt_output))

            tt_iter_predictions = []
            for output, word_ind in zip(tt_output, word_index_list):
                output = output.squeeze(0)
                predicted_token_class_ids = output.argmax(dim=-1)
                predicted_tokens_classes = [config.id2label[t.item()] for t in predicted_token_class_ids]
                sel_pred_tok = []
                for i in word_ind:
                    sel_pred_tok.append(predicted_tokens_classes[i])
                logger.info(f"Tt Predicted {sel_pred_tok}")
                tt_iter_predictions.append(sel_pred_tok)

            cpu_predictions.extend(torch_iter_predictions)
            tt_predictions.extend(tt_iter_predictions)

        print("\n\n torch results:")
        calculate_metrics(cpu_predictions, labels_list)

        print("\n\n ttnn results:")
        calculate_metrics(tt_predictions, labels_list)

    return tt_predictions


@pytest.mark.parametrize("model_name", ["deepset/roberta-large-squad2"])
@pytest.mark.parametrize("bert", [ttnn_optimized_bert])
def test_demo(
    input_path,
    model_name,
    bert,
    device,
    use_program_cache,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    tt_lib.profiler.set_profiler_location(f"tt_metal/tools/profiler/logs/functional_robert")
    return run_roberta_question_and_answering_inference(
        device=device,
        use_program_cache=use_program_cache,
        model_name=model_name,
        batch_size=8,
        sequence_size=384,
        bert=bert,
        input_path=input_path,
    )


@pytest.mark.parametrize("model_name", ["deepset/roberta-large-squad2"])
@pytest.mark.parametrize("bert", [ttnn_bert, ttnn_optimized_bert])
@pytest.mark.parametrize(
    "n_iterations",
    ((3),),
)
def test_demo_squadv2(
    model_name,
    bert,
    n_iterations,
    device,
    use_program_cache,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    return run_roberta_question_and_answering_inference_squad_v2(
        device=device,
        use_program_cache=use_program_cache,
        model_name=model_name,
        batch_size=8,
        sequence_size=384,
        bert=bert,
        n_iterations=n_iterations,
    )


@pytest.mark.parametrize("model_name", ["Jean-Baptiste/roberta-large-ner-english"])
@pytest.mark.parametrize("bert", [ttnn_optimized_bert])
def test_token_demo(
    input_path,
    model_name,
    bert,
    device,
    use_program_cache,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    tt_lib.profiler.set_profiler_location(f"tt_metal/tools/profiler/logs/functional_robert")
    return run_roberta_token_inference(
        device=device,
        use_program_cache=use_program_cache,
        model_name=model_name,
        batch_size=8,
        sequence_size=384,
        bert=bert,
        input_path=input_path,
    )


@pytest.mark.parametrize("model_name", ["Jean-Baptiste/roberta-large-ner-english"])
@pytest.mark.parametrize("bert", [ttnn_optimized_bert])
@pytest.mark.parametrize(
    "n_iterations",
    ((10),),
)
def test_token_demo_conll2003(
    input_path,
    model_name,
    bert,
    device,
    use_program_cache,
    n_iterations,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()

    tt_lib.profiler.set_profiler_location(f"tt_metal/tools/profiler/logs/functional_robert")
    return run_roberta_token_inference_conll2003(
        device=device,
        use_program_cache=use_program_cache,
        model_name=model_name,
        batch_size=8,
        sequence_size=384,
        bert=bert,
        n_iterations=n_iterations,
    )
