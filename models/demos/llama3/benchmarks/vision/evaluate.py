import argparse
import torch
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets

from llama_models.models.llama3.api.datatypes import ImageMedia, UserMessage
from llama_models.models.llama3.reference_impl.generation import Llama

from mmmu_utils import (
    CAT_SHORT2LONG,
    process_single_sample,
    process_prompt,
    eval_multi_choice,
    eval_open,
    parse_multi_choice_response,
)


# Main function to run the evaluation
def main(args):
    # Load the dataset
    if args.dataset == "MMMU/MMMU":
        sub_dataset_list = [
            load_dataset(args.dataset, subject, split=args.split) for subject in CAT_SHORT2LONG.values()
        ]
        dataset = concatenate_datasets(sub_dataset_list)
    else:
        dataset = load_dataset(args.dataset, args.setting, split=args.split)

    print(f"Loaded dataset with {len(samples)} samples")

    # Load model
    model = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
        model_parallel_size=None,
    )
    print("Model loaded")

    # Evaluate model
    correct_sum = 0
    with torch.no_grad():
        for sample in tqdm(samples):
            if args.dataset == "MMMU/MMMU_Pro":
                prompt, images = process_prompt(sample, args.setting)
                dialog = [UserMessage(content=[ImageMedia(image=image) for image in images] + [prompt])]
            else:
                sample = process_single_sample(sample, args)
                prompt, image = process_prompt_mmmu(sample)
                dialog = [UserMessage(content=[ImageMedia(image=image), prompt])]
            result = model.chat_completion(dialog, max_gen_len=None, temperature=0.0, top_p=0.0)
            response = result.generation.content

            if args.dataset == "MMMU/MMMU_Pro" or sample["question_type"] == "multiple-choice":
                index2ans, all_choices = get_multi_choice_info(sample["options"])
                pred_ans = parse_multi_choice_response(response, all_choices, index2ans)
                correct = eval_multi_choice(pred_ans, sample["ans"])
            else:
                correct = eval_open(pred_ans, sample["ans"])

            correct_sum += int(correct)

    # Print accuracy
    print(f"Accuracy: {correct_sum / len(samples):.2f}")


# Argument parser for command-line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on multiple datasets")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="/home/ubuntu/.llama/checkpoints/Llama3.2-11B-Vision-Instruct",
        help="Checkpoint directory for the model",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="/home/ubuntu/.llama/checkpoints/Llama3.2-11B-Vision-Instruct/tokenizer.model",
        help="Path to the tokenizer",
    )
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length for the model")
    parser.add_argument("--max_batch_size", type=int, default=1, help="Maximum batch size")
    parser.add_argument(
        "--dataset", type=str, default="MMMU/MMMU_Pro", help="Dataset to evaluate, or 'all' for concatenated dataset"
    )
    parser.add_argument("--setting", type=str, default="standard", help="MMMU Pro setting (standard/vision)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use (e.g., 'test', 'validation')")
    args = parser.parse_args()

    main(args)
