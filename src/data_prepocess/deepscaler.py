"""
Preprocess the DeepScaleR dataset to parquet format.
"""

import argparse
import os

import datasets


def load_deepscaler_dataset(dataset_path: str | None, data_source: str):
    if dataset_path is not None:
        return datasets.load_dataset(dataset_path)
    return datasets.load_dataset(data_source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/DeepScaleR-Preview-Dataset")
    parser.add_argument(
        "--dataset_path",
        default=None,
        help="Path to a locally prepared Hugging Face dataset directory.",
    )
    parser.add_argument(
        "--data_source",
        default="agentica-org/DeepScaleR-Preview-Dataset",
        help="Fallback dataset source on HuggingFace if --dataset_path is not provided.",
    )

    args = parser.parse_args()

    data_source = args.data_source
    dataset = load_deepscaler_dataset(
        args.dataset_path, data_source
    )

    train_dataset = dataset["train"]

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}"

    def make_map_fn(split: str):
        def process_fn(example, idx):
            question_raw = example.get("problem") or example.get("question")
            if question_raw is None:
                raise KeyError("Input example is missing 'problem' or 'question' field.")

            answer_raw = example.get("answer")
            if answer_raw is None:
                raise KeyError("Input example is missing 'answer' field.")

            solution_raw = example.get("solution")

            question = f"{question_raw} {instruction_following}"
            solution = str(answer_raw).strip()
            source_for_metrics = f"{data_source}_{split}"

            data = {
                "data_source": source_for_metrics,
                "reward_data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "solution": solution_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
