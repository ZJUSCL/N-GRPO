"""
Preprocess the AMC 2023 dataset to parquet format.
"""

import argparse
import os
from typing import Any

from datasets import load_dataset

from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def load_amc23_dataset(dataset_path: str | None, data_source: str, split: str):
    if dataset_path is not None:
        return load_dataset(dataset_path, split=split, trust_remote_code=True)
    return load_dataset(data_source, split=split, trust_remote_code=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/amc23")
    parser.add_argument(
        "--dataset_path",
        default=None,
        help="Path to a locally prepared Hugging Face dataset directory.",
    )
    parser.add_argument(
        "--data_source",
        default="math-ai/amc23",
        help="Fallback dataset source on HuggingFace if --dataset_path is not provided.",
    )

    args = parser.parse_args()

    data_source = args.data_source

    dataset = load_amc23_dataset(args.dataset_path, data_source, "test")
    column_names = dataset.column_names

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    def make_map_fn():
        def process_fn(example, idx):
            question_raw = example.pop("question")
            question = (
                f"{question_raw} {instruction_following}" if instruction_following is not None else question_raw
            )
            answer_raw = example.pop("answer")
            solution = str(answer_raw)
            source_for_metrics = f"{data_source}"
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
                    "split": "test",
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    dataset = dataset.map(
        function=make_map_fn(),
        with_indices=True,
        remove_columns=column_names,
    )

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
