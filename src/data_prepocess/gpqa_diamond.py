"""
Preprocess the GPQA-Diamond dataset to parquet format.
"""

import argparse
import os

import datasets


def load_gpqa_dataset(dataset_path: str | None, data_source: str):
    if dataset_path is not None:
        return datasets.load_dataset(dataset_path)
    return datasets.load_dataset(data_source)


def normalize_answer(answer: str | None) -> str:
    if answer is None:
        return ""
    return str(answer).strip().upper()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./data/gpqa_diamond")
    parser.add_argument(
        "--dataset_path",
        default="./GPQA-Diamond",
        help="Path to a locally prepared Hugging Face dataset directory.",
    )
    parser.add_argument(
        "--data_source",
        default="gpqa_diamond",
        help="Fallback dataset source on HuggingFace if --dataset_path is not provided.",
    )

    args = parser.parse_args()

    data_source = args.data_source
    dataset = load_gpqa_dataset(args.dataset_path, data_source)

    instruction_following = (
        "Let's think step by step and output the final answer in the last line as:\n"
        "Answer: <A/B/C/D>"
    )

    def make_map_fn(split: str):
        def process_fn(example, idx):
            question_raw = example.get("question")
            if question_raw is None:
                raise KeyError("Input example is missing 'question' field.")

            answer_raw = example.get("answer")
            if answer_raw is None:
                raise KeyError("Input example is missing 'answer' field.")

            question = f"{question_raw}\n\n{instruction_following}"
            solution = normalize_answer(answer_raw)
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
                "ability": "qa",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    split = "test"
    if split not in dataset:
        raise KeyError("GPQA-Diamond only provides a 'test' split, but it was not found.")
    split_dataset = dataset[split]
    split_dataset = split_dataset.map(function=make_map_fn(split), with_indices=True)
    split_dataset.to_parquet(os.path.join(local_dir, f"{split}.parquet"))
