# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re

_ANSWER_CLIP_CHARS = 1000


def extract_answer(solution_str: str) -> str | None:
    if solution_str is None:
        return None

    text = str(solution_str).strip()
    if len(text) > _ANSWER_CLIP_CHARS:
        text = text[-_ANSWER_CLIP_CHARS:]

    patterns = [
        r"<answer>\s*([A-Z])\s*</answer>",
        r"\\boxed\s*\{\s*([A-Z])\s*\}",
        r"(?:final answer|answer)\s*(?:is)?\s*[:：-]?\s*\(?\s*([A-Z])\s*\)?",
        r"\(([A-Z])\)\s*(?:$|[\n\r])",
        r"^\s*([A-Z])\s*$",
        r"\b([A-Z])\b(?=[^A-Za-z]*$)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    return None


def compute_score(solution_str: str, ground_truth: str) -> dict[str, object]:
    pred = extract_answer(solution_str) or ""
    gt = str(ground_truth).strip().upper()

    score = 1.0 if pred == gt else 0.0
    return {
        "score": score,
        "acc": bool(score),
        "pred": pred,
    }
