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

_ANSWER_CLIP_CHARS = 500


def extract_answer(solution_str: str) -> str | None:
    if solution_str is None:
        return None

    solution_str = str(solution_str).strip()
    if len(solution_str) > _ANSWER_CLIP_CHARS:
        solution_str = solution_str[-_ANSWER_CLIP_CHARS:]

    patterns = [
        r"\\boxed\s*\{\s*([ABCD])\s*\}",
        r"(?:final answer|answer)\s*[:：]\s*([ABCD])",
        r"\b([ABCD])\b(?=[^A-Za-z]*$)",
        r"\b([ABCD])\b",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, solution_str, flags=re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    return None


def compute_score(solution_str: str, ground_truth: str) -> dict[str, object]:
    pred = extract_answer(solution_str)
    if pred is None:
        pred = ""
    gt = str(ground_truth).strip().upper()

    score = 1.0 if pred == gt else 0.0
    acc = bool(score)

    return {
        "score": score,
        "acc": acc,
        "pred": pred,
    }
