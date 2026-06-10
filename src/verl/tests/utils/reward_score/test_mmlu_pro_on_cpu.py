# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import pytest

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score.mmlu_pro import extract_answer


@pytest.mark.parametrize(
    ("solution_str", "expected"),
    [
        ("Answer: C", "C"),
        ("The answer is (J).", "J"),
        ("Reasoning...\n\\boxed{A}", "A"),
        ("<answer>h</answer>", "H"),
        ("B", "B"),
        ("Final answer: i", "I"),
        ("No valid choice in this response.", None),
    ],
)
def test_extract_answer(solution_str, expected):
    assert extract_answer(solution_str) == expected


def test_default_compute_score_with_mmlu_pro_source():
    correct = default_compute_score("TIGER-Lab/MMLU-Pro_test", "Final answer: (I)", "I")
    wrong = default_compute_score("mmlu_pro_validation", "Answer: A", "B")

    assert isinstance(correct, dict)
    assert correct["score"] == 1.0
    assert correct["acc"] is True
    assert correct["pred"] == "I"

    assert isinstance(wrong, dict)
    assert wrong["score"] == 0.0
    assert wrong["acc"] is False
    assert wrong["pred"] == "A"
