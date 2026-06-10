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
"""
The base class for Actor
"""

from abc import ABC, abstractmethod
import math
from typing import Optional, Tuple

import torch

from verl import DataProto

__all__ = ["BasePPOActor"]


class BasePPOActor(ABC):
    def __init__(self, config):
        """The base class for PPO actor

        Args:
            config (DictConfig): a config passed to the PPOActor. We expect the type to be
                DictConfig (https://omegaconf.readthedocs.io/), but it can be any namedtuple in general.
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute logits given a batch of data.

        Args:
            data (DataProto): a batch of data represented by DataProto. It must contain key ```input_ids```,
                ```attention_mask``` and ```position_ids```.

        Returns:
            DataProto: a DataProto containing the key ```log_probs```


        """
        pass

    @abstractmethod
    def update_policy(self, data: DataProto) -> dict:
        """Update the policy with an iterator of DataProto

        Args:
            data (DataProto): an iterator over the DataProto that returns by
                ```make_minibatch_iterator```

        Returns:
            Dict: a dictionary contains anything. Typically, it contains the statistics during updating the model
            such as ```loss```, ```grad_norm```, etc,.

        """
        pass

    def _get_entropy_top_ratio(self) -> float:
        """Fetch and normalize the entropy top-ratio config."""
        ratio_cfg = self.config.get("entropy_top_ratio", None)
        if ratio_cfg is None:
            return 0.0
        try:
            ratio = float(ratio_cfg)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid entropy_top_ratio={ratio_cfg}. Expect a float in (0, 1).")
        if ratio <= 0.0:
            return 0.0
        if ratio >= 1.0:
            return 1.0
        return ratio

    def _compute_entropy_top_mask(
        self,
        entropy: Optional[torch.Tensor],
        response_mask: torch.Tensor,
        top_ratio: Optional[float],
    ) -> Tuple[Optional[torch.Tensor], Optional[float]]:
        """Return a mask that keeps only the top-ratio entropy tokens."""
        if top_ratio is None or not (0.0 < top_ratio < 1.0):
            return None, None
        if entropy is None:
            raise RuntimeError("entropy_top_ratio requires entropy outputs, got None.")

        with torch.no_grad():
            entropy_fp32 = entropy.to(torch.float32)
            valid_mask = response_mask > 0 if response_mask.dtype != torch.bool else response_mask
            flat_entropy = torch.masked_select(entropy_fp32, valid_mask)
            token_count = flat_entropy.numel()
            if token_count == 0:
                return None, None
            top_k = max(1, min(token_count, math.ceil(token_count * top_ratio)))
            topk_values = torch.topk(flat_entropy, k=top_k, sorted=True).values
            threshold = topk_values[-1]
            entropy_mask_bool = (entropy_fp32 >= threshold) & valid_mask
            entropy_mask = entropy_mask_bool.to(response_mask.dtype)

        return entropy_mask, float(threshold.item())
