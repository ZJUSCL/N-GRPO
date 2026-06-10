"""Lightweight latent rollout patch for SGLang models.

- 支持在 forward 时传入 `thinking_mask` 与 `input_embeds`，并用 mask 位置的 latent embedding
  覆盖原始 token embedding。
- 将 mask 与 ratio 追加到 hidden_states，便于上层统计。
"""

from __future__ import annotations

from types import MethodType
from typing import Optional, Tuple

import torch
from torch import nn


def apply_latent_rollout_patch(model: nn.Module) -> None:
    if getattr(model, "_latent_rollout_patched", False):
        return

    base = getattr(model, "model", None)
    if base is None:
        base = model
    if not hasattr(base, "forward") or not hasattr(base, "embed_tokens"):
        return

    original_forward = base.forward
    eps = 1e-8

    def forward_with_latent(self, *args, **kwargs):
        thinking_mask = kwargs.pop("thinking_mask", None)
        input_embeds = kwargs.get("input_embeds", None)
        latent_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        if thinking_mask is not None and input_embeds is not None:
            input_ids = kwargs.get("input_ids", None)
            base_embeds = (
                self.embed_tokens(input_ids)
                if input_ids is not None
                else input_embeds
            )
            mask = thinking_mask
            if mask.dim() == 1:
                mask = mask.unsqueeze(-1)
            if mask.size(-1) == 1:
                mask = mask.expand(-1, base_embeds.size(-1))

            mixed = torch.where(mask.bool(), input_embeds, base_embeds)
            kwargs["input_embeds"] = mixed

            delta = (input_embeds - base_embeds).to(torch.float32)
            ratio = torch.linalg.norm(delta, dim=-1) / (
                torch.linalg.norm(base_embeds, dim=-1) + eps
            )
            latent_cache = (mask.detach(), ratio.detach())

        outputs = original_forward(*args, **kwargs)

        if latent_cache is not None and hasattr(outputs, "hidden_states"):
            mask_meta, ratio_meta = latent_cache
            extras = [] if outputs.hidden_states is None else list(outputs.hidden_states)
            extras.append(mask_meta)
            extras.append(ratio_meta)
            outputs.hidden_states = tuple(extras)

        return outputs

    base.forward = MethodType(forward_with_latent, base)
    setattr(model, "_latent_rollout_patched", True)
