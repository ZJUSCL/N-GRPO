"""Model instrumentation for latent-state implicit GRPO rollouts."""

from __future__ import annotations

from types import MethodType
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast


def _prepare_mask(mask: torch.Tensor, *, hidden_dim: int) -> torch.Tensor:
    """Broadcast a 2D mask to match embedding dimensionality."""

    if mask.dim() == 2:
        mask = mask.unsqueeze(-1)
    if mask.size(-1) == 1:
        mask = mask.expand(-1, -1, hidden_dim)
    return mask


def apply_latent_rollout_patch(model: nn.Module, *, latent_cfg) -> None:
    """Allow passing latent hidden states directly into the transformer forward."""

    if latent_cfg is None or not getattr(latent_cfg, "enable", False):
        return

    base = getattr(model, "model", None)
    if base is None:
        raise RuntimeError("Latent rollout patch expects AutoModelForCausalLM-style modules with a `model` attribute.")

    original_forward = base.forward
    outer_forward = model.forward
    hidden_size = getattr(base.config, "hidden_size", None)
    if hidden_size is None:
        raise RuntimeError("Unable to infer hidden_size for latent rollout instrumentation.")

    eps = 1e-8

    def forward_with_latent(self, *call_args, **call_kwargs):
        latent_mask = call_kwargs.pop("thinking_mask", None)
        if latent_mask is None and hasattr(self, "_latent_rollout_mask"):
            latent_mask = getattr(self, "_latent_rollout_mask")
            delattr(self, "_latent_rollout_mask")
        latent_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        if latent_mask is not None:
            latent_embeds = call_kwargs.get("inputs_embeds")
            if latent_embeds is None:
                raise ValueError(
                    "Latent rollout forward expects `inputs_embeds` when `thinking_mask` is provided."
                )

            input_ids = call_kwargs.pop("input_ids", None)
            base_embeds = self.embed_tokens(input_ids) if input_ids is not None else latent_embeds
            mask = _prepare_mask(latent_mask.bool(), hidden_dim=base_embeds.size(-1))

            latent_embeds = latent_embeds.to(base_embeds.dtype)
            combined = torch.where(mask, latent_embeds, base_embeds)
            call_kwargs["inputs_embeds"] = combined

            delta = latent_embeds - base_embeds
            ratio = torch.linalg.norm(delta, dim=-1) / (torch.linalg.norm(base_embeds, dim=-1) + eps)
            latent_cache = (mask.detach(), ratio.detach())

        outputs = original_forward.__func__(self, *call_args, **call_kwargs)

        if latent_cache is not None and isinstance(outputs, (BaseModelOutputWithPast, CausalLMOutputWithPast)):
            mask_meta, ratio_meta = latent_cache
            extras = [] if outputs.hidden_states is None else list(outputs.hidden_states)
            extras.append(mask_meta)
            extras.append(ratio_meta)
            outputs.hidden_states = tuple(extras)

        return outputs

    base._latent_rollout_original_forward = original_forward  # type: ignore[attr-defined]
    base.forward = MethodType(forward_with_latent, base)

    def outer_forward_with_latent(self, *call_args, **call_kwargs):
        latent_mask = call_kwargs.pop("thinking_mask", None)
        if latent_mask is not None and hasattr(self, "model"):
            setattr(self.model, "_latent_rollout_mask", latent_mask)
        try:
            return outer_forward(*call_args, **call_kwargs)
        finally:
            if latent_mask is not None and hasattr(self, "model") and hasattr(self.model, "_latent_rollout_mask"):
                delattr(self.model, "_latent_rollout_mask")

    model.forward = MethodType(outer_forward_with_latent, model)

    answer_start = getattr(latent_cfg, "answer_start", "####")
    setattr(model, "answer_start", answer_start)
