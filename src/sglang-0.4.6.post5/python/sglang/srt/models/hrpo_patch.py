"""HRPO patch for SGLang models.

- Supports passing `thinking_mask` and `input_embeds` to forward.
- Applies HRPO residual gating to mix latent embeddings.
- Appends mask and ratio to hidden_states when available.
"""

from __future__ import annotations

from types import MethodType
from typing import Optional, Tuple

import torch
from torch import nn


class ThinkingResidualLambda(nn.Module):
    """Learnable Lambda mapping used in HRPO gating."""

    def __init__(self, hidden_size: int, c: float) -> None:
        super().__init__()
        self.c = float(c)
        self.lambda_param = nn.Parameter(torch.zeros(hidden_size))

    @torch.no_grad()
    def reset_parameters(self, r_min: float, r_max: float) -> None:
        uniform = torch.empty_like(self.lambda_param).uniform_(r_min, r_max)
        transformed = -torch.log(torch.pow(uniform, -1.0 / self.c) - 1.0)
        self.lambda_param.copy_(transformed)

    def forward(self, r_t: torch.Tensor) -> torch.Tensor:
        lambda_pos = torch.nn.functional.softplus(-self.lambda_param, beta=1.0, threshold=20.0)
        return torch.exp(-self.c * lambda_pos * r_t)


def _ensure_hrpo_modules(base_model: nn.Module, hidden_size: int, c: float) -> bool:
    created = False
    if not hasattr(base_model, "thinking_residual_gate_r"):
        base_model.thinking_residual_gate_r = nn.Linear(hidden_size, hidden_size)
        created = True
    if not hasattr(base_model, "thinking_residual_gate_i"):
        base_model.thinking_residual_gate_i = nn.Linear(hidden_size, hidden_size)
        created = True
    if not hasattr(base_model, "thinking_residual_Lambda"):
        base_model.thinking_residual_Lambda = ThinkingResidualLambda(hidden_size, c)
        created = True
    return created


def _thinking_residual(
    self, embeds: torch.Tensor, residual: torch.Tensor, eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    gate_r = torch.sigmoid(self.thinking_residual_gate_r(embeds))
    gate_i = torch.sigmoid(self.thinking_residual_gate_i(embeds))
    a_t = self.thinking_residual_Lambda(gate_r)
    mixed = a_t * embeds + torch.sqrt(1.0 - a_t.pow(2) + eps) * (gate_i * residual)
    return mixed, a_t


def apply_hrpo_patch(
    model: nn.Module,
    *,
    mix_constant: float = 8.0,
    residual_r_min: float = 0.99,
    residual_r_max: float = 0.999,
) -> None:
    """Instrument the given model with HRPO gating if not already patched."""

    if getattr(model, "_hrpo_patched", False):
        return

    base = getattr(model, "model", None)
    if base is None:
        base = model
    if not hasattr(base, "forward") or not hasattr(base, "embed_tokens"):
        return

    hidden_size = getattr(base.config, "hidden_size", None)
    if hidden_size is None:
        return

    created = _ensure_hrpo_modules(base, hidden_size, float(mix_constant))
    base.thinking_residual = MethodType(_thinking_residual, base)
    if created:
        base.thinking_residual_Lambda.reset_parameters(float(residual_r_min), float(residual_r_max))

    original_forward = base.forward

    def forward_with_hrpo(self, *args, **kwargs):
        thinking_mask = kwargs.pop("thinking_mask", None)
        input_embeds = kwargs.get("input_embeds", None)
        hrpo_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        if thinking_mask is not None and input_embeds is not None:
            input_ids = kwargs.get("input_ids", None)
            base_embeds = self.embed_tokens(input_ids) if input_ids is not None else input_embeds

            mask = thinking_mask
            if mask.dim() == 1:
                mask = mask.unsqueeze(-1)
            if mask.size(-1) == 1:
                mask = mask.expand(-1, base_embeds.size(-1))

            mixed_embeds, a_t = self.thinking_residual(input_embeds, base_embeds)
            combined = torch.where(mask.bool(), mixed_embeds, base_embeds).to(mixed_embeds.dtype)
            kwargs["input_embeds"] = combined
            hrpo_cache = (mask.detach(), a_t.detach())

        outputs = original_forward(*args, **kwargs)

        if hrpo_cache is not None and hasattr(outputs, "hidden_states"):
            mask_meta, ratios = hrpo_cache
            ratio_reduced = ratios.mean(dim=-1)
            extras = [] if outputs.hidden_states is None else list(outputs.hidden_states)
            extras.append(mask_meta)
            extras.append(ratio_reduced)
            outputs.hidden_states = tuple(extras)

        return outputs

    base.forward = MethodType(forward_with_hrpo, base)
    setattr(model, "_hrpo_patched", True)

