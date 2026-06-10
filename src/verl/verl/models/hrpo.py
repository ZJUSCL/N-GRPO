"""HRPO-specific model instrumentation for gating latent representations."""

from __future__ import annotations

from types import MethodType
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast


class ThinkingResidualLambda(nn.Module):
    """Implements the learnable Lambda mapping described in the HRPO paper."""

    def __init__(self, hidden_size: int, c: float) -> None:
        super().__init__()
        self.c = float(c)
        self.lambda_param = nn.Parameter(torch.zeros(hidden_size))

    @torch.no_grad()
    def reset_parameters(self, r_min: float, r_max: float) -> None:
        """Initialise Lambda such that a_t starts inside (r_min, r_max)."""
        uniform = torch.empty_like(self.lambda_param).uniform_(r_min, r_max)
        # Convert radius r in (0, 1) to unconstrained lambda following HRPO implementation.
        transformed = -torch.log(torch.pow(uniform, -1.0 / self.c) - 1.0)
        self.lambda_param.copy_(transformed)

    def forward(self, r_t: torch.Tensor) -> torch.Tensor:
        # softplus ensures positivity, the exponential controls residual scaling.
        lambda_pos = torch.nn.functional.softplus(-self.lambda_param, beta=1.0, threshold=20.0)
        return torch.exp(-self.c * lambda_pos * r_t)


def _ensure_hrpo_modules(base_model: nn.Module, hidden_size: int, c: float) -> None:
    if not hasattr(base_model, "thinking_residual_gate_r"):
        base_model.thinking_residual_gate_r = nn.Linear(hidden_size, hidden_size)
    if not hasattr(base_model, "thinking_residual_gate_i"):
        base_model.thinking_residual_gate_i = nn.Linear(hidden_size, hidden_size)
    if not hasattr(base_model, "thinking_residual_Lambda"):
        base_model.thinking_residual_Lambda = ThinkingResidualLambda(hidden_size, c)


def _thinking_residual(self, embeds: torch.Tensor, residual: torch.Tensor, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    gate_r = torch.sigmoid(self.thinking_residual_gate_r(embeds))
    gate_i = torch.sigmoid(self.thinking_residual_gate_i(embeds))
    a_t = self.thinking_residual_Lambda(gate_r)
    mixed = a_t * embeds + torch.sqrt(1.0 - a_t.pow(2) + eps) * (gate_i * residual)
    return mixed, a_t


def _patch_forward(base_model: nn.Module, residual_key: str = "thinking_mask") -> None:
    original_forward = base_model.forward

    def forward_with_hrpo(self, *call_args, **call_kwargs):
        thinking_mask = call_kwargs.pop(residual_key, None)
        hrpo_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        if thinking_mask is not None:
            inputs_embeds = call_kwargs.get("inputs_embeds")
            if inputs_embeds is None:
                raise ValueError("HRPO forward expects inputs_embeds when thinking_mask is provided.")

            input_ids = call_kwargs.get("input_ids")
            base_embeds = self.embed_tokens(input_ids) if input_ids is not None else inputs_embeds

            mask = thinking_mask.bool()
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            if mask.size(-1) == 1:
                mask = mask.expand(-1, -1, base_embeds.size(-1))

            mixed_embeds, a_t = self.thinking_residual(inputs_embeds, base_embeds)
            combined = torch.where(mask, mixed_embeds, base_embeds).to(mixed_embeds.dtype)
            call_kwargs["inputs_embeds"] = combined

            hrpo_cache = (mask.detach(), a_t.detach())

        outputs = original_forward.__func__(self, *call_args, **call_kwargs)

        if hrpo_cache is not None and isinstance(outputs, (BaseModelOutputWithPast, CausalLMOutputWithPast)):
            mask, ratios = hrpo_cache
            ratio_reduced = ratios.mean(dim=-1)
            extras = [] if outputs.hidden_states is None else list(outputs.hidden_states)
            extras.append(mask)
            extras.append(ratio_reduced)
            outputs.hidden_states = tuple(extras)

        return outputs

    base_model._hrpo_original_forward = original_forward  # type: ignore[attr-defined]
    base_model.forward = MethodType(forward_with_hrpo, base_model)


def apply_hrpo_patch(model: nn.Module, *, hrpo_cfg) -> None:
    """Instrument the given causal LM with HRPO gating if requested.

    Args:
        model: The actor model (value-head variants supported) whose ``model`` attribute holds the transformer.
        hrpo_cfg: Config-like object with attributes ``enable``, ``residual_r_min``, ``residual_r_max``,
            ``mix_constant`` (named ``c`` in the paper), and ``answer_start``.
    """

    if not getattr(hrpo_cfg, "enable", False):
        return

    base = getattr(model, "model", None)
    if base is None:
        raise RuntimeError("HRPO patch expects AutoModelForCausalLM-style modules with a `model` attribute.")

    hidden_size = getattr(base.config, "hidden_size", None)
    if hidden_size is None:
        raise RuntimeError("Unable to infer hidden_size for HRPO instrumentation.")

    mix_constant = float(getattr(hrpo_cfg, "mix_constant", 8.0))
    _ensure_hrpo_modules(base, hidden_size, mix_constant)
    base.thinking_residual = MethodType(_thinking_residual, base)
    base.thinking_residual_Lambda.reset_parameters(
        getattr(hrpo_cfg, "residual_r_min", 0.99), getattr(hrpo_cfg, "residual_r_max", 0.999)
    )

    answer_start = getattr(hrpo_cfg, "answer_start", "####")
    setattr(model, "answer_start", answer_start)

    _patch_forward(base)

    # Expose helper for downstream reset if required.
    def reset_hrpo_state(self, r_min: Optional[float] = None, r_max: Optional[float] = None) -> None:
        self.model.thinking_residual_Lambda.reset_parameters(
            r_min if r_min is not None else getattr(hrpo_cfg, "residual_r_min", 0.99),
            r_max if r_max is not None else getattr(hrpo_cfg, "residual_r_max", 0.999),
        )

    model.reset_hrpo_state = MethodType(reset_hrpo_state, model)  # type: ignore[attr-defined]
