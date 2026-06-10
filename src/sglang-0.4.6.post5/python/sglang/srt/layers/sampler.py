import logging
from typing import List

import torch
import torch.distributed as dist
from torch import nn

from sglang.srt.distributed import get_tensor_model_parallel_group
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import global_server_args_dict
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.utils import crash_on_warnings, get_bool_env_var, is_cuda

if is_cuda():
    from sgl_kernel import (
        min_p_sampling_from_probs,
        top_k_renorm_prob,
        top_k_top_p_sampling_from_probs,
        top_p_renorm_prob,
    )


logger = logging.getLogger(__name__)

SYNC_TOKEN_IDS_ACROSS_TP = get_bool_env_var("SYNC_TOKEN_IDS_ACROSS_TP")


class Sampler(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_nan_detection = global_server_args_dict["enable_nan_detection"]
        self.tp_sync_group = get_tensor_model_parallel_group().device_group

        if global_server_args_dict["enable_dp_attention"]:
            self.tp_sync_group = get_attention_tp_group().device_group

    def _get_vocab_norm(self, vocab_weight: torch.Tensor) -> torch.Tensor:
        """Cache normalized vocab embeddings to avoid per-step recompute."""
        cache = getattr(self, "_latent_vocab_norm_cache", None)
        version = getattr(vocab_weight, "_version", None)
        key = (id(vocab_weight), version, vocab_weight.dtype, vocab_weight.device)
        if isinstance(cache, dict) and cache.get("key") == key:
            return cache["norm"]
        with torch.no_grad():
            vocab_norm = torch.nn.functional.normalize(vocab_weight, dim=-1)
        self._latent_vocab_norm_cache = {"key": key, "norm": vocab_norm}
        return vocab_norm

    def _get_vocab_sq_norm(self, vocab_weight: torch.Tensor) -> torch.Tensor:
        """Cache squared L2 norms for vocab embeddings."""
        cache = getattr(self, "_latent_vocab_sq_norm_cache", None)
        version = getattr(vocab_weight, "_version", None)
        key = (id(vocab_weight), version, vocab_weight.dtype, vocab_weight.device)
        if isinstance(cache, dict) and cache.get("key") == key:
            return cache["sq_norm"]
        with torch.no_grad():
            vocab_sq_norm = (vocab_weight * vocab_weight).sum(dim=-1)
        self._latent_vocab_sq_norm_cache = {"key": key, "sq_norm": vocab_sq_norm}
        return vocab_sq_norm

    def forward(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        return_logprob: bool,
        top_logprobs_nums: List[int],
        token_ids_logprobs: List[List[int]],
    ):
        """Run a sampler & compute logprobs and update logits_output accordingly.

        Args:
            logits_output: The logits from the model forward
            sampling_info: Metadata for sampling
            return_logprob: If set, store the output logprob information to
                logits_output
            top_logprobs_nums: Number of top lobprobs per sequence in a batch
            batch_next_token_ids: next token IDs. If set, skip sampling and only
                compute output logprobs It is used for speculative decoding which
                performs sampling in draft workers.
        """
        logits = logits_output.next_token_logits

        # Apply the custom logit processors if registered in the sampling info.
        if sampling_info.has_custom_logit_processor:
            self._apply_custom_logit_processor(logits, sampling_info)

        if self.use_nan_detection and torch.any(torch.isnan(logits)):
            logger.warning("Detected errors during sampling! NaN in the logits.")
            logits = torch.where(
                torch.isnan(logits), torch.full_like(logits, -1e5), logits
            )
            if crash_on_warnings():
                raise ValueError("Detected errors during sampling! NaN in the logits.")

        hrpo_probs = None
        if sampling_info.is_all_greedy:
            # Use torch.argmax if all requests use greedy sampling
            batch_next_token_ids = torch.argmax(logits, -1)
            if return_logprob:
                logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            if sampling_info.hrpo_enable:
                hrpo_probs = torch.softmax(logits, dim=-1)
        else:
            # Post process logits
            logits.div_(sampling_info.temperatures)
            logits[:] = torch.softmax(logits, dim=-1)
            probs = logits
            del logits

            if global_server_args_dict["sampling_backend"] == "flashinfer":
                if return_logprob:
                    # NOTE: the top_p_renorm_prob from flashinfer has numerical problems,
                    # https://github.com/flashinfer-ai/flashinfer/issues/708
                    # so we use the torch implementation.

                    # clamp to avoid -inf
                    logprobs = torch.log(
                        top_p_normalize_probs_torch(probs, sampling_info.top_ps)
                    ).clamp(min=torch.finfo(probs.dtype).min)

                max_top_k_round, batch_size = 32, probs.shape[0]
                if sampling_info.need_min_p_sampling:
                    probs = top_k_renorm_prob(probs, sampling_info.top_ks)
                    probs = top_p_renorm_prob(probs, sampling_info.top_ps)
                    batch_next_token_ids = min_p_sampling_from_probs(
                        probs, sampling_info.min_ps
                    )
                else:
                    # Check Nan will throw exception, only check when crash_on_warnings is True
                    check_nan = self.use_nan_detection and crash_on_warnings()
                    batch_next_token_ids = top_k_top_p_sampling_from_probs(
                        probs,
                        sampling_info.top_ks,
                        sampling_info.top_ps,
                        filter_apply_order="joint",
                        check_nan=check_nan,
                    )

            elif global_server_args_dict["sampling_backend"] == "pytorch":
                # A slower fallback implementation with torch native operations.
                batch_next_token_ids = top_k_top_p_min_p_sampling_from_probs_torch(
                    probs,
                    sampling_info.top_ks,
                    sampling_info.top_ps,
                    sampling_info.min_ps,
                    sampling_info.need_min_p_sampling,
                )

                if return_logprob:
                    # clamp to avoid -inf
                    logprobs = torch.log(
                        top_p_normalize_probs_torch(probs, sampling_info.top_ps)
                    ).clamp(min=torch.finfo(probs.dtype).min)
            else:
                raise ValueError(
                    f"Invalid sampling backend: {global_server_args_dict['sampling_backend']}"
                )
            if sampling_info.hrpo_enable:
                hrpo_probs = probs

        # Attach logprobs to logits_output (in-place modification)
        if return_logprob:
            if any(x > 0 for x in top_logprobs_nums):
                (
                    logits_output.next_token_top_logprobs_val,
                    logits_output.next_token_top_logprobs_idx,
                ) = get_top_logprobs(logprobs, top_logprobs_nums)

            if any(x is not None for x in token_ids_logprobs):
                (
                    logits_output.next_token_token_ids_logprobs_val,
                    logits_output.next_token_token_ids_logprobs_idx,
                ) = get_token_ids_logprobs(logprobs, token_ids_logprobs)

            logits_output.next_token_logprobs = logprobs[
                torch.arange(len(batch_next_token_ids), device=sampling_info.device),
                batch_next_token_ids,
            ]

        if SYNC_TOKEN_IDS_ACROSS_TP or sampling_info.grammars:
            # For performance reasons, SGLang does not sync the final token IDs across TP ranks by default.
            # This saves one all-reduce, but the correctness of this approach depends on the determinism of several operators:
            # the last all-reduce, the last lm_head matmul, and all sampling kernels.
            # These kernels are deterministic in most cases, but there are some rare instances where they are not deterministic.
            # In such cases, enable this env variable to prevent hanging due to TP ranks becoming desynchronized.
            # When using xgrammar, this becomes more likely so we also do the sync when grammar is used.

            torch.distributed.all_reduce(
                batch_next_token_ids,
                op=dist.ReduceOp.MIN,
                group=self.tp_sync_group,
            )

        # ==========
        # begin of HRPO rollout (verl)
        # ==========
        if sampling_info.hrpo_enable:
            vocab_weight = getattr(logits_output, "vocab_embedding_weight", None)
            if vocab_weight is None:
                raise RuntimeError("HRPO requires vocab embedding weight in logits_output.")
            if hrpo_probs is None:
                hrpo_probs = torch.softmax(logits_output.next_token_logits, dim=-1)

            denom = torch.linalg.norm(hrpo_probs, dim=-1, keepdim=True).clamp(min=1e-6)
            probs_cast = hrpo_probs.to(vocab_weight.dtype)

            tp_world = getattr(logits_output, "tp_world_size", 1) or 1
            start_idx = getattr(logits_output, "vocab_start_index", None)
            end_idx = getattr(logits_output, "vocab_end_index", None)
            if tp_world > 1 and start_idx is not None and end_idx is not None:
                local_probs = probs_cast[:, start_idx:end_idx]
                local_embed = torch.matmul(local_probs, vocab_weight)
                torch.distributed.all_reduce(
                    local_embed, op=dist.ReduceOp.SUM, group=self.tp_sync_group
                )
                thinking_embeds = local_embed
            else:
                thinking_embeds = torch.matmul(probs_cast, vocab_weight)

            thinking_embeds = thinking_embeds / denom.to(thinking_embeds.dtype)
            logits_output.latent_thinking_embeds = thinking_embeds
        # ==========
        # end of HRPO rollout (verl)
        # ==========

        # ==========
        # begin of latent rollout (verl)
        # ==========
        if sampling_info.latent_rollout_enable:
            vocab_weight = getattr(logits_output, "vocab_embedding_weight", None)
            mix_strategy = sampling_info.latent_mix_strategy
            mix_top_k = sampling_info.latent_mix_top_k
            mix_temperature = max(float(getattr(sampling_info, "latent_mix_temperature", 1.0)), 1e-6)
            latent_is_validate = bool(getattr(sampling_info, "latent_is_validate", False))
            need_entropy = sampling_info.latent_need_entropy
            if (
                (mix_strategy == "neighbor" and mix_top_k > 0)
                or need_entropy
                or (mix_top_k > 0)
                or (mix_strategy == "gaussian_embed_noise")
            ):
                latent_logits = logits_output.next_token_logits
                latent_logits_raw = latent_logits
                temperatures = sampling_info.temperatures
                top_ps = sampling_info.top_ps
                top_ks = sampling_info.top_ks
                vocab_size = latent_logits.size(-1)

                if temperatures is not None:
                    latent_logits = latent_logits / temperatures

                active_topk = (top_ks > 0) & (top_ks < vocab_size)
                if torch.any(active_topk):
                    max_k = int(top_ks[active_topk].max().item())
                    topk_vals, _ = torch.topk(latent_logits, k=max_k, dim=-1)
                    kth_indices = (top_ks.clamp_min(1) - 1).view(-1, 1)
                    kth_vals = topk_vals.gather(1, kth_indices).squeeze(1)
                    kth_vals = torch.where(
                        active_topk, kth_vals, torch.full_like(kth_vals, float("inf"))
                    )
                    latent_logits = torch.where(
                        latent_logits < kth_vals.view(-1, 1),
                        torch.full_like(latent_logits, float("-inf")),
                        latent_logits,
                    )

                active_topp = (top_ps > 0.0) & (top_ps < 1.0)
                if torch.any(active_topp):
                    sorted_logits, sorted_indices = torch.sort(
                        latent_logits, descending=True, dim=-1
                    )
                    sorted_probs = torch.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    topp_mask = cumulative_probs > top_ps.view(-1, 1)
                    topp_mask = topp_mask & active_topp.view(-1, 1)
                    sorted_logits = sorted_logits.masked_fill(
                        topp_mask, float("-inf")
                    )
                    latent_logits = torch.full_like(
                        latent_logits, float("-inf")
                    ).scatter(-1, sorted_indices, sorted_logits)

                latent_probs = torch.softmax(latent_logits, dim=-1)

                if need_entropy:
                    log_probs = torch.log(torch.clamp(latent_probs, min=1e-9))
                    logits_output.latent_entropy = -(latent_probs * log_probs).sum(-1)
                    max_prob = latent_probs.max(dim=-1).values.mean().item()
                    entropy_mean = logits_output.latent_entropy.mean().item()
                    temp_mean = (
                        temperatures.mean().item() if temperatures is not None else 1.0
                    )
                    raw_max = latent_logits_raw.max(dim=-1).values.mean().item()
                    raw_std = latent_logits_raw.std(dim=-1).mean().item()
                    print(
                        "[latent_entropy debug] "
                        f"max_prob={max_prob:.6f} entropy={entropy_mean:.6f} "
                        f"temp={temp_mean:.4f} raw_max={raw_max:.4f} raw_std={raw_std:.4f}"
                    )

                if mix_strategy == "orthogonal_noise" and mix_top_k > 0:
                    if vocab_weight is None or (getattr(logits_output, "tp_world_size", 1) or 1) > 1:
                        raise RuntimeError("orthogonal_noise requires vocab embedding weight and tp_size=1")
                    top_k = min(int(mix_top_k), latent_logits.size(-1))
                    scores = latent_logits / mix_temperature
                    topk_vals, topk_indices = torch.topk(scores, k=top_k, dim=-1)
                    top1_idx = topk_indices[:, 0]
                    top1_embed = vocab_weight[top1_idx]
                    if top_k > 1:
                        other_indices = topk_indices[:, 1:]
                        other_embeds = vocab_weight[other_indices]
                        top1_norm_sq = (top1_embed * top1_embed).sum(dim=-1, keepdim=True).clamp(min=1e-6)
                        proj_coeff = (other_embeds * top1_embed.unsqueeze(1)).sum(dim=-1, keepdim=True) / top1_norm_sq.unsqueeze(1)
                        orth_dirs = other_embeds - proj_coeff * top1_embed.unsqueeze(1)
                        dir_norm = orth_dirs.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                        dir_unit = orth_dirs / dir_norm
                        noise_weights = torch.softmax(topk_vals[:, 1:] / mix_temperature, dim=-1).to(latent_logits.dtype)
                        noise_std = noise_weights.unsqueeze(-1) * dir_norm
                        noise = torch.randn_like(dir_unit) * noise_std
                        noise_sum = noise.sum(dim=1)
                        latent_embed = top1_embed + noise_sum
                    else:
                        latent_embed = top1_embed
                    logits_output.latent_thinking_embeds = latent_embed

                if mix_strategy == "gaussian_embed_noise" and mix_top_k > 0:
                    top_k = min(int(mix_top_k), latent_logits.size(-1))
                    scores = latent_logits / mix_temperature
                    _, topk_indices = torch.topk(scores, k=top_k, dim=-1)
                    selected_scores = torch.gather(scores, -1, topk_indices)
                    topk_probs = torch.softmax(selected_scores, dim=-1).to(latent_logits.dtype)
                    logits_output.latent_topk_indices = topk_indices
                    logits_output.latent_topk_probs = topk_probs

                if mix_strategy in ("topk", "topk-gumbel") and mix_top_k > 0:
                    top_k = min(int(mix_top_k), latent_logits.size(-1))
                    scores = latent_logits / mix_temperature
                    if mix_strategy == "topk-gumbel" and not latent_is_validate:
                        uniform = torch.rand_like(scores, dtype=torch.float32)
                        uniform = uniform.clamp_(1e-6, 1 - 1e-6)
                        gumbel = -torch.log(-torch.log(uniform)).to(scores.dtype)
                        scores = scores + gumbel
                    _, topk_indices = torch.topk(scores, k=top_k, dim=-1)
                    selected_logits = torch.gather(latent_logits, -1, topk_indices)
                    topk_probs = torch.softmax(selected_logits / mix_temperature, dim=-1).to(latent_logits.dtype)
                    logits_output.latent_topk_indices = topk_indices
                    logits_output.latent_topk_probs = topk_probs

                if (
                    mix_strategy == "neighbor"
                    and mix_top_k > 0
                    and vocab_weight is not None
                    and (getattr(logits_output, "tp_world_size", 1) or 1) <= 1
                ):
                    neighbor_metric = sampling_info.latent_neighbor_metric or "cosine"
                    neighbor_metric = str(neighbor_metric).lower()
                    if neighbor_metric == "__mixed__":
                        raise RuntimeError(
                            "latent_rollout_neighbor_metric must be consistent within a batch"
                        )
                    allowed_metrics = {"cosine", "dot", "l1", "l2"}
                    if neighbor_metric not in allowed_metrics:
                        raise RuntimeError(
                            "latent_rollout_neighbor_metric must be one of: cosine, dot, l1, l2"
                        )
                    top1_idx = torch.argmax(latent_logits, dim=-1)
                    top1_prob = latent_probs.gather(
                        1, top1_idx.view(-1, 1)
                    ).squeeze(1)

                    top1_embed = vocab_weight[top1_idx]
                    if neighbor_metric == "cosine":
                        vocab_norm = self._get_vocab_norm(vocab_weight)
                        top1_norm = torch.nn.functional.normalize(top1_embed, dim=-1)
                        sim = torch.matmul(vocab_norm, top1_norm.transpose(0, 1)).transpose(
                            0, 1
                        )
                    elif neighbor_metric == "dot":
                        sim = torch.matmul(vocab_weight, top1_embed.transpose(0, 1)).transpose(0, 1)
                    elif neighbor_metric == "l1":
                        sim = -(
                            (vocab_weight.unsqueeze(1) - top1_embed.unsqueeze(0))
                            .abs()
                            .sum(dim=-1)
                        )
                    elif neighbor_metric == "l2":
                        vocab_sq_norm = self._get_vocab_sq_norm(vocab_weight).view(-1, 1)
                        top1_sq_norm = (top1_embed * top1_embed).sum(dim=-1, keepdim=True)
                        dot = torch.matmul(vocab_weight, top1_embed.transpose(0, 1))
                        sim = (2.0 * dot - vocab_sq_norm - top1_sq_norm.transpose(0, 1)).transpose(0, 1)
                    else:
                        raise RuntimeError(
                            f"Unexpected latent_rollout_neighbor_metric: {neighbor_metric}"
                        )
                    sim = sim.clone()
                    sim.scatter_(1, top1_idx.view(-1, 1), float("-inf"))

                    neighbor_k = max(int(mix_top_k) - 1, 0)
                    if neighbor_k > 0:
                        _, neighbor_idx = torch.topk(sim, k=neighbor_k, dim=-1)
                        neighbor_probs = latent_probs.gather(1, neighbor_idx)
                        latent_neighbor_indices = torch.cat(
                            [top1_idx.view(-1, 1), neighbor_idx], dim=-1
                        )
                        latent_neighbor_probs = torch.cat(
                            [top1_prob.view(-1, 1), neighbor_probs], dim=-1
                        )
                    else:
                        latent_neighbor_indices = top1_idx.view(-1, 1)
                        latent_neighbor_probs = top1_prob.view(-1, 1)

                    logits_output.latent_neighbor_indices = latent_neighbor_indices
                    logits_output.latent_neighbor_probs = latent_neighbor_probs
        # ==========
        # end of latent rollout (verl)
        # ==========

        return batch_next_token_ids

    def _apply_custom_logit_processor(
        self, logits: torch.Tensor, sampling_batch_info: SamplingBatchInfo
    ):
        """Apply custom logit processors to the logits.
        This function will modify the logits in-place."""

        assert logits.shape[0] == len(sampling_batch_info), (
            f"The batch size of logits ({logits.shape[0]}) does not match the batch size of "
            f"sampling_batch_info ({len(sampling_batch_info)})"
        )

        for _, (
            processor,
            batch_mask,
        ) in sampling_batch_info.custom_logit_processor.items():
            # Get the batch indices that need to be processed
            batch_indices = batch_mask.nonzero(as_tuple=True)[0]

            assert batch_mask.shape[0] == len(sampling_batch_info), (
                f"The number of batch mask ({batch_mask.shape[0]}) does not match the number of "
                f"sampling_batch_info ({len(sampling_batch_info)})"
            )

            # Apply the processor to the logits
            logits[batch_mask] = processor(
                logits[batch_mask],
                [sampling_batch_info.custom_params[i] for i in batch_indices],
            )

            logger.debug(
                f"Custom logit processor {processor.__class__.__name__} is applied."
            )


def top_k_top_p_min_p_sampling_from_probs_torch(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
    need_min_p_sampling: bool,
):
    """A top-k, top-p and min-p sampling implementation with native pytorch operations."""
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[
        torch.arange(0, probs.shape[-1], device=probs.device).view(1, -1)
        >= top_ks.view(-1, 1)
    ] = 0.0
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0

    if need_min_p_sampling:
        min_p_thresholds = probs_sort[:, 0] * min_ps
        probs_sort[probs_sort < min_p_thresholds.view(-1, 1)] = 0.0

    sampled_index = torch.multinomial(probs_sort, num_samples=1)
    # int32 range is enough to represent the token ids
    probs_idx = probs_idx.to(torch.int32)
    batch_next_token_ids = torch.gather(probs_idx, dim=1, index=sampled_index).view(-1)
    return batch_next_token_ids


def top_p_normalize_probs_torch(
    probs: torch.Tensor,
    top_ps: torch.Tensor,
):
    # See also top_k_top_p_min_p_sampling_from_probs_torch
    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[(probs_sum - probs_sort) > top_ps.view(-1, 1)] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return torch.zeros_like(probs_sort).scatter_(-1, probs_idx, probs_sort)


def get_top_logprobs(logprobs: torch.Tensor, top_logprobs_nums: List[int]):
    max_k = max(top_logprobs_nums)
    ret = logprobs.topk(max_k, dim=1)
    values = ret.values.tolist()
    indices = ret.indices.tolist()

    output_top_logprobs_val = []
    output_top_logprobs_idx = []
    for i, k in enumerate(top_logprobs_nums):
        output_top_logprobs_val.append(values[i][:k])
        output_top_logprobs_idx.append(indices[i][:k])
    return output_top_logprobs_val, output_top_logprobs_idx


def get_token_ids_logprobs(logprobs: torch.Tensor, token_ids_logprobs: List[List[int]]):
    output_token_ids_logprobs_val = []
    output_token_ids_logprobs_idx = []
    for i, token_ids in enumerate(token_ids_logprobs):
        if token_ids is not None:
            output_token_ids_logprobs_val.append(logprobs[i, token_ids].tolist())
            output_token_ids_logprobs_idx.append(token_ids)
        else:
            output_token_ids_logprobs_val.append([])
            output_token_ids_logprobs_idx.append([])

    return output_token_ids_logprobs_val, output_token_ids_logprobs_idx
