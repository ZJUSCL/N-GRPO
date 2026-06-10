from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.distributed import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.io_struct import BatchEmbeddingOut, BatchTokenIDOut
from sglang.srt.managers.schedule_batch import BaseFinishReason, Req, ScheduleBatch

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import (
        EmbeddingBatchResult,
        GenerationBatchResult,
        ScheduleBatch,
        Scheduler,
    )

logger = logging.getLogger(__name__)

DEFAULT_FORCE_STREAM_INTERVAL = 50


class SchedulerOutputProcessorMixin:
    """
    This class implements the output processing logic for Scheduler.
    We put them into a separate file to make the `scheduler.py` shorter.
    """

    def _apply_latent_mix_for_req(
        self,
        req: Req,
        logits_output: LogitsProcessorOutput,
        logit_idx: int,
        next_token_id: int,
        is_validate: bool = False,
    ):
        """Compute latent embedding/mask for next step and stash in req buffers."""
        cfg = req.latent_cfg or {}
        mix_top_k = cfg.get("latent_rollout_mix_top_k", None)
        mix_temperature = float(cfg.get("latent_rollout_mix_temperature", 1.0))
        mix_entropy_threshold = cfg.get("latent_rollout_mix_entropy_threshold", None)
        mix_rate = cfg.get("latent_rollout_mix_rate", None)
        if mix_rate is not None:
            mix_rate = float(mix_rate)
            mix_rate = min(max(mix_rate, 0.0), 1.0)
        noise_rms_scale = max(float(cfg.get("latent_rollout_noise_rms_scale", 0.33)), 0.0)
        noise_on_eval = bool(cfg.get("latent_rollout_noise_on_eval", False))
        mix_strategy = cfg.get("latent_rollout_mix_strategy", None)
        use_neighbor_mix = mix_strategy == "neighbor"
        use_orthogonal_noise_mix = mix_strategy == "orthogonal_noise"
        use_gaussian_embed_noise_mix = mix_strategy == "gaussian_embed_noise"
        allowed_strategies = {
            "topk",
            "topk-gumbel",
            "neighbor",
            "orthogonal_noise",
            "gaussian_embed_noise",
        }
        if mix_strategy not in allowed_strategies:
            raise RuntimeError(
                "unsupported latent mix strategy (must be explicit): "
                "topk, topk-gumbel, neighbor, orthogonal_noise, gaussian_embed_noise"
            )
        disable_eval_mix = bool(cfg.get("latent_rollout_disable_mix_on_eval", False))
        answer_start_ids = cfg.get("latent_rollout_answer_start_ids", [])
        # force-decode 已弃用：保留配置字段但不再生效
        force_decode_interval = None
        force_decode_burn_in = 0
        # eval 严格禁用 latent 混合：直接返回，保持与离散生成一致
        if disable_eval_mix and is_validate:
            return

        logits = logits_output.next_token_logits[logit_idx]
        vocab_weight = getattr(logits_output, "vocab_embedding_weight", None)
        tp_world = getattr(logits_output, "tp_world_size", 1) or 1
        start_idx = getattr(logits_output, "vocab_start_index", None)
        end_idx = getattr(logits_output, "vocab_end_index", None)
        tp_group = get_tensor_model_parallel_group().device_group if tp_world > 1 else None

        def _gather_vocab_rows(id_tensor: torch.Tensor) -> torch.Tensor:
            if vocab_weight is None:
                raise RuntimeError("latent rollout requires vocab embedding weight in logits_output")
            id_tensor = id_tensor.to(torch.long).view(-1)
            if tp_world <= 1 or tp_group is None or start_idx is None or end_idx is None:
                return vocab_weight[id_tensor]

            local_out = torch.zeros(
                (id_tensor.numel(), vocab_weight.size(1)),
                device=vocab_weight.device,
                dtype=vocab_weight.dtype,
            )
            local_mask = (id_tensor >= start_idx) & (id_tensor < end_idx)
            if local_mask.any():
                local_indices = (id_tensor[local_mask] - start_idx).long()
                local_out[local_mask] = vocab_weight[local_indices]
            dist.all_reduce(local_out, op=dist.ReduceOp.SUM, group=tp_group)
            return local_out

        temperature = float(req.sampling_params.temperature)
        top_p = float(req.sampling_params.top_p)
        top_k_sampling = int(req.sampling_params.top_k)
        do_sample = not (
            top_k_sampling <= 1 and abs(top_p - 1.0) < 1e-6 and abs(temperature - 1.0) < 1e-6
        )

        if temperature != 1.0:
            logits = logits / max(temperature, 1e-6)

        # top_k / top_p warpers
        if top_k_sampling > 0 and top_k_sampling < logits.numel():
            kth = torch.topk(logits, top_k_sampling).values[-1]
            logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            sorted_logits[cumulative_probs > top_p] = float("-inf")
            logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_indices, sorted_logits)

        # 探索开关：
        # - 如果配置了 mix_rate：训练按概率，验证关闭
        # - 否则：按熵阈值（或默认 True），验证若未设置 mix_rate 仍会应用熵逻辑
        if disable_eval_mix and is_validate:
            explore_mask = False
        elif mix_rate is not None:
            explore_mask = False if is_validate else bool(torch.rand((), device=logits.device) < mix_rate)
        else:
            explore_mask = True

        latent_neighbor_indices = getattr(logits_output, "latent_neighbor_indices", None)
        latent_neighbor_probs = getattr(logits_output, "latent_neighbor_probs", None)
        latent_topk_indices = getattr(logits_output, "latent_topk_indices", None)
        latent_topk_probs = getattr(logits_output, "latent_topk_probs", None)
        latent_thinking_embeds = getattr(logits_output, "latent_thinking_embeds", None)
        latent_entropy_tensor = getattr(logits_output, "latent_entropy", None)
        use_precomputed_neighbor = (
            use_neighbor_mix
            and latent_neighbor_indices is not None
            and latent_neighbor_probs is not None
        )
        use_precomputed_topk = (
            mix_strategy in ("topk", "topk-gumbel", "gaussian_embed_noise")
            and latent_topk_indices is not None
            and latent_topk_probs is not None
        )
        if use_neighbor_mix and not use_precomputed_neighbor:
            raise RuntimeError("latent neighbor precompute missing; sampler path is required")
        if mix_strategy in ("topk", "topk-gumbel", "gaussian_embed_noise") and mix_top_k and mix_top_k > 0:
            if not use_precomputed_topk:
                raise RuntimeError("latent top-k precompute missing; sampler path is required")

        need_entropy = mix_rate is None and mix_entropy_threshold is not None
        if need_entropy and latent_entropy_tensor is None:
            raise RuntimeError("latent entropy precompute missing; sampler path is required")
        need_probs = False
        if mix_rate is not None and not explore_mask:
            need_entropy = False
            need_probs = False

        if need_probs:
            probs = torch.softmax(logits, dim=-1)
            if need_entropy:
                log_probs = torch.log(torch.clamp(probs, min=1e-9))
                entropy = -(probs * log_probs).sum()
            else:
                entropy = torch.tensor(0.0, device=logits.device)
        else:
            probs = None
            if need_entropy and latent_entropy_tensor is not None:
                entropy = latent_entropy_tensor[logit_idx]
            else:
                entropy = torch.tensor(0.0, device=logits.device)

        if mix_rate is None and mix_entropy_threshold is not None:
            explore_mask = bool(entropy >= mix_entropy_threshold)

        # answer detection
        if answer_start_ids:
            req.latent_recent_tokens.append(int(next_token_id))
            req.latent_recent_tokens = req.latent_recent_tokens[-len(answer_start_ids) :]
            if len(req.latent_recent_tokens) == len(answer_start_ids):
                if req.latent_recent_tokens == list(answer_start_ids):
                    req.latent_answer_detected = True

        # force decode interval
        latent_active = not req.latent_answer_detected
        req.latent_step_count += 1
        latent_mask = latent_active

        use_mix = (
            latent_mask
            and explore_mask
            and mix_top_k is not None
            and mix_top_k > 0
            and ((use_neighbor_mix and use_precomputed_neighbor) or use_precomputed_topk)
        )
        gaussian_record_topk_indices = None
        gaussian_record_topk_probs = None
        if use_orthogonal_noise_mix:
            if latent_thinking_embeds is None:
                raise RuntimeError("latent embed missing; sampler path is required")
            if logits_output.vocab_embedding_weight is None or (logits_output.tp_world_size or 1) > 1:
                raise RuntimeError("latent orthogonal_noise requires vocab embedding weight and tp_size=1")
            if use_mix:
                req.latent_next_embed = latent_thinking_embeds[logit_idx].detach()
            else:
                req.latent_next_embed = logits_output.vocab_embedding_weight[next_token_id].detach()
            req.latent_next_topk_indices = None
            req.latent_next_topk_probs = None
        elif use_gaussian_embed_noise_mix:
            if use_mix:
                cand_indices = latent_topk_indices[logit_idx]
                cand_probs = latent_topk_probs[logit_idx]
            else:
                cand_indices = torch.tensor([int(next_token_id)], device=logits.device, dtype=torch.long)
                cand_probs = torch.tensor([1.0], device=logits.device, dtype=logits.dtype)

            cand_indices = cand_indices.to(device=logits.device, dtype=torch.long).view(-1)
            cand_probs = cand_probs.to(device=logits.device, dtype=logits.dtype).view(-1)
            cand_probs = cand_probs / cand_probs.sum().clamp_min(1e-9)
            cand_embeds = _gather_vocab_rows(cand_indices)
            latent_embed = torch.matmul(cand_probs.to(cand_embeds.dtype), cand_embeds)
            if noise_rms_scale > 0.0 and (noise_on_eval or not is_validate):
                embed_rms = torch.sqrt(torch.mean(latent_embed.float().pow(2))).clamp(min=1e-6)
                noise_std = (noise_rms_scale * embed_rms).to(latent_embed.dtype)
                latent_embed = latent_embed + torch.randn_like(latent_embed) * noise_std

            req.latent_next_embed = latent_embed.detach()
            req.latent_next_topk_indices = None
            req.latent_next_topk_probs = None
            gaussian_record_topk_indices = cand_indices.detach()
            gaussian_record_topk_probs = cand_probs.detach()
        else:
            if use_mix:
                if use_neighbor_mix:
                    cand_indices = latent_neighbor_indices[logit_idx]
                    cand_probs = latent_neighbor_probs[logit_idx]
                else:
                    cand_indices = latent_topk_indices[logit_idx]
                    cand_probs = latent_topk_probs[logit_idx]
                req.latent_next_topk_indices = cand_indices.detach()
                req.latent_next_topk_probs = cand_probs.detach()
            else:
                req.latent_next_topk_indices = torch.tensor(
                    [int(next_token_id)], device=logits.device, dtype=torch.long
                )
                req.latent_next_topk_probs = torch.tensor(
                    [1.0], device=logits.device, dtype=logits.dtype
                )

        mask_val = latent_mask if mix_rate is None else (latent_mask and explore_mask)
        if use_orthogonal_noise_mix or use_gaussian_embed_noise_mix:
            req.latent_next_mask = bool(mask_val)
        else:
            req.latent_next_mask = None
        req.latent_thinking_mask.append(bool(mask_val))
        req.latent_mix_explore_mask.append(bool(explore_mask))
        req.latent_entropy.append(float(entropy))
        if use_orthogonal_noise_mix:
            req.latent_thinking_embeds.append(req.latent_next_embed.detach().cpu().tolist())
        elif use_gaussian_embed_noise_mix:
            req.latent_topk_indices_list.append(gaussian_record_topk_indices.detach().cpu().tolist())
            req.latent_topk_probs_list.append(gaussian_record_topk_probs.detach().cpu().tolist())
        if req.latent_next_topk_indices is not None and req.latent_next_topk_probs is not None:
            req.latent_topk_indices_list.append(req.latent_next_topk_indices.detach().cpu().tolist())
            req.latent_topk_probs_list.append(req.latent_next_topk_probs.detach().cpu().tolist())

        # disable cached embed state
        req.latent_state_prev = None
        if not (use_orthogonal_noise_mix or use_gaussian_embed_noise_mix):
            req.latent_next_embed = None

    def _apply_hrpo_for_req(
        self,
        req: Req,
        logits_output: LogitsProcessorOutput,
        logit_idx: int,
        next_token_id: int,
    ) -> None:
        """Compute HRPO latent embedding/mask for next step and stash in req buffers."""
        cfg = req.latent_cfg or {}
        answer_start_ids = cfg.get("hrpo_answer_start_ids", [])

        if answer_start_ids:
            req.latent_recent_tokens.append(int(next_token_id))
            req.latent_recent_tokens = req.latent_recent_tokens[-len(answer_start_ids) :]
            if len(req.latent_recent_tokens) == len(answer_start_ids):
                if req.latent_recent_tokens == list(answer_start_ids):
                    req.latent_answer_detected = True

        hrpo_active = not req.latent_answer_detected
        req.latent_step_count += 1

        latent_thinking_embeds = getattr(logits_output, "latent_thinking_embeds", None)
        if latent_thinking_embeds is None:
            raise RuntimeError("HRPO thinking embeds missing; sampler path is required")

        req.latent_next_embed = latent_thinking_embeds[logit_idx].detach()
        req.latent_next_mask = hrpo_active
        req.latent_thinking_embeds.append(req.latent_next_embed.detach().cpu().tolist())
        req.latent_thinking_mask.append(bool(hrpo_active))

    def _get_vocab_norm(self, vocab_weight: torch.Tensor) -> torch.Tensor:
        """Cache normalized vocab embeddings on the worker to avoid per-token recompute."""
        cache = getattr(self, "_latent_vocab_norm_cache", None)
        version = getattr(vocab_weight, "_version", None)
        key = (id(vocab_weight), version, vocab_weight.dtype, vocab_weight.device)
        if isinstance(cache, dict) and cache.get("key") == key:
            return cache["norm"]
        with torch.no_grad():
            vocab_norm = torch.nn.functional.normalize(vocab_weight, dim=-1)
        self._latent_vocab_norm_cache = {"key": key, "norm": vocab_norm}
        return vocab_norm

    def process_batch_result_prefill(
        self: Scheduler,
        batch: ScheduleBatch,
        result: Union[GenerationBatchResult, EmbeddingBatchResult],
        launch_done: Optional[threading.Event] = None,
    ):
        skip_stream_req = None

        if self.is_generation:
            (
                logits_output,
                next_token_ids,
                extend_input_len_per_req,
                extend_logprob_start_len_per_req,
            ) = (
                result.logits_output,
                result.next_token_ids,
                result.extend_input_len_per_req,
                result.extend_logprob_start_len_per_req,
            )

            if self.enable_overlap:
                logits_output, next_token_ids, _ = (
                    self.tp_worker.resolve_last_batch_result(launch_done)
                )
            else:
                # Move next_token_ids and logprobs to cpu
                next_token_ids = next_token_ids.tolist()
                if batch.return_logprob:
                    if logits_output.next_token_logprobs is not None:
                        logits_output.next_token_logprobs = (
                            logits_output.next_token_logprobs.tolist()
                        )
                    if logits_output.input_token_logprobs is not None:
                        logits_output.input_token_logprobs = tuple(
                            logits_output.input_token_logprobs.tolist()
                        )

            hidden_state_offset = 0

            # Check finish conditions
            logprob_pt = 0
            for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
                if req.is_retracted:
                    continue

                if self.is_mixed_chunk and self.enable_overlap and req.finished():
                    # Free the one delayed token for the mixed decode batch
                    j = len(batch.out_cache_loc) - len(batch.reqs) + i
                    self.token_to_kv_pool_allocator.free(batch.out_cache_loc[j : j + 1])
                    continue

                if req.is_chunked <= 0:
                    # req output_ids are set here
                    req.output_ids.append(next_token_id)
                    req.check_finished()

                    if req.finished():
                        self.tree_cache.cache_finished_req(req)
                        req.time_stats.completion_time = time.time()
                    elif not batch.decoding_reqs or req not in batch.decoding_reqs:
                        # This updates radix so others can match
                        self.tree_cache.cache_unfinished_req(req)

                    # Latent rollout mixing for prefill-stage generated token
                    latent_cfg = getattr(req, "latent_cfg", None)
                    if latent_cfg and latent_cfg.get("latent_rollout_enable", False):
                        self._apply_latent_mix_for_req(
                            req=req,
                            logits_output=logits_output,
                            logit_idx=i,
                            next_token_id=next_token_id,
                            is_validate=latent_cfg.get("latent_is_validate", False),
                        )
                    elif latent_cfg and latent_cfg.get("hrpo_enable", False):
                        self._apply_hrpo_for_req(
                            req=req,
                            logits_output=logits_output,
                            logit_idx=i,
                            next_token_id=next_token_id,
                        )

                    if req.return_logprob:
                        assert extend_logprob_start_len_per_req is not None
                        assert extend_input_len_per_req is not None
                        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                        extend_input_len = extend_input_len_per_req[i]
                        num_input_logprobs = extend_input_len - extend_logprob_start_len
                        self.add_logprob_return_values(
                            i,
                            req,
                            logprob_pt,
                            next_token_ids,
                            num_input_logprobs,
                            logits_output,
                        )
                        logprob_pt += num_input_logprobs

                    if (
                        req.return_hidden_states
                        and logits_output.hidden_states is not None
                    ):
                        req.hidden_states.append(
                            logits_output.hidden_states[
                                hidden_state_offset : (
                                    hidden_state_offset := hidden_state_offset
                                    + len(req.origin_input_ids)
                                )
                            ]
                            .cpu()
                            .clone()
                            .tolist()
                        )

                    if req.grammar is not None:
                        req.grammar.accept_token(next_token_id)
                        req.grammar.finished = req.finished()
                else:
                    # being chunked reqs' prefill is not finished
                    req.is_chunked -= 1
                    # There is only at most one request being currently chunked.
                    # Because this request does not finish prefill,
                    # we don't want to stream the request currently being chunked.
                    skip_stream_req = req

                    # Incrementally update input logprobs.
                    if req.return_logprob:
                        extend_logprob_start_len = extend_logprob_start_len_per_req[i]
                        extend_input_len = extend_input_len_per_req[i]
                        if extend_logprob_start_len < extend_input_len:
                            # Update input logprobs.
                            num_input_logprobs = (
                                extend_input_len - extend_logprob_start_len
                            )
                            self.add_input_logprob_return_values(
                                i,
                                req,
                                logits_output,
                                logprob_pt,
                                num_input_logprobs,
                                last_prefill_chunk=False,
                            )
                            logprob_pt += num_input_logprobs

            self.set_next_batch_sampling_info_done(batch)

        else:  # embedding or reward model
            embeddings, bid = result.embeddings, result.bid
            embeddings = embeddings.tolist()

            # Check finish conditions
            for i, req in enumerate(batch.reqs):
                if req.is_retracted:
                    continue

                req.embedding = embeddings[i]
                if req.is_chunked <= 0:
                    # Dummy output token for embedding models
                    req.output_ids.append(0)
                    req.check_finished()

                    if req.finished():
                        self.tree_cache.cache_finished_req(req)
                    else:
                        self.tree_cache.cache_unfinished_req(req)
                else:
                    # being chunked reqs' prefill is not finished
                    req.is_chunked -= 1

        self.stream_output(batch.reqs, batch.return_logprob, skip_stream_req)

    def process_batch_result_decode(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        launch_done: Optional[threading.Event] = None,
    ):
        logits_output, next_token_ids, can_run_cuda_graph = (
            result.logits_output,
            result.next_token_ids,
            result.can_run_cuda_graph,
        )
        self.num_generated_tokens += len(batch.reqs)

        if self.enable_overlap:
            logits_output, next_token_ids, can_run_cuda_graph = (
                self.tp_worker.resolve_last_batch_result(launch_done)
            )
            next_token_logprobs = logits_output.next_token_logprobs
        elif batch.spec_algorithm.is_none():
            # spec decoding handles output logprobs inside verify process.
            next_token_ids = next_token_ids.tolist()
            if batch.return_logprob:
                next_token_logprobs = logits_output.next_token_logprobs.tolist()

        self.token_to_kv_pool_allocator.free_group_begin()

        # Check finish condition
        # NOTE: the length of reqs and next_token_ids don't match if it is spec decoding.
        # We should ignore using next_token_ids for spec decoding cases.
        for i, (req, next_token_id) in enumerate(zip(batch.reqs, next_token_ids)):
            if req.is_retracted:
                continue

            if self.enable_overlap and req.finished():
                # Free the one extra delayed token
                if self.page_size == 1:
                    self.token_to_kv_pool_allocator.free(batch.out_cache_loc[i : i + 1])
                else:
                    # Only free when the extra token is in a new page
                    if (
                        len(req.origin_input_ids) + len(req.output_ids) - 1
                    ) % self.page_size == 0:
                        self.token_to_kv_pool_allocator.free(
                            batch.out_cache_loc[i : i + 1]
                        )
                continue

            if batch.spec_algorithm.is_none():
                # speculative worker will solve the output_ids in speculative decoding
                req.output_ids.append(next_token_id)

            req.check_finished()
            if req.finished():
                self.tree_cache.cache_finished_req(req)
                req.time_stats.completion_time = time.time()

            if req.return_logprob and batch.spec_algorithm.is_none():
                # speculative worker handles logprob in speculative decoding
                req.output_token_logprobs_val.append(next_token_logprobs[i])
                req.output_token_logprobs_idx.append(next_token_id)
                if req.top_logprobs_num > 0:
                    req.output_top_logprobs_val.append(
                        logits_output.next_token_top_logprobs_val[i]
                    )
                    req.output_top_logprobs_idx.append(
                        logits_output.next_token_top_logprobs_idx[i]
                    )
                if req.token_ids_logprob is not None:
                        req.output_token_ids_logprobs_val.append(
                            logits_output.next_token_token_ids_logprobs_val[i]
                        )
                        req.output_token_ids_logprobs_idx.append(
                            logits_output.next_token_token_ids_logprobs_idx[i]
                        )

            # Latent rollout mixing (align with HF path)
            latent_cfg = getattr(req, "latent_cfg", None)
            if latent_cfg and latent_cfg.get("latent_rollout_enable", False):
                self._apply_latent_mix_for_req(
                    req=req,
                    logits_output=logits_output,
                    logit_idx=i,
                    next_token_id=next_token_id,
                    is_validate=latent_cfg.get("latent_is_validate", False),
                )
            elif latent_cfg and latent_cfg.get("hrpo_enable", False):
                self._apply_hrpo_for_req(
                    req=req,
                    logits_output=logits_output,
                    logit_idx=i,
                    next_token_id=next_token_id,
                )

            if req.return_hidden_states and logits_output.hidden_states is not None:
                req.hidden_states.append(
                    logits_output.hidden_states[i].cpu().clone().tolist()
                )

            if req.grammar is not None and batch.spec_algorithm.is_none():
                req.grammar.accept_token(next_token_id)
                req.grammar.finished = req.finished()

        self.set_next_batch_sampling_info_done(batch)
        self.stream_output(batch.reqs, batch.return_logprob)
        self.token_to_kv_pool_allocator.free_group_end()

        self.forward_ct_decode = (self.forward_ct_decode + 1) % (1 << 30)
        if (
            self.attn_tp_rank == 0
            and self.forward_ct_decode % self.server_args.decode_log_interval == 0
        ):
            self.log_decode_stats(can_run_cuda_graph, running_batch=batch)

    def add_input_logprob_return_values(
        self: Scheduler,
        i: int,
        req: Req,
        output: LogitsProcessorOutput,
        logprob_pt: int,
        num_input_logprobs: int,
        last_prefill_chunk: bool,  # If True, it means prefill is finished.
    ):
        """Incrementally add input logprobs to `req`.

        Args:
            i: The request index in a batch.
            req: The request. Input logprobs inside req are modified as a
                consequence of the API
            fill_ids: The prefill ids processed.
            output: Logit processor output that's used to compute input logprobs
            last_prefill_chunk: True if it is the last prefill (when chunked).
                Some of input logprob operation should only happen at the last
                prefill (e.g., computing input token logprobs).
        """
        assert output.input_token_logprobs is not None
        if req.input_token_logprobs is None:
            req.input_token_logprobs = []
        if req.temp_input_top_logprobs_val is None:
            req.temp_input_top_logprobs_val = []
        if req.temp_input_top_logprobs_idx is None:
            req.temp_input_top_logprobs_idx = []
        if req.temp_input_token_ids_logprobs_val is None:
            req.temp_input_token_ids_logprobs_val = []
        if req.temp_input_token_ids_logprobs_idx is None:
            req.temp_input_token_ids_logprobs_idx = []

        if req.input_token_logprobs_val is not None:
            # The input logprob has been already computed. It only happens
            # upon retract.
            if req.top_logprobs_num > 0:
                assert req.input_token_logprobs_val is not None
            return

        # Important for the performance.
        assert isinstance(output.input_token_logprobs, tuple)
        input_token_logprobs: Tuple[int] = output.input_token_logprobs
        input_token_logprobs = input_token_logprobs[
            logprob_pt : logprob_pt + num_input_logprobs
        ]
        req.input_token_logprobs.extend(input_token_logprobs)

        if req.top_logprobs_num > 0:
            req.temp_input_top_logprobs_val.append(output.input_top_logprobs_val[i])
            req.temp_input_top_logprobs_idx.append(output.input_top_logprobs_idx[i])

        if req.token_ids_logprob is not None:
            req.temp_input_token_ids_logprobs_val.append(
                output.input_token_ids_logprobs_val[i]
            )
            req.temp_input_token_ids_logprobs_idx.append(
                output.input_token_ids_logprobs_idx[i]
            )

        if last_prefill_chunk:
            input_token_logprobs = req.input_token_logprobs
            req.input_token_logprobs = None
            assert req.input_token_logprobs_val is None
            assert req.input_token_logprobs_idx is None
            assert req.input_top_logprobs_val is None
            assert req.input_top_logprobs_idx is None

            # Compute input_token_logprobs_val
            # Always pad the first one with None.
            req.input_token_logprobs_val = [None]
            req.input_token_logprobs_val.extend(input_token_logprobs)
            # The last input logprob is for sampling, so just pop it out.
            req.input_token_logprobs_val.pop()

            # Compute input_token_logprobs_idx
            input_token_logprobs_idx = req.origin_input_ids[req.logprob_start_len :]
            # Clip the padded hash values from image tokens.
            # Otherwise, it will lead to detokenization errors.
            input_token_logprobs_idx = [
                x if x < self.model_config.vocab_size - 1 else 0
                for x in input_token_logprobs_idx
            ]
            req.input_token_logprobs_idx = input_token_logprobs_idx

            if req.top_logprobs_num > 0:
                req.input_top_logprobs_val = [None]
                req.input_top_logprobs_idx = [None]
                assert len(req.temp_input_token_ids_logprobs_val) == len(
                    req.temp_input_token_ids_logprobs_idx
                )
                for val, idx in zip(
                    req.temp_input_top_logprobs_val,
                    req.temp_input_top_logprobs_idx,
                    strict=True,
                ):
                    req.input_top_logprobs_val.extend(val)
                    req.input_top_logprobs_idx.extend(idx)

                # Last token is a sample token.
                req.input_top_logprobs_val.pop()
                req.input_top_logprobs_idx.pop()
                req.temp_input_top_logprobs_idx = None
                req.temp_input_top_logprobs_val = None

            if req.token_ids_logprob is not None:
                req.input_token_ids_logprobs_val = [None]
                req.input_token_ids_logprobs_idx = [None]

                for val, idx in zip(
                    req.temp_input_token_ids_logprobs_val,
                    req.temp_input_token_ids_logprobs_idx,
                    strict=True,
                ):
                    req.input_token_ids_logprobs_val.extend(val)
                    req.input_token_ids_logprobs_idx.extend(idx)

                # Last token is a sample token.
                req.input_token_ids_logprobs_val.pop()
                req.input_token_ids_logprobs_idx.pop()
                req.temp_input_token_ids_logprobs_idx = None
                req.temp_input_token_ids_logprobs_val = None

            if req.return_logprob:
                relevant_tokens_len = len(req.origin_input_ids) - req.logprob_start_len
                assert len(req.input_token_logprobs_val) == relevant_tokens_len
                assert len(req.input_token_logprobs_idx) == relevant_tokens_len
                if req.top_logprobs_num > 0:
                    assert len(req.input_top_logprobs_val) == relevant_tokens_len
                    assert len(req.input_top_logprobs_idx) == relevant_tokens_len
                if req.token_ids_logprob is not None:
                    assert len(req.input_token_ids_logprobs_val) == relevant_tokens_len
                    assert len(req.input_token_ids_logprobs_idx) == relevant_tokens_len

    def add_logprob_return_values(
        self: Scheduler,
        i: int,
        req: Req,
        pt: int,
        next_token_ids: List[int],
        num_input_logprobs: int,
        output: LogitsProcessorOutput,
    ):
        """Attach logprobs to the return values."""
        req.output_token_logprobs_val.append(output.next_token_logprobs[i])
        req.output_token_logprobs_idx.append(next_token_ids[i])

        self.add_input_logprob_return_values(
            i, req, output, pt, num_input_logprobs, last_prefill_chunk=True
        )

        if req.top_logprobs_num > 0:
            req.output_top_logprobs_val.append(output.next_token_top_logprobs_val[i])
            req.output_top_logprobs_idx.append(output.next_token_top_logprobs_idx[i])

        if req.token_ids_logprob is not None:
            req.output_token_ids_logprobs_val.append(
                output.next_token_token_ids_logprobs_val[i]
            )
            req.output_token_ids_logprobs_idx.append(
                output.next_token_token_ids_logprobs_idx[i]
            )

        return num_input_logprobs

    def _resolve_vocab_weight(self):
        """Resolve vocab embedding weight from model if logits output doesn't carry it."""
        model = None
        tp_worker = getattr(self, "tp_worker", None)
        if tp_worker is not None:
            model_runner = getattr(tp_worker, "model_runner", None)
            if model_runner is None and hasattr(tp_worker, "worker"):
                model_runner = getattr(tp_worker.worker, "model_runner", None)
            model = getattr(model_runner, "model", None) if model_runner is not None else None
        head = None
        if model is not None:
            if hasattr(model, "get_embed_and_head"):
                try:
                    _, head = model.get_embed_and_head()
                except Exception:
                    head = None
            if head is None and hasattr(model, "lm_head"):
                head = model.lm_head
            if head is None and hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
                head = model.model.embed_tokens
        vocab_weight = None
        start_idx = None
        end_idx = None
        if isinstance(head, torch.Tensor):
            vocab_weight = head
        elif head is not None and hasattr(head, "weight"):
            vocab_weight = head.weight
            shard = getattr(head, "shard_indices", None)
            if shard is not None:
                start_idx = getattr(shard, "org_vocab_start_index", None)
                end_idx = getattr(shard, "org_vocab_end_index", None)
        tp_world = get_tensor_model_parallel_world_size()
        return vocab_weight, start_idx, end_idx, tp_world

    def stream_output(
        self: Scheduler,
        reqs: List[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
    ):
        """Stream the output to detokenizer."""
        if self.is_generation:
            self.stream_output_generation(reqs, return_logprob, skip_req)
        else:  # embedding or reward model
            self.stream_output_embedding(reqs)

    def stream_output_generation(
        self: Scheduler,
        reqs: List[Req],
        return_logprob: bool,
        skip_req: Optional[Req] = None,
    ):
        rids = []
        finished_reasons: List[BaseFinishReason] = []

        decoded_texts = []
        decode_ids_list = []
        read_offsets = []
        output_ids = []

        skip_special_tokens = []
        spaces_between_special_tokens = []
        no_stop_trim = []
        prompt_tokens = []
        completion_tokens = []
        cached_tokens = []
        spec_verify_ct = []
        output_hidden_states = None

        if return_logprob:
            input_token_logprobs_val = []
            input_token_logprobs_idx = []
            output_token_logprobs_val = []
            output_token_logprobs_idx = []
            input_top_logprobs_val = []
            input_top_logprobs_idx = []
            output_top_logprobs_val = []
            output_top_logprobs_idx = []
            input_token_ids_logprobs_val = []
            input_token_ids_logprobs_idx = []
            output_token_ids_logprobs_val = []
            output_token_ids_logprobs_idx = []
        else:
            input_token_logprobs_val = input_token_logprobs_idx = (
                output_token_logprobs_val
            ) = output_token_logprobs_idx = input_top_logprobs_val = (
                input_top_logprobs_idx
            ) = output_top_logprobs_val = output_top_logprobs_idx = (
                input_token_ids_logprobs_val
            ) = input_token_ids_logprobs_idx = output_token_ids_logprobs_val = (
                output_token_ids_logprobs_idx
            ) = None

        for req in reqs:
            if req is skip_req:
                continue

            # Multimodal partial stream chunks break the detokenizer, so drop aborted requests here.
            if self.model_config.is_multimodal_gen and req.to_abort:
                continue

            if req.finished():
                if req.finished_output:
                    # With the overlap schedule, a request will try to output twice and hit this line twice
                    # because of the one additional delayed token. This "continue" prevented the dummy output.
                    continue
                req.finished_output = True
                should_output = True
            else:
                if req.stream:
                    stream_interval = (
                        req.sampling_params.stream_interval or self.stream_interval
                    )
                    should_output = len(req.output_ids) % stream_interval == 0
                else:
                    should_output = (
                        len(req.output_ids) % DEFAULT_FORCE_STREAM_INTERVAL == 0
                        and not self.model_config.is_multimodal_gen
                    )

            if should_output:
                send_token_offset = req.send_token_offset
                send_output_token_logprobs_offset = (
                    req.send_output_token_logprobs_offset
                )
                rids.append(req.rid)
                finished_reasons.append(
                    req.finished_reason.to_json() if req.finished_reason else None
                )
                decoded_texts.append(req.decoded_text)
                decode_ids, read_offset = req.init_incremental_detokenize()

                if self.model_config.is_multimodal_gen:
                    decode_ids_list.append(decode_ids)
                else:
                    decode_ids_list.append(decode_ids[req.send_decode_id_offset :])

                req.send_decode_id_offset = len(decode_ids)
                read_offsets.append(read_offset)
                if self.skip_tokenizer_init:
                    output_ids.append(req.output_ids[send_token_offset:])
                req.send_token_offset = len(req.output_ids)
                skip_special_tokens.append(req.sampling_params.skip_special_tokens)
                spaces_between_special_tokens.append(
                    req.sampling_params.spaces_between_special_tokens
                )
                no_stop_trim.append(req.sampling_params.no_stop_trim)
                prompt_tokens.append(len(req.origin_input_ids))
                completion_tokens.append(len(req.output_ids))
                cached_tokens.append(req.cached_tokens)

                if not self.spec_algorithm.is_none():
                    spec_verify_ct.append(req.spec_verify_ct)

                if return_logprob:
                    if (
                        req.return_logprob
                        and not req.input_logprob_sent
                        # Decode server does not send input logprobs
                        and self.disaggregation_mode != DisaggregationMode.DECODE
                    ):
                        input_token_logprobs_val.append(req.input_token_logprobs_val)
                        input_token_logprobs_idx.append(req.input_token_logprobs_idx)
                        input_top_logprobs_val.append(req.input_top_logprobs_val)
                        input_top_logprobs_idx.append(req.input_top_logprobs_idx)
                        input_token_ids_logprobs_val.append(
                            req.input_token_ids_logprobs_val
                        )
                        input_token_ids_logprobs_idx.append(
                            req.input_token_ids_logprobs_idx
                        )
                        req.input_logprob_sent = True
                    else:
                        input_token_logprobs_val.append([])
                        input_token_logprobs_idx.append([])
                        input_top_logprobs_val.append([])
                        input_top_logprobs_idx.append([])
                        input_token_ids_logprobs_val.append([])
                        input_token_ids_logprobs_idx.append([])

                    if req.return_logprob:
                        output_token_logprobs_val.append(
                            req.output_token_logprobs_val[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_token_logprobs_idx.append(
                            req.output_token_logprobs_idx[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_top_logprobs_val.append(
                            req.output_top_logprobs_val[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_top_logprobs_idx.append(
                            req.output_top_logprobs_idx[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_token_ids_logprobs_val.append(
                            req.output_token_ids_logprobs_val[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        output_token_ids_logprobs_idx.append(
                            req.output_token_ids_logprobs_idx[
                                send_output_token_logprobs_offset:
                            ]
                        )
                        req.send_output_token_logprobs_offset = len(
                            req.output_token_logprobs_val
                        )
                    else:
                        output_token_logprobs_val.append([])
                        output_token_logprobs_idx.append([])
                        output_top_logprobs_val.append([])
                        output_top_logprobs_idx.append([])
                        output_token_ids_logprobs_val.append([])
                        output_token_ids_logprobs_idx.append([])

                if req.return_hidden_states:
                    if output_hidden_states is None:
                        output_hidden_states = []
                    output_hidden_states.append(req.hidden_states)

            if (
                req.finished()
                and self.tp_rank == 0
                and self.server_args.enable_request_time_stats_logging
            ):
                req.log_time_stats()

        # Send to detokenizer
        if rids:
            if self.model_config.is_multimodal_gen:
                return

            latent_thinking_embeds = []
            latent_thinking_mask = []
            latent_embeds_ratio = []
            latent_entropy = []
            latent_mix_explore_mask = []
            latent_topk_probs = []
            latent_topk_indices = []
            for req in reqs:
                if getattr(req, "latent_thinking_embeds", None) is not None:
                    embeds = []
                    for item in getattr(req, "latent_thinking_embeds", []):
                        embeds.append(item.tolist() if hasattr(item, "tolist") else item)
                    latent_thinking_embeds.append(embeds)
                else:
                    latent_thinking_embeds.append([])
                latent_thinking_mask.append(getattr(req, "latent_thinking_mask", []))
                latent_embeds_ratio.append(getattr(req, "latent_embeds_ratio", []))
                latent_entropy.append(getattr(req, "latent_entropy", []))
                latent_mix_explore_mask.append(getattr(req, "latent_mix_explore_mask", []))
                latent_topk_probs.append(getattr(req, "latent_topk_probs_list", []))
                latent_topk_indices.append(getattr(req, "latent_topk_indices_list", []))

            self.send_to_detokenizer.send_pyobj(
                BatchTokenIDOut(
                    rids=rids,
                    finished_reasons=finished_reasons,
                    decoded_texts=decoded_texts,
                    decode_ids=decode_ids_list,
                    read_offsets=read_offsets,
                    output_ids=output_ids,
                    skip_special_tokens=skip_special_tokens,
                    spaces_between_special_tokens=spaces_between_special_tokens,
                    no_stop_trim=no_stop_trim,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cached_tokens=cached_tokens,
                    spec_verify_ct=spec_verify_ct,
                    input_token_logprobs_val=input_token_logprobs_val,
                    input_token_logprobs_idx=input_token_logprobs_idx,
                    output_token_logprobs_val=output_token_logprobs_val,
                    output_token_logprobs_idx=output_token_logprobs_idx,
                    input_top_logprobs_val=input_top_logprobs_val,
                    input_top_logprobs_idx=input_top_logprobs_idx,
                    output_top_logprobs_val=output_top_logprobs_val,
                    output_top_logprobs_idx=output_top_logprobs_idx,
                    input_token_ids_logprobs_val=input_token_ids_logprobs_val,
                    input_token_ids_logprobs_idx=input_token_ids_logprobs_idx,
                    output_token_ids_logprobs_val=output_token_ids_logprobs_val,
                    output_token_ids_logprobs_idx=output_token_ids_logprobs_idx,
                    output_hidden_states=output_hidden_states,
                    latent_thinking_embeds=latent_thinking_embeds,
                    latent_thinking_mask=latent_thinking_mask,
                    latent_embeds_ratio=latent_embeds_ratio,
                    latent_entropy=latent_entropy,
                    latent_mix_explore_mask=latent_mix_explore_mask,
                    latent_topk_probs=latent_topk_probs,
                    latent_topk_indices=latent_topk_indices,
                )
            )

    def stream_output_embedding(self: Scheduler, reqs: List[Req]):
        rids = []
        finished_reasons: List[BaseFinishReason] = []

        embeddings = []
        prompt_tokens = []
        cached_tokens = []
        for req in reqs:
            if req.finished():
                rids.append(req.rid)
                finished_reasons.append(req.finished_reason.to_json())
                embeddings.append(req.embedding)
                prompt_tokens.append(len(req.origin_input_ids))
                cached_tokens.append(req.cached_tokens)
        self.send_to_detokenizer.send_pyobj(
            BatchEmbeddingOut(
                rids, finished_reasons, embeddings, prompt_tokens, cached_tokens
            )
        )
