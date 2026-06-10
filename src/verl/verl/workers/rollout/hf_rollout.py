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
Rollout with huggingface models.
TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single
GPU model. Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model
to perform generation.
"""

import contextlib

import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from verl import DataProto
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.torch_functional import get_response_mask

from .base import BaseRollout

__all__ = ["HFRollout"]


class HFRollout(BaseRollout):
    def __init__(self, module: nn.Module, config):
        super().__init__()
        self.config = config
        self.module = module

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = [self._generate_minibatch(p) for p in batch_prompts]
        output = DataProto.concat(output)
        return output

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        if getattr(self.config, "latent_rollout_enable", False) and getattr(self.config, "hrpo_enable", False):
            raise ValueError("latent_rollout_enable and hrpo_enable cannot be enabled at the same time.")
        if getattr(self.config, "latent_rollout_enable", False):
            if (
                getattr(self.config, "latent_rollout_disable_mix_on_eval", False)
                and bool(prompts.meta_info.get("validate", False))
            ):
                return self._generate_minibatch_standard(prompts)
            return self._generate_minibatch_latent_rollout(prompts)
        if getattr(self.config, "hrpo_enable", False):
            return self._generate_minibatch_hrpo(prompts)
        return self._generate_minibatch_standard(prompts)

    def _get_effective_num_return_sequences(self, prompts: DataProto, do_sample: bool, is_validate: bool) -> int:
        """根据上游 repeat 情况推断真实需要的返回条数。"""
        if not do_sample or is_validate:
            return 1

        target_n = int(getattr(self.config, "n", 1) or 1)
        repeat_times = prompts.meta_info.get("rollout_repeat_times", 1)
        try:
            repeat_times = int(repeat_times)
        except (TypeError, ValueError):
            repeat_times = 1

        if repeat_times <= 0:
            repeat_times = 1

        if target_n >= repeat_times:
            return max(1, target_n // repeat_times)

        return 1

    @torch.no_grad()
    def _generate_minibatch_standard(self, prompts: DataProto) -> DataProto:
        # make sampling args can be overridden by inputs
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompts.meta_info.get("validate", False)

        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))  # to be compatible with vllm

        effective_n = self._get_effective_num_return_sequences(prompts, do_sample, is_validate)

        if not do_sample:
            # do_sample==False -> greedy decoding
            kwargs = {
                "do_sample": False,
                "num_beams": 1,
            }
        elif is_validate:
            # do validate and do sample -> use val_kwargs
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_k": max(0, self.config.val_kwargs.top_k),  # to be compatible with vllm
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "num_return_sequences": 1,  # if validate, already repeat in ray_trainer
            }
        else:
            # do_sample -> use rollout config, prompts already repeated upstream
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "num_return_sequences": effective_n,
            }

        # make config according to generate mode
        generation_config = GenerationConfig(**kwargs)

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        prompt_length = idx.size(1)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]

        self.module.eval()
        param_ctx = contextlib.nullcontext()

        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        with param_ctx, torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
            output = self.module.generate(
                input_ids=idx,
                attention_mask=attention_mask,
                position_ids=position_ids,
                do_sample=do_sample,
                max_new_tokens=response_length,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                generation_config=generation_config,
                output_scores=False,  # this is potentially very large
                return_dict_in_generate=True,
                use_cache=True,
            )

        # TODO: filter out the seq with no answers like ds-chat
        seq = output.sequences
        generated_batch_size = seq.size(0)  # bs * num_return_sequences

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        sequence_length = prompt_length + self.config.response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(size=(generated_batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)
        assert seq.shape[1] == sequence_length

        # make necessary reputations if num_return_sequences > 1
        if effective_n > 1:
            position_ids = position_ids.repeat_interleave(effective_n, dim=0)
            attention_mask = attention_mask.repeat_interleave(effective_n, dim=0)

        prompt = seq[:, :prompt_length]  # (generated_batch_size, prompt_length)
        response = seq[:, prompt_length:]  # (generated_batch_size, response_length)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)

        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=generated_batch_size,
        )

        # empty cache before compute old_log_prob
        get_torch_device().empty_cache()

        self.module.train()
        return DataProto(batch=batch)

    def _hrpo_build_logits_warpers(self, do_sample: bool, temperature: float, top_k: int, top_p: float) -> LogitsProcessorList:
        warpers = LogitsProcessorList()
        if do_sample:
            if temperature is not None:
                temperature = float(temperature)
                if temperature <= 0.0:
                    temperature = 1e-6
            if temperature is not None and abs(temperature - 1.0) > 1e-6:
                warpers.append(TemperatureLogitsWarper(temperature))
            if top_k > 0:
                warpers.append(TopKLogitsWarper(top_k))
            if 0.0 < top_p < 1.0:
                warpers.append(TopPLogitsWarper(top_p))
        return warpers

    @torch.no_grad()
    def _generate_minibatch_latent_rollout(self, prompts: DataProto) -> DataProto:
        model = self.module
        tokenizer = getattr(model, "latent_rollout_tokenizer", None) or getattr(model, "hrpo_tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "Latent rollout requires the model to expose a tokenizer under `latent_rollout_tokenizer`."
            )

        answer_start = getattr(
            self.config,
            "latent_rollout_answer_start",
            getattr(model, "answer_start", getattr(self.config, "hrpo_answer_start", "####")),
        )
        answer_token_ids = tokenizer(answer_start, add_special_tokens=False).input_ids
        if not answer_token_ids:
            raise ValueError(f"Tokenizer produced empty token ids for answer_start '{answer_start}'.")

        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompts.meta_info.get("validate", False)
        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))

        if is_validate:
            do_sample = self.config.val_kwargs.do_sample
            temperature = self.config.val_kwargs.temperature
            top_p = self.config.val_kwargs.top_p
            top_k = max(0, self.config.val_kwargs.top_k)

        warpers = self._hrpo_build_logits_warpers(do_sample, temperature, top_k, top_p)

        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        prompt_length = idx.size(1)
        device = idx.device

        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]
        answer_token_tensor = torch.tensor(answer_token_ids, device=device, dtype=idx.dtype)
        answer_token_length = answer_token_tensor.size(0)

        effective_n = self._get_effective_num_return_sequences(prompts, do_sample, is_validate)

        if effective_n > 1:
            idx = idx.repeat_interleave(effective_n, dim=0)
            attention_mask = attention_mask.repeat_interleave(effective_n, dim=0)
            position_ids = position_ids.repeat_interleave(effective_n, dim=0)

        batch_size = idx.size(0)
        max_total_len = prompt_length + response_length

        embedding_module = model.get_input_embeddings()
        embedding_weight = embedding_module.weight
        hidden_size = embedding_weight.size(1)

        # force-decode 已弃用：保持配置兼容但不再生效
        force_decode_interval = None
        force_decode_burn_in = 0
        mix_top_k = getattr(self.config, "latent_rollout_mix_top_k", None)
        if mix_top_k is not None:
            mix_top_k = int(mix_top_k)
            if mix_top_k <= 0:
                mix_top_k = None
        mix_temperature = float(getattr(self.config, "latent_rollout_mix_temperature", 1.0))
        mix_temperature = max(mix_temperature, 1e-6)
        mix_rate = getattr(self.config, "latent_rollout_mix_rate", None)
        if mix_rate is not None:
            mix_rate = float(mix_rate)
            mix_rate = min(max(mix_rate, 0.0), 1.0)
        mix_entropy_threshold = getattr(self.config, "latent_rollout_mix_entropy_threshold", None)
        if mix_entropy_threshold is not None:
            mix_entropy_threshold = max(float(mix_entropy_threshold), 0.0)
        mix_strategy = getattr(self.config, "latent_rollout_mix_strategy", None)
        allowed_strategies = {
            "topk",
            "topk-gumbel",
            "neighbor",
            "orthogonal_noise",
            "gaussian_embed_noise",
        }
        if mix_strategy not in allowed_strategies:
            raise ValueError(
                "latent_rollout_mix_strategy must be explicitly set to one of: "
                "topk, topk-gumbel, neighbor, orthogonal_noise, gaussian_embed_noise"
            )
        use_flip_mix = False
        use_neighbor_mix = mix_strategy == "neighbor"
        use_orthogonal_noise_mix = mix_strategy == "orthogonal_noise"
        use_gaussian_embed_noise_mix = mix_strategy == "gaussian_embed_noise"
        disable_eval_mix = bool(getattr(self.config, "latent_rollout_disable_mix_on_eval", False))
        noise_rms_scale = max(float(getattr(self.config, "latent_rollout_noise_rms_scale", 0.33)), 0.0)
        noise_on_eval = bool(getattr(self.config, "latent_rollout_noise_on_eval", False))
        neighbor_metric = getattr(self.config, "latent_rollout_neighbor_metric", "cosine")
        if neighbor_metric is None:
            neighbor_metric = "cosine"
        neighbor_metric = str(neighbor_metric).lower()
        if use_neighbor_mix:
            allowed_metrics = {"cosine", "dot", "l1", "l2"}
            if neighbor_metric not in allowed_metrics:
                raise ValueError(
                    "latent_rollout_neighbor_metric must be one of: cosine, dot, l1, l2"
                )
        if mix_top_k is None or mix_top_k <= 0:
            raise ValueError("latent_rollout_mix_top_k must be > 0 when latent rollout is enabled.")
        use_topk_mix = mix_top_k is not None
        use_gumbel_mix = mix_strategy == "topk-gumbel" and not is_validate
        use_hidden_state_path = not use_topk_mix

        thinking_embeds = torch.zeros(
            (batch_size, max_total_len, hidden_size), device=device, dtype=embedding_weight.dtype
        )
        thinking_mask = torch.zeros((batch_size, max_total_len), device=device, dtype=torch.bool)
        embeds_ratio = torch.zeros((batch_size, max_total_len), device=device, dtype=embedding_weight.dtype)
        latent_entropy = None
        latent_mix_explore_mask = torch.zeros((batch_size, max_total_len), device=device, dtype=torch.bool)
        latent_mix_topk_indices = (
            torch.full((batch_size, max_total_len, mix_top_k), -1, device=device, dtype=torch.int64)
            if use_topk_mix and mix_top_k is not None
            else None
        )

        answer_detected = torch.zeros(batch_size, dtype=torch.bool, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        latent_steps_since_text = None

        model.eval()

        def _split_hidden_states(hidden_states):
            if hidden_states is None or len(hidden_states) == 0:
                raise RuntimeError("Latent rollout requires hidden_states output from the model.")

            ratio_meta = None
            mask_meta = None
            last_hidden = hidden_states[-1]

            if last_hidden.dim() == 2 and len(hidden_states) >= 3:
                ratio_meta = last_hidden
                mask_meta = hidden_states[-2]
                last_hidden = hidden_states[-3]

            return last_hidden, mask_meta, ratio_meta

        outputs = model(
            input_ids=idx,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
            output_hidden_states=use_hidden_state_path,
        )

        logits = outputs.logits[:, -1, :]
        latent_entropy = torch.zeros((batch_size, max_total_len), device=device, dtype=logits.dtype)
        past_key_values = outputs.past_key_values
        if use_hidden_state_path:
            hidden_state, _, _ = _split_hidden_states(outputs.hidden_states)
            latent_state_prev = hidden_state[:, -1, :].to(embedding_weight.dtype)
        else:
            latent_state_prev = None

        for step in range(response_length):
            warped_logits = logits
            if len(warpers) > 0:
                warped_logits = warpers(idx, warped_logits)

            probs = torch.softmax(warped_logits, dim=-1)
            if mix_rate is None and mix_entropy_threshold is not None:
                log_probs = torch.log(torch.clamp(probs, min=1e-9))
                entropy = -(probs * log_probs).sum(dim=-1)
            else:
                entropy = torch.zeros(batch_size, device=device, dtype=probs.dtype)
            mix_apply_mask = None
            if use_topk_mix or use_gaussian_embed_noise_mix:
                if disable_eval_mix and is_validate:
                    mix_apply_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
                elif mix_rate is not None:
                    mix_apply_mask = (
                        torch.zeros(batch_size, dtype=torch.bool, device=device)
                        if is_validate
                        else torch.rand(batch_size, device=device) < mix_rate
                    )
                elif mix_entropy_threshold is not None:
                    mix_apply_mask = entropy >= mix_entropy_threshold

            if use_topk_mix:
                top_k = None
                topk_indices = None
                if use_flip_mix:
                    # mix_top_k interpreted as: slot0=top1, remaining (mix_top_k-1) from flipped probs
                    flip_pick = max((mix_top_k or 3) - 1, 1)
                    top1_idx = torch.argmax(warped_logits, dim=-1)
                    top1_prob = probs.gather(-1, top1_idx.unsqueeze(-1)).squeeze(-1)
                    comp_probs = 1.0 - probs
                    # mask out top1 for selecting flipped top2
                    comp_masked = comp_probs.masked_fill(
                        torch.nn.functional.one_hot(top1_idx, num_classes=comp_probs.size(-1)).bool(),
                        float("-inf"),
                    )
                    flip_pick = min(flip_pick, comp_masked.size(-1))
                    top_vals, top_idx = torch.topk(comp_masked, k=flip_pick, dim=-1)
                    # gather flipped weights for candidates
                    top1_comp = (1.0 - top1_prob).unsqueeze(-1)
                    flip_weights = torch.cat([top1_comp, top_vals], dim=-1)
                    weight_denom = flip_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    weights = (flip_weights / weight_denom).to(embedding_weight.dtype)
                    cand_indices = torch.cat([top1_idx.unsqueeze(-1), top_idx], dim=-1)
                    cand_embeds = embedding_module(cand_indices).to(embedding_weight.dtype)
                    latent_state = torch.bmm(weights.unsqueeze(1), cand_embeds).squeeze(1)
                    top_k = cand_indices.size(-1)
                    topk_indices = cand_indices
                    if mix_apply_mask is not None and not torch.all(mix_apply_mask):
                        greedy_indices = torch.argmax(warped_logits, dim=-1)
                        greedy_embeds = embedding_module(greedy_indices).to(embedding_weight.dtype)
                        mask = mix_apply_mask.unsqueeze(-1)
                        latent_state = torch.where(mask, latent_state, greedy_embeds)
                elif use_neighbor_mix:
                    top1_idx = torch.argmax(warped_logits, dim=-1)
                    top1_prob = probs.gather(-1, top1_idx.unsqueeze(-1)).squeeze(-1)
                    top1_embed = embedding_module(top1_idx).to(embedding_weight.dtype)

                    if neighbor_metric == "cosine":
                        norm_emb = torch.nn.functional.normalize(embedding_weight, dim=-1)
                        top1_norm = torch.nn.functional.normalize(top1_embed, dim=-1)
                        sim = top1_norm @ norm_emb.transpose(0, 1)  # (bs, vocab)
                    elif neighbor_metric == "dot":
                        sim = top1_embed @ embedding_weight.transpose(0, 1)  # (bs, vocab)
                    elif neighbor_metric == "l1":
                        sim = -(top1_embed.unsqueeze(1) - embedding_weight.unsqueeze(0)).abs().sum(dim=-1)  # (bs, vocab)
                    elif neighbor_metric == "l2":
                        vocab_sq_norm = (embedding_weight * embedding_weight).sum(dim=-1).unsqueeze(0)  # (1, vocab)
                        top1_sq_norm = (top1_embed * top1_embed).sum(dim=-1, keepdim=True)  # (bs, 1)
                        dot = top1_embed @ embedding_weight.transpose(0, 1)  # (bs, vocab)
                        sim = 2.0 * dot - top1_sq_norm - vocab_sq_norm
                    else:
                        raise RuntimeError(
                            f"Unexpected latent_rollout_neighbor_metric: {neighbor_metric}"
                        )
                    gather_mask = torch.zeros_like(sim, dtype=torch.bool)
                    gather_mask.scatter_(1, top1_idx.unsqueeze(-1), True)
                    sim = sim.masked_fill(gather_mask, -float("inf"))

                    neighbor_k = max((mix_top_k or 1) - 1, 0)
                    neighbor_k = min(neighbor_k, sim.size(1))
                    if neighbor_k > 0:
                        _, neighbor_idx = torch.topk(sim, k=neighbor_k, dim=-1)
                        neighbor_probs = probs.gather(-1, neighbor_idx)
                        neighbor_embeds = embedding_module(neighbor_idx).to(embedding_weight.dtype)
                        cand_probs = torch.cat([top1_prob.unsqueeze(-1), neighbor_probs], dim=-1)
                        weights = cand_probs / cand_probs.sum(dim=-1, keepdim=True)
                        cand_embeds = torch.cat(
                            [top1_embed.unsqueeze(1), neighbor_embeds], dim=1
                        )  # (bs, k, hidden)
                        latent_state = torch.sum(weights.unsqueeze(-1) * cand_embeds, dim=1)
                        topk_indices = torch.cat([top1_idx.unsqueeze(-1), neighbor_idx], dim=-1)
                        top_k = topk_indices.size(-1)
                    else:
                        latent_state = top1_embed
                    if mix_apply_mask is not None and not torch.all(mix_apply_mask):
                        greedy_indices = torch.argmax(warped_logits, dim=-1)
                        greedy_embeds = embedding_module(greedy_indices).to(embedding_weight.dtype)
                        mask = mix_apply_mask.unsqueeze(-1)
                        latent_state = torch.where(mask, latent_state, greedy_embeds)
                elif use_orthogonal_noise_mix:
                    top_k = min(mix_top_k or 1, warped_logits.size(-1))
                    scores, topk_indices = torch.topk(warped_logits, k=top_k, dim=-1)
                    topk_embeds = embedding_module(topk_indices).to(embedding_weight.dtype)
                    top1_embed = topk_embeds[:, 0, :]
                    top1_norm_sq = top1_embed.pow(2).sum(dim=-1, keepdim=True).clamp(min=1e-6)

                    if top_k > 1:
                        other_embeds = topk_embeds[:, 1:, :]
                        proj_coeff = (
                            (other_embeds * top1_embed.unsqueeze(1)).sum(dim=-1, keepdim=True)
                            / top1_norm_sq.unsqueeze(1)
                        )
                        orth_dirs = other_embeds - proj_coeff * top1_embed.unsqueeze(1)
                        dir_norm = orth_dirs.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                        dir_unit = orth_dirs / dir_norm

                        noise_weights = torch.softmax(scores[:, 1:] / mix_temperature, dim=-1).to(embedding_weight.dtype)
                        noise_std = noise_weights.unsqueeze(-1) * dir_norm
                        noise = torch.randn_like(dir_unit) * noise_std
                        noise_sum = noise.sum(dim=1)
                        latent_state = top1_embed + noise_sum
                    else:
                        latent_state = top1_embed

                    if mix_apply_mask is not None and not torch.all(mix_apply_mask):
                        greedy_indices = torch.argmax(warped_logits, dim=-1)
                        greedy_embeds = embedding_module(greedy_indices).to(embedding_weight.dtype)
                        mask = mix_apply_mask.unsqueeze(-1)
                        latent_state = torch.where(mask, latent_state, greedy_embeds)
                elif use_gaussian_embed_noise_mix:
                    top_k = min(mix_top_k or 1, warped_logits.size(-1))
                    if mix_apply_mask is not None and not torch.all(mix_apply_mask):
                        greedy_indices = torch.argmax(warped_logits, dim=-1)
                        greedy_embeds = embedding_module(greedy_indices).to(embedding_weight.dtype)
                        latent_state = greedy_embeds
                        topk_indices = None
                        if mix_apply_mask.any():
                            scores = warped_logits[mix_apply_mask] / mix_temperature
                            _, topk_indices_mix = torch.topk(scores, k=top_k, dim=-1)
                            selected_scores = torch.gather(scores, -1, topk_indices_mix)
                            weights = torch.softmax(selected_scores, dim=-1).to(embedding_weight.dtype)
                            topk_embeds = embedding_module(topk_indices_mix).to(embedding_weight.dtype)
                            latent_state_mix = torch.bmm(weights.unsqueeze(1), topk_embeds).squeeze(1)
                            if noise_rms_scale > 0.0 and (noise_on_eval or not is_validate):
                                latent_rms = torch.sqrt(
                                    torch.mean(latent_state_mix.float().pow(2), dim=-1, keepdim=True)
                                ).clamp(min=1e-6)
                                noise_std = (noise_rms_scale * latent_rms).to(latent_state_mix.dtype)
                                latent_state_mix = latent_state_mix + torch.randn_like(latent_state_mix) * noise_std
                            latent_state[mix_apply_mask] = latent_state_mix
                            topk_indices = torch.full(
                                (batch_size, top_k),
                                -1,
                                device=device,
                                dtype=topk_indices_mix.dtype,
                            )
                            topk_indices[mix_apply_mask] = topk_indices_mix
                    else:
                        scores = warped_logits / mix_temperature
                        _, topk_indices = torch.topk(scores, k=top_k, dim=-1)
                        selected_scores = torch.gather(scores, -1, topk_indices)
                        weights = torch.softmax(selected_scores, dim=-1).to(embedding_weight.dtype)
                        topk_embeds = embedding_module(topk_indices).to(embedding_weight.dtype)
                        latent_state = torch.bmm(weights.unsqueeze(1), topk_embeds).squeeze(1)
                        if noise_rms_scale > 0.0 and (noise_on_eval or not is_validate):
                            latent_rms = torch.sqrt(
                                torch.mean(latent_state.float().pow(2), dim=-1, keepdim=True)
                            ).clamp(min=1e-6)
                            noise_std = (noise_rms_scale * latent_rms).to(latent_state.dtype)
                            latent_state = latent_state + torch.randn_like(latent_state) * noise_std
                else:
                    top_k = min(mix_top_k or 1, warped_logits.size(-1))
                    if mix_apply_mask is not None and not torch.all(mix_apply_mask):
                        greedy_indices = torch.argmax(warped_logits, dim=-1)
                        greedy_embeds = embedding_module(greedy_indices).to(embedding_weight.dtype)
                        latent_state = greedy_embeds
                        topk_indices = None
                        if mix_apply_mask.any():
                            logits_mix = warped_logits[mix_apply_mask]
                            if use_gumbel_mix:
                                uniform = torch.rand(
                                    logits_mix.size(0),
                                    logits_mix.size(1),
                                    device=device,
                                    dtype=torch.float32,
                                )
                                uniform = uniform.clamp_(1e-6, 1 - 1e-6)
                                gumbel = -torch.log(-torch.log(uniform)).to(logits_mix.dtype)
                                scores = logits_mix / mix_temperature + gumbel
                            else:
                                # During validation we keep the deterministic top-k ordering.
                                scores = logits_mix / mix_temperature
                            _, topk_indices_mix = torch.topk(scores, k=top_k, dim=-1)
                            selected_logits = torch.gather(logits_mix, -1, topk_indices_mix)
                            weights = torch.softmax(selected_logits / mix_temperature, dim=-1).to(
                                embedding_weight.dtype
                            )
                            topk_embeds = embedding_module(topk_indices_mix).to(embedding_weight.dtype)
                            latent_state_mix = torch.bmm(weights.unsqueeze(1), topk_embeds).squeeze(1)
                            latent_state[mix_apply_mask] = latent_state_mix
                            topk_indices = torch.full(
                                (batch_size, top_k),
                                -1,
                                device=device,
                                dtype=topk_indices_mix.dtype,
                            )
                            topk_indices[mix_apply_mask] = topk_indices_mix
                    else:
                        if use_gumbel_mix:
                            uniform = torch.rand(
                                warped_logits.size(0),
                                warped_logits.size(1),
                                device=device,
                                dtype=torch.float32,
                            )
                            uniform = uniform.clamp_(1e-6, 1 - 1e-6)
                            gumbel = -torch.log(-torch.log(uniform)).to(warped_logits.dtype)
                            scores = warped_logits / mix_temperature + gumbel
                        else:
                            # During validation we keep the deterministic top-k ordering.
                            scores = warped_logits / mix_temperature
                        _, topk_indices = torch.topk(scores, k=top_k, dim=-1)
                        selected_logits = torch.gather(warped_logits, -1, topk_indices)
                        weights = torch.softmax(selected_logits / mix_temperature, dim=-1).to(
                            embedding_weight.dtype
                        )
                        topk_embeds = embedding_module(topk_indices).to(embedding_weight.dtype)
                        latent_state = torch.bmm(weights.unsqueeze(1), topk_embeds).squeeze(1)
            else:
                latent_state = latent_state_prev

            latent_state = latent_state.to(embedding_weight.dtype)

            if do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(warped_logits, dim=-1)
            if mix_apply_mask is not None and do_sample:
                greedy_tokens = torch.argmax(warped_logits, dim=-1)
                next_tokens = torch.where(mix_apply_mask, next_tokens, greedy_tokens)

            if not self.config.ignore_eos and not is_validate:
                next_tokens = torch.where(finished, torch.full_like(next_tokens, pad_token_id), next_tokens)

            if not self.config.ignore_eos:
                finished = finished | (next_tokens == eos_token_id)

            idx = torch.cat([idx, next_tokens.unsqueeze(-1)], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype)], dim=1
            )
            next_position_ids = position_ids[:, -1:] + 1
            position_ids = torch.cat([position_ids, next_position_ids], dim=1)

            pos = prompt_length + step
            latent_entropy[:, pos] = entropy.to(latent_entropy.dtype)
            if (use_topk_mix or use_gaussian_embed_noise_mix) and mix_apply_mask is not None:
                latent_mix_explore_mask[:, pos] = mix_apply_mask
                if latent_mix_topk_indices is not None and topk_indices is not None and top_k is not None:
                    active_mask = mix_apply_mask
                    if active_mask.any():
                        latent_mix_topk_indices[active_mask, pos, :top_k] = topk_indices[active_mask, :top_k]
            latent_active = ~answer_detected
            force_decode_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

            # force-decode 逻辑移除后，latent_mask 仅由 latent_active 和 mix gating 决定
            latent_mask = latent_active
            if mix_rate is not None and mix_apply_mask is not None:
                latent_mask = latent_mask & mix_apply_mask
            latent_mask = latent_mask.to(torch.bool)
            thinking_embeds[:, pos, :] = latent_state
            thinking_mask[:, pos] = latent_mask

            model_inputs = {
                "input_ids": next_tokens.unsqueeze(-1),
                "attention_mask": attention_mask,
                "position_ids": next_position_ids,
                "past_key_values": past_key_values,
                "use_cache": True,
                "return_dict": True,
                "output_hidden_states": use_hidden_state_path,
            }

            if latent_mask.any():
                base_embed = embedding_module(next_tokens).to(embedding_weight.dtype)
                mask_3d = latent_mask.view(-1, 1, 1)
                combined = torch.where(mask_3d, latent_state.unsqueeze(1), base_embed.unsqueeze(1))
                model_inputs["inputs_embeds"] = combined
                model_inputs.pop("input_ids", None)
                target_module = getattr(model, "_fsdp_wrapped_module", model)
                base_module = getattr(target_module, "model", target_module)
                setattr(base_module, "_latent_rollout_mask", latent_mask.unsqueeze(1))
                if use_topk_mix or use_gaussian_embed_noise_mix:
                    base_norm = base_embed.norm(dim=-1).clamp(min=1e-6)
                    delta_norm = (latent_state - base_embed).norm(dim=-1)
                    ratio = (delta_norm / base_norm).to(embeds_ratio.dtype)
                    embeds_ratio[:, pos] = torch.where(latent_mask, ratio, torch.zeros_like(ratio))
            else:
                target_module = getattr(model, "_fsdp_wrapped_module", model)
                base_module = getattr(target_module, "model", target_module)
                if hasattr(base_module, "_latent_rollout_mask"):
                    delattr(base_module, "_latent_rollout_mask")
                if use_topk_mix or use_gaussian_embed_noise_mix:
                    embeds_ratio[:, pos] = torch.zeros(batch_size, device=device, dtype=embeds_ratio.dtype)

            if (step + 1) >= answer_token_length:
                window_start = prompt_length + step + 1 - answer_token_length
                window_end = prompt_length + step + 1
                recent_tokens = idx[:, window_start:window_end]
                matches = torch.all(recent_tokens == answer_token_tensor, dim=1)
                answer_detected = answer_detected | (matches & ~answer_detected)

            outputs = model(**model_inputs)
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            if use_hidden_state_path:
                hidden_state, _, ratio_meta = _split_hidden_states(outputs.hidden_states)
                latent_state_prev = hidden_state[:, -1, :].to(embedding_weight.dtype)

                if ratio_meta is not None:
                    step_ratio = ratio_meta[:, -1].to(embedding_weight.dtype)
                    embeds_ratio[:, pos] = torch.where(latent_mask, step_ratio, torch.zeros_like(step_ratio))
                else:
                    embeds_ratio[:, pos] = torch.zeros(batch_size, device=device, dtype=embedding_weight.dtype)

        seq = idx
        generated_batch_size = seq.size(0)

        if seq.size(1) < max_total_len:
            pad_len = max_total_len - seq.size(1)
            pad_tokens = torch.full((generated_batch_size, pad_len), pad_token_id, device=device, dtype=seq.dtype)
            seq = torch.cat([seq, pad_tokens], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.zeros(generated_batch_size, pad_len, device=device, dtype=attention_mask.dtype)],
                dim=1,
            )
            position_ids = torch.cat(
                [position_ids, position_ids[:, -1:].expand(generated_batch_size, pad_len)],
                dim=1,
            )
            thinking_embeds = torch.nn.functional.pad(thinking_embeds, (0, 0, 0, pad_len), value=0.0)
            thinking_mask = torch.nn.functional.pad(thinking_mask, (0, pad_len), value=False)
            embeds_ratio = torch.nn.functional.pad(embeds_ratio, (0, pad_len), value=0.0)
            latent_mix_explore_mask = torch.nn.functional.pad(latent_mix_explore_mask, (0, pad_len), value=False)
            if latent_mix_topk_indices is not None:
                latent_mix_topk_indices = torch.nn.functional.pad(
                    latent_mix_topk_indices, (0, 0, 0, pad_len), value=-1
                )

        prompt = seq[:, :prompt_length]
        response = seq[:, prompt_length:prompt_length + response_length]
        response_length_actual = response.size(1)
        delta_position_id = torch.arange(1, response_length_actual + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)
        response_position_ids = position_ids[:, prompt_length - 1 : prompt_length] + delta_position_id
        position_ids = torch.cat([position_ids[:, :prompt_length], response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask[:, :prompt_length], response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq[:, : prompt_length + response_length],
                "attention_mask": attention_mask[:, : prompt_length + response_length],
                "position_ids": position_ids[:, : prompt_length + response_length],
                "thinking_embeds": thinking_embeds[:, : prompt_length + response_length, :],
                "thinking_mask": thinking_mask[:, : prompt_length + response_length],
                "embeds_ratio": embeds_ratio[:, : prompt_length + response_length],
                "latent_entropy": latent_entropy[:, : prompt_length + response_length],
                "latent_mix_explore_mask": latent_mix_explore_mask[:, : prompt_length + response_length],
            },
            batch_size=generated_batch_size,
        )
        if latent_mix_topk_indices is not None:
            batch["latent_mix_topk_indices"] = latent_mix_topk_indices[:, : prompt_length + response_length, :]

        get_torch_device().empty_cache()
        self.module.train()
        return DataProto(batch=batch)

    @torch.no_grad()
    def _generate_minibatch_hrpo(self, prompts: DataProto) -> DataProto:
        model = self.module
        tokenizer = getattr(model, "hrpo_tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("HRPO rollout requires the model to expose a tokenizer under `hrpo_tokenizer`.")

        answer_start = getattr(self.config, "hrpo_answer_start", getattr(model, "answer_start", "####"))
        answer_token_ids = tokenizer(answer_start, add_special_tokens=False).input_ids
        if not answer_token_ids:
            raise ValueError(f"Tokenizer produced empty token ids for answer_start '{answer_start}'.")

        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompts.meta_info.get("validate", False)
        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))

        if is_validate:
            do_sample = self.config.val_kwargs.do_sample
            temperature = self.config.val_kwargs.temperature
            top_p = self.config.val_kwargs.top_p
            top_k = max(0, self.config.val_kwargs.top_k)

        warpers = self._hrpo_build_logits_warpers(do_sample, temperature, top_k, top_p)

        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        prompt_length = idx.size(1)
        device = idx.device

        eos_token_id = prompts.meta_info["eos_token_id"]
        pad_token_id = prompts.meta_info["pad_token_id"]
        answer_token_tensor = torch.tensor(answer_token_ids, device=device, dtype=idx.dtype)
        answer_token_length = answer_token_tensor.size(0)

        effective_n = self._get_effective_num_return_sequences(prompts, do_sample, is_validate)

        if effective_n > 1:
            idx = idx.repeat_interleave(effective_n, dim=0)
            attention_mask = attention_mask.repeat_interleave(effective_n, dim=0)
            position_ids = position_ids.repeat_interleave(effective_n, dim=0)

        batch_size = idx.size(0)
        max_total_len = prompt_length + response_length

        embedding_module = model.get_input_embeddings()
        embedding_weight = embedding_module.weight
        hidden_size = embedding_weight.size(1)

        thinking_embeds = torch.zeros(
            (batch_size, max_total_len, hidden_size), device=device, dtype=embedding_weight.dtype
        )
        thinking_mask = torch.zeros((batch_size, max_total_len), device=device, dtype=torch.bool)
        embeds_ratio = torch.zeros((batch_size, max_total_len), device=device, dtype=embedding_weight.dtype)

        answer_detected = torch.zeros(batch_size, dtype=torch.bool, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        model.eval()
        param_ctx = contextlib.nullcontext()
        if isinstance(model, FSDP):
            param_ctx = FSDP.summon_full_params(model, writeback=False, recurse=False)

        with param_ctx:
            outputs = model(
                input_ids=idx,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )

            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values

            for step in range(response_length):
                warped_logits = logits
                if len(warpers) > 0:
                    warped_logits = warpers(idx, warped_logits)

                probs = torch.softmax(warped_logits, dim=-1)
                thinking_state = torch.matmul(probs.to(embedding_weight.dtype), embedding_weight)
                denom = torch.linalg.norm(probs, dim=-1, keepdim=True)
                denom = torch.clamp(denom, min=1e-6).to(thinking_state.dtype)
                thinking_state = thinking_state / denom

                if do_sample:
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(warped_logits, dim=-1)

                if not self.config.ignore_eos and not is_validate:
                    next_tokens = torch.where(finished, torch.full_like(next_tokens, pad_token_id), next_tokens)

                if not self.config.ignore_eos:
                    finished = finished | (next_tokens == eos_token_id)

                idx = torch.cat([idx, next_tokens.unsqueeze(-1)], dim=1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(batch_size, 1, device=device, dtype=attention_mask.dtype)], dim=1
                )
                next_position_ids = position_ids[:, -1:] + 1
                position_ids = torch.cat([position_ids, next_position_ids], dim=1)

                pos = prompt_length + step
                thinking_embeds[:, pos, :] = thinking_state
                thinking_mask[:, pos] = ~answer_detected

                if (step + 1) >= answer_token_length:
                    window_start = prompt_length + step + 1 - answer_token_length
                    window_end = prompt_length + step + 1
                    recent_tokens = idx[:, window_start:window_end]
                    matches = torch.all(recent_tokens == answer_token_tensor, dim=1)
                    answer_detected = answer_detected | (matches & ~answer_detected)

                model_inputs = {
                    "input_ids": next_tokens.unsqueeze(-1),
                    "attention_mask": attention_mask,
                    "position_ids": next_position_ids,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                    "return_dict": True,
                }

                if (~answer_detected).any():
                    model_inputs["inputs_embeds"] = thinking_state.unsqueeze(1)
                    model_inputs["thinking_mask"] = (~answer_detected).unsqueeze(1)

                outputs = model(**model_inputs)
                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                if outputs.hidden_states:
                    ratio_values = outputs.hidden_states[-1]
                    if ratio_values.dim() > 1:
                        ratio_values = ratio_values.squeeze(-1)
                    embeds_ratio[:, pos] = ratio_values.to(embeds_ratio.dtype)

        seq = idx
        generated_batch_size = seq.size(0)

        if seq.size(1) < max_total_len:
            pad_len = max_total_len - seq.size(1)
            pad_tokens = torch.full((generated_batch_size, pad_len), pad_token_id, device=device, dtype=seq.dtype)
            seq = torch.cat([seq, pad_tokens], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.zeros(generated_batch_size, pad_len, device=device, dtype=attention_mask.dtype)],
                dim=1,
            )
            position_ids = torch.cat(
                [position_ids, position_ids[:, -1:].expand(generated_batch_size, pad_len)],
                dim=1,
            )
            thinking_embeds = torch.nn.functional.pad(thinking_embeds, (0, 0, 0, pad_len), value=0.0)
            thinking_mask = torch.nn.functional.pad(thinking_mask, (0, pad_len), value=False)
            embeds_ratio = torch.nn.functional.pad(embeds_ratio, (0, pad_len), value=0.0)

        prompt = seq[:, :prompt_length]
        response = seq[:, prompt_length:prompt_length + response_length]
        response_length_actual = response.size(1)
        delta_position_id = torch.arange(1, response_length_actual + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(generated_batch_size, 1)
        response_position_ids = position_ids[:, prompt_length - 1 : prompt_length] + delta_position_id
        position_ids = torch.cat([position_ids[:, :prompt_length], response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask[:, :prompt_length], response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt,
                "responses": response,
                "input_ids": seq[:, : prompt_length + response_length],
                "attention_mask": attention_mask[:, : prompt_length + response_length],
                "position_ids": position_ids[:, : prompt_length + response_length],
                "thinking_embeds": thinking_embeds[:, : prompt_length + response_length, :],
                "thinking_mask": thinking_mask[:, : prompt_length + response_length],
                "embeds_ratio": embeds_ratio[:, : prompt_length + response_length],
            },
            batch_size=generated_batch_size,
        )

        get_torch_device().empty_cache()
        self.module.train()
        return DataProto(batch=batch)
