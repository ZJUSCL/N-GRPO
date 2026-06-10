import types

import pytest

torch = pytest.importorskip("torch")
from tensordict import TensorDict
from transformers.modeling_outputs import CausalLMOutputWithPast

from verl import DataProto
from verl.models.latent_rollout import apply_latent_rollout_patch
from verl.workers.rollout.hf_rollout import HFRollout


def _make_embed_weight(dtype: torch.dtype) -> torch.Tensor:
    base = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]],
        dtype=torch.float32,
    )
    return base.to(dtype)


class _DummyTokenizer:
    def __init__(self, answer_start: str = "####") -> None:
        self._answer_start = answer_start

    def __call__(self, text: str, add_special_tokens: bool = False):
        if text != self._answer_start:
            raise ValueError("unexpected answer_start request")
        return types.SimpleNamespace(input_ids=[4])


class _ToyBackbone(torch.nn.Module):
    def __init__(self, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(4, 2, dtype=dtype)
        self.embed_tokens.weight.data.copy_(_make_embed_weight(dtype))
        self.config = types.SimpleNamespace(hidden_size=2)

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        batch, seq_len, _ = hidden.shape
        logits = torch.full(
            (batch, seq_len, self.embed_tokens.num_embeddings),
            -5.0,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        logits[..., 0] = 1.0
        logits[..., 1] = 0.5
        hidden_states = (hidden,)
        return CausalLMOutputWithPast(logits=logits, hidden_states=hidden_states, past_key_values=None)


class _ToyCausalLM(torch.nn.Module):
    def __init__(self, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.model = _ToyBackbone(dtype=dtype)
        self.latent_rollout_tokenizer = _DummyTokenizer()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class _SimpleConfig:
    def __init__(self):
        self.latent_rollout_enable = True
        self.latent_rollout_answer_start = "####"
        self.latent_rollout_noise_std = 0.0
        self.latent_rollout_noise_scale_by_norm = True
        self.latent_rollout_use_random_projection = False
        self.latent_rollout_projection_dim = None
        self.latent_rollout_randomize_each_step = False
        self.latent_rollout_seed = None
        self.latent_rollout_force_decode_interval = 2
        self.latent_rollout_force_decode_burn_in = 0
        self.latent_rollout_mix_top_k = 2
        self.latent_rollout_mix_temperature = 1.0
        self.latent_rollout_mix_strategy = "topk"
        self.latent_rollout_noise_rms_scale = 0.33
        self.latent_rollout_noise_on_eval = False
        self.do_sample = False
        self.temperature = 1.0
        self.response_length = 4
        self.top_p = 1.0
        self.top_k = 0
        self.ignore_eos = True
        self.n = 1
        self.val_kwargs = types.SimpleNamespace(top_k=0, top_p=1.0, temperature=0.0, do_sample=False)

    def get(self, key, default=None):
        return getattr(self, key, default)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_latent_rollout_topk_interpolation_dtype(dtype):
    model = _ToyCausalLM(dtype=dtype)
    apply_latent_rollout_patch(model, latent_cfg=types.SimpleNamespace(enable=True, answer_start="####"))

    config = _SimpleConfig()
    rollout = HFRollout(model, config)

    batch = TensorDict(
        {
            "input_ids": torch.tensor([[0]]),
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
            "position_ids": torch.zeros(1, 1, dtype=torch.long),
        },
        batch_size=(1,),
    )
    prompts = DataProto(batch=batch, meta_info={"eos_token_id": 3, "pad_token_id": 0, "do_sample": False})

    output = rollout._generate_minibatch_latent_rollout(prompts)
    thinking_embeds = output.batch["thinking_embeds"].to(torch.float32)
    embeds_ratio = output.batch["embeds_ratio"].to(torch.float32)
    thinking_mask = output.batch["thinking_mask"]

    # force_decode 逻辑已移除，思考段不再被周期性截断
    expected_mask = torch.tensor([[True, True, True, True]], dtype=torch.bool)
    torch.testing.assert_close(thinking_mask[:, 1:], expected_mask)

    weight = torch.softmax(torch.tensor([1.0, 0.5]), dim=0)
    embed_weight = _make_embed_weight(dtype=torch.float32)
    expected_latent = weight[0] * embed_weight[0] + weight[1] * embed_weight[1]
    tol = 1e-3 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(thinking_embeds[0, 1], expected_latent, atol=tol, rtol=tol)

    base_embed = embed_weight[0]
    expected_ratio = torch.linalg.norm(expected_latent - base_embed) / torch.linalg.norm(base_embed)
    torch.testing.assert_close(embeds_ratio[0, 1], torch.tensor(expected_ratio, dtype=torch.float32), atol=tol, rtol=tol)
    # 后续步骤仍保持 latent 混合，比例应为非负
    assert torch.all(embeds_ratio[0, 2:] >= 0)


def test_latent_rollout_orthogonal_noise_mix():
    torch.manual_seed(0)
    model = _ToyCausalLM(dtype=torch.float32)
    apply_latent_rollout_patch(model, latent_cfg=types.SimpleNamespace(enable=True, answer_start="####"))

    config = _SimpleConfig()
    config.latent_rollout_mix_top_k = 3
    config.latent_rollout_mix_strategy = "orthogonal_noise"
    rollout = HFRollout(model, config)

    batch = TensorDict(
        {
            "input_ids": torch.tensor([[0]]),
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
            "position_ids": torch.zeros(1, 1, dtype=torch.long),
        },
        batch_size=(1,),
    )
    prompts = DataProto(batch=batch, meta_info={"eos_token_id": 3, "pad_token_id": 0, "do_sample": False})

    rng_state = torch.random.get_rng_state()
    output = rollout._generate_minibatch_latent_rollout(prompts)
    torch.random.set_rng_state(rng_state)

    # 手动复刻一次正交噪声混合，确认与实现一致
    logits = torch.tensor([1.0, 0.5, -5.0, -5.0])
    scores, topk_indices = torch.topk(logits, k=3, dim=-1)
    embed_weight = _make_embed_weight(dtype=torch.float32)
    topk_embeds = embed_weight[topk_indices]
    top1_embed = topk_embeds[0]
    top1_norm_sq = torch.dot(top1_embed, top1_embed).clamp(min=1e-6)

    other_embeds = topk_embeds[1:]
    proj_coeff = torch.matmul(other_embeds, top1_embed) / top1_norm_sq
    orth_dirs = other_embeds - proj_coeff.unsqueeze(-1) * top1_embed
    dir_norm = torch.linalg.norm(orth_dirs, dim=-1).clamp(min=1e-6)
    dir_unit = orth_dirs / dir_norm.unsqueeze(-1)
    noise_weights = torch.softmax(scores[1:], dim=-1)
    noise = torch.randn_like(dir_unit) * (noise_weights * dir_norm).unsqueeze(-1)
    expected = top1_embed + noise.sum(dim=0)

    thinking_embeds = output.batch["thinking_embeds"].to(torch.float32)
    torch.testing.assert_close(thinking_embeds[0, 1], expected)


def test_latent_rollout_gaussian_embed_noise_without_noise_matches_soft_embed():
    model = _ToyCausalLM(dtype=torch.float32)
    apply_latent_rollout_patch(model, latent_cfg=types.SimpleNamespace(enable=True, answer_start="####"))

    config = _SimpleConfig()
    config.latent_rollout_mix_strategy = "gaussian_embed_noise"
    config.latent_rollout_mix_top_k = 4
    config.latent_rollout_noise_rms_scale = 0.0
    rollout = HFRollout(model, config)

    batch = TensorDict(
        {
            "input_ids": torch.tensor([[0]]),
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
            "position_ids": torch.zeros(1, 1, dtype=torch.long),
        },
        batch_size=(1,),
    )
    prompts = DataProto(batch=batch, meta_info={"eos_token_id": 3, "pad_token_id": 0, "do_sample": False})

    output = rollout._generate_minibatch_latent_rollout(prompts)
    thinking_embeds = output.batch["thinking_embeds"].to(torch.float32)
    thinking_mask = output.batch["thinking_mask"]

    logits = torch.tensor([1.0, 0.5, -5.0, -5.0], dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)
    expected = probs @ _make_embed_weight(dtype=torch.float32)
    torch.testing.assert_close(thinking_embeds[0, 1], expected, atol=1e-6, rtol=1e-6)
    assert bool(thinking_mask[0, 1])
    assert "latent_mix_topk_indices" in output.batch.keys()


def test_latent_rollout_neighbor_mix_shape_on_cpu():
    model = _ToyCausalLM(dtype=torch.float32)
    apply_latent_rollout_patch(model, latent_cfg=types.SimpleNamespace(enable=True, answer_start="####"))

    config = _SimpleConfig()
    config.latent_rollout_mix_strategy = "neighbor"
    config.latent_rollout_mix_top_k = 3
    config.latent_rollout_neighbor_metric = "cosine"
    rollout = HFRollout(model, config)

    batch = TensorDict(
        {
            "input_ids": torch.tensor([[0]]),
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
            "position_ids": torch.zeros(1, 1, dtype=torch.long),
        },
        batch_size=(1,),
    )
    prompts = DataProto(batch=batch, meta_info={"eos_token_id": 3, "pad_token_id": 0, "do_sample": False})

    output = rollout._generate_minibatch_latent_rollout(prompts)
    thinking_mask = output.batch["thinking_mask"]

    assert bool(thinking_mask[0, 1])
    assert "latent_mix_topk_indices" in output.batch.keys()
