import types

import pytest

torch = pytest.importorskip("torch")
from transformers.modeling_outputs import CausalLMOutputWithPast

from verl.models.latent_rollout import apply_latent_rollout_patch


class ToyBackbone(torch.nn.Module):
    def __init__(self, vocab_size: int = 8, hidden_size: int = 4):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.config = types.SimpleNamespace(hidden_size=hidden_size)

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):  # noqa: D401
        if inputs_embeds is None:
            hidden = self.embed_tokens(input_ids)
        else:
            hidden = inputs_embeds
        logits = hidden.mean(dim=-1, keepdim=True).expand(-1, -1, self.embed_tokens.num_embeddings)
        return CausalLMOutputWithPast(logits=logits, hidden_states=None, past_key_values=None)


class ToyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyBackbone()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def test_latent_rollout_patch_adds_metadata_and_mixes_embeddings():
    model = ToyCausalLM()
    cfg = types.SimpleNamespace(enable=True, answer_start="####")

    apply_latent_rollout_patch(model, latent_cfg=cfg)

    input_ids = torch.tensor([[1, 2]])
    base_embeds = model.get_input_embeddings()(input_ids)
    latent_embeds = base_embeds.clone()
    latent_embeds[:, 0, :] += 1.0
    mask = torch.tensor([[True, False]])

    out = model(input_ids=input_ids, inputs_embeds=latent_embeds, thinking_mask=mask)

    assert out.hidden_states is not None
    mask_meta = out.hidden_states[-2]
    ratio_meta = out.hidden_states[-1]

    assert mask_meta.shape[:2] == mask.shape
    assert ratio_meta.shape == mask.shape

    expected_ratio = (
        torch.linalg.norm(latent_embeds[:, 0, :] - base_embeds[:, 0, :], dim=-1)
        / torch.linalg.norm(base_embeds[:, 0, :], dim=-1)
    )
    torch.testing.assert_close(ratio_meta[:, 0], expected_ratio)
    torch.testing.assert_close(ratio_meta[:, 1], torch.zeros_like(ratio_meta[:, 1]))
