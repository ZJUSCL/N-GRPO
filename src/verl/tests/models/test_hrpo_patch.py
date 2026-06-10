import types

import pytest

torch = pytest.importorskip("torch")
from transformers.modeling_outputs import CausalLMOutputWithPast

from verl.models.hrpo import apply_hrpo_patch


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


def test_hrpo_forward_appends_hidden_state_metadata():
    model = ToyCausalLM()
    cfg = types.SimpleNamespace(
        enable=True,
        residual_r_min=0.9,
        residual_r_max=0.9,
        mix_constant=1.0,
        answer_start="####",
    )

    apply_hrpo_patch(model, hrpo_cfg=cfg)

    # Without HRPO mask there should be no hidden state augmentation.
    out_plain = model(input_ids=torch.tensor([[1, 2]]))
    assert out_plain.hidden_states == () or out_plain.hidden_states is None

    thinking_embeds = torch.zeros(1, 2, model.get_input_embeddings().embedding_dim)
    thinking_mask = torch.tensor([[True, False]])

    out_hrpo = model(
        input_ids=torch.tensor([[1, 2]]),
        inputs_embeds=thinking_embeds,
        thinking_mask=thinking_mask,
    )

    assert out_hrpo.hidden_states is not None
    mask_meta = out_hrpo.hidden_states[-2]
    ratio_meta = out_hrpo.hidden_states[-1]
    assert mask_meta.shape[:2] == (1, thinking_mask.shape[1])
    assert ratio_meta.shape == (1, thinking_mask.shape[1])
