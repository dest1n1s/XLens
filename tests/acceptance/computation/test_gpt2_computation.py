import jax.numpy as jnp
import pytest

from xlens import HookedTransformer
from xlens.pretrained import get_pretrained_model_config, get_pretrained_state_dict
from xlens.utils import load_pretrained_weights

pytest.importorskip("torch")

import torch  # noqa: E402
from transformers import GPT2LMHeadModel, GPT2Tokenizer  # noqa: E402


@torch.no_grad()
def test_get_pretrained_state_dict():
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    hf_model.eval()
    cfg = get_pretrained_model_config("gpt2")
    state_dict = get_pretrained_state_dict("gpt2", cfg)
    model = HookedTransformer(cfg)
    model = load_pretrained_weights(model, state_dict)

    hf_input = tokenizer("Hello, my dog is cute", return_tensors="pt")["input_ids"]
    assert hf_input.shape == (1, 6)
    input = jnp.array(hf_input)

    hf_output = hf_model(hf_input).logits
    output = model(input)

    assert jnp.allclose(output, jnp.array(hf_output), atol=1e-4)
