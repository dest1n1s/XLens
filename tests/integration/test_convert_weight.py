import jax.numpy as jnp
import pytest

from xlens import HookedTransformer, HookedTransformerConfig
from xlens.pretrained.convert_weight import convert_gpt2_weights
from xlens.utils import load_pretrained_weights

pytest.importorskip("torch")

import torch  # noqa: E402
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402


@torch.no_grad()
def test_convert_gpt2_weight():
    # Initialize GPT-2 model with a minimal configuration
    hf_config = GPT2Config(
        vocab_size=16,
        n_positions=16,
        n_embd=8,
        n_inner=8,
        n_layer=2,
        n_head=2,
        d_head=4,
    )

    hf_model = GPT2LMHeadModel(hf_config)
    hf_model.eval()

    # Convert the model weights to a dictionary
    def flatten_dict(d, parent_key="", sep="."):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    params = flatten_dict(hf_model.state_dict())
    params = {k: jnp.array(v) for k, v in params.items()}

    # Convert the weights
    cfg = HookedTransformerConfig(
        d_vocab=16,
        d_model=8,
        n_ctx=16,
        n_layers=2,
        n_heads=2,
        d_head=4,
        act_fn="gelu_new",
    )
    converted_params = convert_gpt2_weights(params, cfg)

    model = HookedTransformer(cfg)
    model = load_pretrained_weights(model, converted_params)

    hf_input = torch.randint(0, 16, (1, 16))
    input = jnp.array(hf_input)
    output = model(input)

    hf_output = hf_model(hf_input)
    assert jnp.allclose(output, jnp.array(hf_output.logits), atol=1e-4)
