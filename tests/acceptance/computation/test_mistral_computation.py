import jax
import jax.numpy as jnp
import pytest

from xlens import HookedTransformer
from xlens.pretrained import get_pretrained_model_config, get_pretrained_state_dict
from xlens.utils import load_pretrained_weights

pytest.importorskip("torch")

import torch  # noqa: E402
from transformers import AutoTokenizer, MistralForCausalLM  # noqa: E402


@torch.no_grad()
def test_mistral_computation():
    hf_model = MistralForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1", torch_dtype=torch.float32, attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    hf_model.eval()

    hf_input = tokenizer("Hello, my dog is cute!", return_tensors="pt")["input_ids"]
    hf_output = hf_model(hf_input)
    hf_logits = hf_output.logits

    del hf_model
    torch.cuda.empty_cache()

    cfg = get_pretrained_model_config("mistralai/Mistral-7B-v0.1")
    state_dict = get_pretrained_state_dict("mistralai/Mistral-7B-v0.1", cfg)
    model = HookedTransformer(cfg)
    model = load_pretrained_weights(model, state_dict)

    input = jnp.array(hf_input)
    logits = model(input)

    print("Logits Difference: ", jnp.linalg.norm(logits - jnp.array(hf_logits)))

    hf_probs = torch.nn.functional.softmax(hf_logits, dim=-1)
    probs = jax.nn.softmax(logits, axis=-1)

    print("Probs Difference: ", jnp.linalg.norm(probs - jnp.array(hf_probs)))

    assert jnp.allclose(probs, jnp.array(hf_probs), atol=1e-3)


if __name__ == "__main__":
    test_mistral_computation()
