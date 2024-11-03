import jax
import jax.numpy as jnp
import pytest

from xlens import HookedTransformer
from xlens.pretrained import get_pretrained_model_config, get_pretrained_state_dict
from xlens.utils import load_pretrained_weights

pytest.importorskip("torch")

import torch  # noqa: E402
from transformers import AutoTokenizer, LlamaForCausalLM  # noqa: E402


@torch.no_grad()
def test_get_pretrained_state_dict():
    hf_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=torch.float32)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    hf_model.eval()

    hf_input = tokenizer("Hello, my dog is cute!", return_tensors="pt")["input_ids"]
    hf_output = hf_model(hf_input)
    hf_logits = hf_output.logits

    del hf_model
    torch.cuda.empty_cache()

    cfg = get_pretrained_model_config("meta-llama/Llama-3.2-1B")
    state_dict = get_pretrained_state_dict("meta-llama/Llama-3.2-1B", cfg)
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
    test_get_pretrained_state_dict()
