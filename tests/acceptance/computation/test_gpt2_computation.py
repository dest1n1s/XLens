import jax
import jax.numpy as jnp
import pytest

from xlens import HookedTransformer

pytest.importorskip("torch")

import torch  # noqa: E402
from transformers import GPT2LMHeadModel, GPT2Tokenizer  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")


@torch.no_grad()
def test_gpt2_computation():
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    hf_model.eval()

    hf_input: torch.Tensor = tokenizer("Hello, my dog is cute", return_tensors="pt")["input_ids"]
    hf_logits = hf_model(hf_input).logits

    del hf_model
    torch.cuda.empty_cache()

    model = HookedTransformer.from_pretrained("gpt2")

    input = jnp.array(hf_input)
    logits, _ = model(input)

    print("Logits Difference: ", jnp.linalg.norm(logits - jnp.array(hf_logits)))

    hf_probs = torch.nn.functional.softmax(hf_logits, dim=-1)
    probs = jax.nn.softmax(logits, axis=-1)

    print("Probs Difference: ", jnp.linalg.norm(probs - jnp.array(hf_probs)))

    assert jnp.allclose(logits, jnp.array(hf_logits), atol=1e-4)


if __name__ == "__main__":
    test_gpt2_computation()
