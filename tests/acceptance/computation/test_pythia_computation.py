import jax
import jax.numpy as jnp
import pytest

from xlens import HookedTransformer

pytest.importorskip("torch")

import torch  # noqa: E402
from transformers import AutoTokenizer, GPTNeoXForCausalLM  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")


@torch.no_grad()
def test_pythia_computation():
    hf_model = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-70m", torch_dtype=torch.float32, attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    hf_model.eval()

    hf_input: torch.Tensor = tokenizer("Hello, my dog is cute!", return_tensors="pt")["input_ids"]
    hf_output = hf_model(hf_input)
    hf_logits = hf_output.logits

    del hf_model
    torch.cuda.empty_cache()

    model = HookedTransformer.from_pretrained("EleutherAI/pythia-70m")

    input = jnp.array(hf_input)
    logits, _ = model(input)

    print("Logits Difference: ", jnp.linalg.norm(logits - jnp.array(hf_logits)))

    hf_probs = torch.nn.functional.softmax(hf_logits, dim=-1)
    probs = jax.nn.softmax(logits, axis=-1)

    print("Probs Difference: ", jnp.linalg.norm(probs - jnp.array(hf_probs)))

    assert jnp.allclose(probs, jnp.array(hf_probs), atol=1e-3)


if __name__ == "__main__":
    test_pythia_computation()
