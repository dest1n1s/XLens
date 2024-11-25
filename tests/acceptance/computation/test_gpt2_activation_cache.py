import jax
import jax.numpy as jnp
import pytest

from xlens import HookedTransformer

pytest.importorskip("torch")

import torch  # noqa: E402
import transformer_lens as tl  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")


@torch.no_grad()
def test_gpt2_activation_cache():
    hook_points = [f"blocks.{i}.hook_resid_post" for i in range(12)]

    tl_model = tl.HookedTransformer.from_pretrained_no_processing("gpt2")
    tokenizer = tl_model.tokenizer
    tl_model.eval()

    tl_input: torch.Tensor = tokenizer("Hello, my dog is cute", return_tensors="pt")["input_ids"]
    tl_logits, tl_cache = tl_model.run_with_cache(tl_input, names_filter=hook_points)

    del tl_model
    torch.cuda.empty_cache()

    model = HookedTransformer.from_pretrained("gpt2")

    input = jnp.array(tl_input)
    logits, cache, _ = model.run_with_cache(input, hook_names=hook_points)

    print("Logits Difference: ", jnp.linalg.norm(logits - jnp.array(tl_logits)))

    assert jnp.allclose(logits, jnp.array(tl_logits), atol=1e-4)

    for i in range(12):
        assert jnp.allclose(
            cache[f"blocks.{i}.hook_resid_post"], jnp.array(tl_cache[f"blocks.{i}.hook_resid_post"]), atol=1e-4
        )


if __name__ == "__main__":
    test_gpt2_activation_cache()
