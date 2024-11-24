import timeit

import jax
import jax.numpy as jnp
from transformers import GPT2Tokenizer

from xlens.hooked_transformer import HookedTransformer
from xlens.utilities.functional import functional


def test_attention_mask():
    model = HookedTransformer.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input = jnp.array(tokenizer("Hello, my dog is cute", return_tensors="np")["input_ids"])

    @functional(transform=jax.jit)
    def no_mask_forward(model: HookedTransformer, input: jax.Array) -> jax.Array:
        return model(input)[0]

    print("No mask JIT time:", timeit.timeit(lambda: no_mask_forward(model, input), number=1) / 1)
    print("No mask JITted time:", timeit.timeit(lambda: no_mask_forward(model, input), number=10) / 10)

    logits = no_mask_forward(model, input)
    assert logits.shape[1] == input.shape[1]

    @functional(transform=jax.jit)
    def mask_forward(model: HookedTransformer, input: jax.Array, attention_mask: jax.Array) -> jax.Array:
        return model(input, attention_mask=attention_mask)[0]

    print(
        "Mask JIT time:",
        timeit.timeit(
            lambda: mask_forward(
                model, input, attention_mask=jnp.ones((input.shape[0], input.shape[1]), dtype=jnp.int32)
            ),
            number=1,
        )
        / 1,
    )
    print(
        "Mask JITted time:",
        timeit.timeit(
            lambda: mask_forward(
                model, input, attention_mask=jnp.ones((input.shape[0], input.shape[1]), dtype=jnp.int32)
            ),
            number=10,
        )
        / 10,
    )

    logits_with_mask = mask_forward(
        model, input, attention_mask=jnp.ones((input.shape[0], input.shape[1]), dtype=jnp.int32)
    )
    assert logits_with_mask.shape[1] == input.shape[1]

    assert jnp.allclose(logits, logits_with_mask)


if __name__ == "__main__":
    test_attention_mask()
