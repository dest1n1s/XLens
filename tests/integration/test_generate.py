import timeit

import jax
from transformers import AutoTokenizer

from xlens.hooked_transformer import HookedTransformer
from xlens.utilities.functional import functional


def test_generate():
    model = HookedTransformer.from_pretrained("Qwen/Qwen2-0.5B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    input_ids = tokenizer("Hello, my dog is cute.", return_tensors="np")["input_ids"]

    @functional(transform=jax.jit)
    def generate(
        model: HookedTransformer, input_ids: jax.Array, eos_token_id: int, top_k: int = 5, top_p: float = 0.95
    ) -> jax.Array:
        return model.generate(input_ids, eos_token_id, top_k, top_p, rng=jax.random.PRNGKey(42))[0]

    def generate_with_timeit():
        return generate(model, input_ids, eos_token_id=tokenizer.eos_token_id)

    print(
        "No JIT time:",
        timeit.timeit(lambda: model.generate(input_ids, eos_token_id=tokenizer.eos_token_id), number=5) / 5,
    )
    print("JIT time:", timeit.timeit(generate_with_timeit, number=1))
    print("JITted time:", timeit.timeit(generate_with_timeit, number=10) / 10)
    generated = generate_with_timeit()
    print(tokenizer.decode(generated[0])[:100])


if __name__ == "__main__":
    test_generate()
