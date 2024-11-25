import jax
from transformers import AutoTokenizer

from xlens.hooked_transformer import HookedTransformer


def example_generate():
    model = HookedTransformer.from_pretrained("Qwen/Qwen2-0.5B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
    input_ids = tokenizer("Hello, my dog is cute.", return_tensors="np")["input_ids"]

    @jax.jit
    def generate(input_ids: jax.Array, eos_token_id: int, top_k: int = 5, top_p: float = 0.95) -> jax.Array:
        return model.generate(input_ids, eos_token_id, top_k, top_p, rng=jax.random.PRNGKey(42))[0]

    generated = generate(input_ids, eos_token_id=tokenizer.eos_token_id)
    print(tokenizer.decode(generated[0]))


if __name__ == "__main__":
    example_generate()
