import jax
import jax.numpy as jnp
from transformers import GPT2Tokenizer

from xlens import HookedTransformer
from xlens.components.attention import Attention
from xlens.utilities.functional import functional


def test_kv_cache_attention():
    model = HookedTransformer.from_pretrained("gpt2")
    input = jax.random.normal(jax.random.PRNGKey(0), (1, 10, 768))
    attention = model.blocks[0].attn

    @functional
    def no_cache_forward(attention: Attention, input: jax.Array) -> jax.Array:
        return attention(input, input, input, attention_mask=jnp.ones((1, 10)))

    no_cache_result = no_cache_forward(attention, input)
    assert attention.past_kv_cache.value is None

    @functional
    def cache_forward(attention: Attention, input: jax.Array) -> jax.Array:
        assert attention.past_kv_cache.value is None
        logits_head, attention = attention(input[:, :-1], input[:, :-1], input[:, :-1])
        assert attention.past_kv_cache.value is not None
        logits_tail, attention = attention(input[:, -1:], input[:, -1:], input[:, -1:])
        return jnp.concatenate([logits_head, logits_tail], axis=1), attention

    cache_result = cache_forward(attention, input)

    assert jnp.allclose(no_cache_result, cache_result, atol=1e-4)


def test_kv_cache():
    model = HookedTransformer.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input = jnp.array(tokenizer("Hello, my dog is cute", return_tensors="np")["input_ids"])

    @functional
    def no_cache_forward(model: HookedTransformer, input: jax.Array) -> jax.Array:
        return model(input)[0]

    no_cache_logits = no_cache_forward(model, input)

    @functional
    def cache_forward(model: HookedTransformer, input: jax.Array) -> jax.Array:
        logits_head, model = model(input[:, :-2])
        assert model.blocks[0].attn.past_kv_cache.value is not None
        logits_tail, model = model(input[:, -2:])
        return jnp.concatenate([logits_head, logits_tail], axis=1)

    cache_logits = cache_forward(model, input)
    print("No Cache Logits: ", no_cache_logits[0, -1, :5])
    print("Cache Logits: ", cache_logits[0, -1, :5])

    assert jnp.allclose(no_cache_logits, cache_logits, atol=1e-4)


if __name__ == "__main__":
    test_kv_cache()
