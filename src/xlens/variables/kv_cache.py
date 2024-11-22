from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Float, Int


class KVCache(
    nnx.Variable[
        Optional[
            tuple[
                Float[jax.Array, "batch kv_pos d_model"],
                Float[jax.Array, "batch kv_pos d_model"],
                Int[jax.Array, "batch kv_pos"],
            ]
        ]
    ]
):
    def __init__(
        self,
        max_length: int,
        value: Optional[
            tuple[
                Float[jax.Array, "batch kv_pos d_model"],
                Float[jax.Array, "batch kv_pos d_model"],
                Int[jax.Array, "batch kv_pos"],
            ]
        ] = None,
    ):
        super().__init__(value)
        self.max_length = max_length
        self.length = 0 if value is None else value[0].shape[1]

    def append(
        self,
        k: Float[jax.Array, "batch kv_pos ..."],
        v: Float[jax.Array, "batch kv_pos ..."],
        attention_mask: Optional[Int[jax.Array, "batch kv_pos"]] = None,
    ):
        assert k.shape[0] == v.shape[0], "Key and value must have the same batch size"
        assert k.shape[1] == v.shape[1], "Key and value must have the same number of positions"
        if self.value is None:
            self.value = (
                jnp.zeros((k.shape[0], self.max_length, *k.shape[2:])),
                jnp.zeros((k.shape[0], self.max_length, *v.shape[2:])),
                jnp.zeros((k.shape[0], self.max_length)),
            )
        if attention_mask is None:
            attention_mask = jnp.ones((k.shape[0], k.shape[1]))
        assert (
            attention_mask.shape[1] == k.shape[1]
        ), "Attention mask must have the same number of positions as key/value"
        assert (
            self.length + k.shape[1] <= self.max_length
        ), f"KV cache max length exceeded. Attempted to append {k.shape[1]} tokens with {self.length} already in the cache, but max length is {self.max_length}."

        cache_k = jax.lax.dynamic_update_slice(self.value[0], k, (0, self.length, *[0] * (len(k.shape) - 2)))
        cache_v = jax.lax.dynamic_update_slice(self.value[1], v, (0, self.length, *[0] * (len(v.shape) - 2)))
        cache_attention_mask = jax.lax.dynamic_update_slice(self.value[2], attention_mask, (0, self.length))
        self.value = (cache_k, cache_v, cache_attention_mask)
        self.length += k.shape[1]
        return self.value
