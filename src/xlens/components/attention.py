from typing import Optional, Tuple, Union

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, Int

from xlens.config import HookedTransformerConfig
from xlens.hooks.hook_point import HookPoint


class Attention(eqx.Module):
    cfg: HookedTransformerConfig = eqx.field(static=True)

    attn_type: str = eqx.field(static=True)
    mask: Int[jax.Array, "pos pos"] = eqx.field(static=True)
    IGNORE: float = eqx.field(static=True)
    layer_id: Optional[int] = eqx.field(static=True)
    attn_scale: float = eqx.field(static=True)

    W_Q: Float[jax.Array, "n_heads d_model d_head"]
    W_O: Float[jax.Array, "n_heads d_head d_model"]
    W_K: Float[jax.Array, "n_heads d_model d_head"]
    W_V: Float[jax.Array, "n_heads d_model d_head"]

    b_Q: Float[jax.Array, "n_heads d_head"]
    b_K: Float[jax.Array, "n_heads d_head"]
    b_V: Float[jax.Array, "n_heads d_head"]
    b_O: Float[jax.Array, " d_model"]

    hook_k: HookPoint
    hook_q: HookPoint
    hook_v: HookPoint
    hook_z: HookPoint
    hook_attn_scores: HookPoint
    hook_pattern: HookPoint
    hook_result: HookPoint

    def __init__(
        self,
        cfg: HookedTransformerConfig,
        attn_type: str = "global",
        layer_id: Optional[int] = None,
    ):
        """Abstract Base Class of Attention Blocks, featuring common functionality of both Attention and GroupedQueryAttention blocks.

        Query and Output projections are defined in this class as they are the same for regular and grouped query attention.
        Attributes related to Key and Value projections are abstract as their implementations may differ. For example, in GroupedQueryAttention there are less query and key heads than value heads.
        To enforce implementation of W_K, W_V, b_K, and b_V by child classes, the better_abc.abstract_attribute class is used. See here for details: https://stackoverflow.com/questions/23831510/abstract-attribute-not-property.

        Args:
            cfg (Union[Dict, HookedTransformerConfig]): Config
            attn_type (str, optional): "global" or "local", used by GPT-Neo. Local attention means the model can only attend back cfg.window_size tokens (here, 256). Not used by any other model at the moment. Defaults to "global".
            layer_id (int, optional): The index of the current layer. Used by the Mistral models (labelled here as stanford-gpt2) to scale down attention scores pre softmax for numerical stability reasons by 1/(layer_id+1). Defaults to None.
        """
        self.cfg = cfg

        self.W_Q = jnp.zeros((self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head))
        self.W_O = jnp.zeros((self.cfg.n_heads, self.cfg.d_head, self.cfg.d_model))
        self.W_K = jnp.zeros((self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head))
        self.W_V = jnp.zeros((self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head))

        self.b_Q = jnp.zeros((self.cfg.n_heads, self.cfg.d_head))
        self.b_K = jnp.zeros((self.cfg.n_heads, self.cfg.d_head))
        self.b_V = jnp.zeros((self.cfg.n_heads, self.cfg.d_head))
        self.b_O = jnp.zeros((self.cfg.d_model,))

        self.attn_type = attn_type
        # Create a max_ctx x max_ctx mask, with True iff that query position
        # can attend to that key position (query is first axis, key is second axis)
        causal_mask = jnp.tril(jnp.ones((self.cfg.n_ctx, self.cfg.n_ctx)).astype(bool))
        if self.attn_type == "global":
            # For global attention, this is a lower triangular matrix - key <= query
            self.mask = causal_mask
        elif self.attn_type == "local":
            # For local, this is banded, query - window_size < key <= query
            if not isinstance(self.cfg.window_size, int):
                raise ValueError("Window size must be an integer for local attention")
            self.mask = jnp.triu(causal_mask, 1 - self.cfg.window_size)
        else:
            raise ValueError(f"Invalid attention type: {self.attn_type}")

        self.IGNORE = -jnp.inf
        self.layer_id = layer_id

        # attn_scale is a constant that we divide the attention scores by pre-softmax. I'm not entirely sure why it matters, but it's probably a mix of softmax not being scale invariant and numerical stability?
        if self.cfg.use_attn_scale:
            self.attn_scale = self.cfg.attn_scale  # Defaults to sqrt(d_head)
        else:
            self.attn_scale = 1.0
        if self.cfg.scale_attn_by_inverse_layer_idx:
            if self.layer_id is None:  # keep mypy happy
                raise ValueError("Layer ID must be provided to scale attention scores")
            self.attn_scale *= self.layer_id + 1

        self.hook_k = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_q = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_v = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_z = HookPoint()  # [batch, pos, head_index, d_head]
        self.hook_attn_scores = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_pattern = HookPoint()  # [batch, head_index, query_pos, key_pos]
        self.hook_result = HookPoint()  # [batch, pos, head_index, d_model]

    def __call__(
        self,
        query_input: Union[
            Float[jax.Array, "batch pos d_model"],
            Float[jax.Array, "batch pos head_index d_model"],
        ],
        key_input: Union[
            Float[jax.Array, "batch kv_pos d_model"],
            Float[jax.Array, "batch kv_pos head_index d_model"],
            Float[jax.Array, "batch kv_pos kv_head_index d_model"],
        ],
        value_input: Union[
            Float[jax.Array, "batch kv_pos d_model"],
            Float[jax.Array, "batch kv_pos head_index d_model"],
            Float[jax.Array, "batch kv_pos kv_head_index d_model"],
        ],
        additive_attention_mask: Optional[Float[jax.Array, "batch 1 1 kv_pos"]] = None,
        attention_mask: Optional[Int[jax.Array, "batch offset_pos"]] = None,
    ) -> Float[jax.Array, "batch pos d_model"]:
        """Forward pass for attention.

        additive_attention_mask is an optional mask to add to the attention weights. Defaults to None.
        attention_mask is the attention mask for padded tokens. Defaults to None.
        """

        q, k, v = self.calculate_qkv_matrices(query_input, key_input, value_input)

        kv_cache_pos_offset = 0

        attn_scores = self.calculate_attention_scores(q, k)  # [batch, head_index, query_pos, key_pos]

        if self.cfg.attention_dir == "causal":
            # If causal attention, we mask it to only attend backwards. If bidirectional, we don't mask.
            attn_scores = self.apply_causal_mask(
                attn_scores, kv_cache_pos_offset, attention_mask
            )  # [batch, head_index, query_pos, key_pos]
        if additive_attention_mask is not None:
            attn_scores += additive_attention_mask

        attn_scores = self.hook_attn_scores(attn_scores)
        pattern = jax.nn.softmax(attn_scores, axis=-1)
        pattern = jnp.where(jnp.isnan(pattern), jnp.zeros_like(pattern), pattern)
        pattern = self.hook_pattern(pattern)  # [batch, head_index, query_pos, key_pos]
        z = self.calculate_z_scores(v, pattern)  # [batch, pos, head_index, d_head]
        w = einops.rearrange(
            self.W_O,
            "head_index d_head d_model -> d_model head_index d_head",
        )
        result = self.hook_result(
            einops.einsum(
                z,
                w,
                "... head_index d_head, d_model head_index d_head -> ... head_index d_model",
            )
        )  # [batch, pos, head_index, d_model]
        out = (
            einops.reduce(result, "batch position index model->batch position model", "sum") + self.b_O
        )  # [batch, pos, d_model]
        return out

    def calculate_qkv_matrices(
        self,
        query_input: Float[jax.Array, "batch pos d_model"],
        key_input: Float[jax.Array, "batch pos d_model"],
        value_input: Float[jax.Array, "batch pos d_model"],
    ) -> Tuple[
        Float[jax.Array, "batch pos head_index d_head"],
        Float[jax.Array, "batch kv_pos head_index d_head"],
        Float[jax.Array, "batch kv_pos head_index d_head"],
    ]:
        def attn_fn(
            input: Float[jax.Array, "batch pos d_model"],
            w: Float[jax.Array, "head_index d_model d_head"],
            b: Float[jax.Array, "head_index d_head"],
        ) -> Float[jax.Array, "batch pos head_index d_head"]:
            """Linear layer for attention calculation."""
            return (
                einops.einsum(
                    input,
                    w,
                    "batch pos d_model, head_index d_model d_head -> batch pos head_index d_head",
                )
                + b
            )

        q = self.hook_q(attn_fn(query_input, self.W_Q, self.b_Q))
        k = self.hook_k(attn_fn(key_input, self.W_K, self.b_K))
        v = self.hook_v(attn_fn(value_input, self.W_V, self.b_V))

        return q, k, v

    def calculate_attention_scores(
        self,
        q: Float[jax.Array, "batch query_pos head_index d_head"],
        k: Float[jax.Array, "batch key_pos head_index d_head"],
    ) -> Float[jax.Array, "batch head_index query_pos key_pos"]:
        q_ = einops.rearrange(q, "batch query_pos head_index d_head -> batch head_index query_pos d_head")
        k_ = einops.rearrange(k, "batch key_pos head_index d_head -> batch head_index d_head key_pos")
        attn_scores = q_ @ k_ / self.attn_scale
        return attn_scores

    def calculate_z_scores(
        self,
        v: Float[jax.Array, "batch key_pos head_index d_head"],
        pattern: Float[jax.Array, "batch head_index query_pos key_pos"],
    ) -> Float[jax.Array, "batch query_pos head_index d_head"]:
        v_ = einops.rearrange(v, "batch key_pos head_index d_head -> batch head_index key_pos d_head")
        pattern_ = einops.rearrange(
            pattern,
            "batch head_index query_pos key_pos -> batch head_index query_pos key_pos",
        )
        z = self.hook_z(
            einops.rearrange(
                pattern_ @ v_,
                "batch head_index query_pos d_head -> batch query_pos head_index d_head",
            )
        )
        return z

    def apply_causal_mask(
        self,
        attn_scores: Float[jax.Array, "batch head_index pos pos_plus_past_kv_pos_offset"],
        past_kv_pos_offset: int = 0,
        attention_mask: Optional[Int[jax.Array, "batch offset_pos"]] = None,
    ):
        # The query context length is the number of positions we take queries from - if not using a past_kv_cache this is just the context length (for the current prompt), but if we're caching it can be different.
        query_ctx_length = attn_scores.shape[-2]
        # The key context length is the number of positions in the past - this includes all positions in the cache
        # If not caching, query_ctx_length == key_ctx_length
        key_ctx_length = attn_scores.shape[-1]

        if query_ctx_length + past_kv_pos_offset != key_ctx_length:
            raise ValueError(
                f"query_ctx_length {query_ctx_length} + past_kv_pos_offset {past_kv_pos_offset} != key_ctx_length {key_ctx_length} - you likely have a bug."
            )

        # Index back to front to ensure local attention works
        final_mask = self.mask[None, None, -query_ctx_length:, -key_ctx_length:]  # [1, 1, pos, pos]
        if attention_mask is not None:
            # Apply a causal mask to the attention scores considering the padding
            final_mask = einops.einsum(
                final_mask, attention_mask, "batch head pos offset_pos, batch offset_pos -> batch head pos offset_pos"
            ).astype(bool)

        return jnp.where(final_mask, attn_scores, self.IGNORE)
