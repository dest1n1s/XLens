from typing import Optional, Tuple, Union

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, Int

from xlens.components.embed import get_offset_position_ids
from xlens.config import HookedTransformerConfig
from xlens.hooks.hook_point import HookPoint


class Attention(eqx.Module):
    cfg: HookedTransformerConfig = eqx.field(static=True)

    attn_type: str = eqx.field(static=True)
    mask: Int[jax.Array, "pos pos"]
    layer_id: Optional[int] = eqx.field(static=True)
    attn_scale: float = eqx.field(static=True)
    repeat_kv_heads: Optional[int] = eqx.field(static=True)
    rotary_sin: Optional[Float[jax.Array, "n_ctx rotary_dim"]]
    rotary_cos: Optional[Float[jax.Array, "n_ctx rotary_dim"]]

    W_Q: Float[jax.Array, "n_heads d_model d_head"]
    W_K: Float[jax.Array, "n_heads d_model d_head"] | Float[jax.Array, "n_key_value_heads d_model d_head"]
    W_V: Float[jax.Array, "n_heads d_model d_head"] | Float[jax.Array, "n_key_value_heads d_model d_head"]
    W_O: Float[jax.Array, "n_heads d_head d_model"]

    b_Q: Float[jax.Array, "n_heads d_head"]
    b_K: Float[jax.Array, "n_heads d_head"] | Float[jax.Array, "n_key_value_heads d_head"]
    b_V: Float[jax.Array, "n_heads d_head"] | Float[jax.Array, "n_key_value_heads d_head"]
    b_O: Float[jax.Array, " d_model"]

    hook_k: HookPoint
    hook_q: HookPoint
    hook_v: HookPoint
    hook_z: HookPoint
    hook_attn_scores: HookPoint
    hook_pattern: HookPoint
    hook_result: HookPoint
    hook_rot_k: Optional[HookPoint]
    hook_rot_q: Optional[HookPoint]

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
        if self.cfg.n_key_value_heads is None:
            self.W_K = jnp.zeros((self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head))
            self.W_V = jnp.zeros((self.cfg.n_heads, self.cfg.d_model, self.cfg.d_head))
        else:
            self.W_K = jnp.zeros((self.cfg.n_key_value_heads, self.cfg.d_model, self.cfg.d_head))
            self.W_V = jnp.zeros((self.cfg.n_key_value_heads, self.cfg.d_model, self.cfg.d_head))

        self.b_Q = jnp.zeros((self.cfg.n_heads, self.cfg.d_head))
        self.b_O = jnp.zeros((self.cfg.d_model,))
        if self.cfg.n_key_value_heads is None:
            self.b_K = jnp.zeros((self.cfg.n_heads, self.cfg.d_head))
            self.b_V = jnp.zeros((self.cfg.n_heads, self.cfg.d_head))
        else:
            self.b_K = jnp.zeros((self.cfg.n_key_value_heads, self.cfg.d_head))
            self.b_V = jnp.zeros((self.cfg.n_key_value_heads, self.cfg.d_head))

        self.repeat_kv_heads = (
            self.cfg.n_heads // self.cfg.n_key_value_heads if self.cfg.n_key_value_heads is not None else None
        )

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
            mask = jnp.triu(causal_mask, 1 - self.cfg.window_size)
            self.mask = jax.lax.stop_gradient(mask)
        else:
            raise ValueError(f"Invalid attention type: {self.attn_type}")

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

        if self.cfg.positional_embedding_type == "rotary":
            # Applies a rotation to each two-element chunk of keys and queries pre dot producting to bake in relative position. See HookedTransformerConfig for details
            self.hook_rot_k = HookPoint()
            self.hook_rot_q = HookPoint()
            if self.cfg.rotary_dim is None:  # keep mypy happy
                raise ValueError("Rotary dim must be provided for rotary positional embeddings")
            rotary_sin, rotary_cos = self.calculate_sin_cos_rotary(
                rotary_dim=self.cfg.rotary_dim,
                n_ctx=self.cfg.n_ctx,
                base=self.cfg.rotary_base,
                use_NTK_by_parts_rope=self.cfg.use_NTK_by_parts_rope,
                NTK_by_parts_factor=self.cfg.NTK_by_parts_factor,
                NTK_by_parts_low_freq_factor=self.cfg.NTK_by_parts_low_freq_factor,
                NTK_by_parts_high_freq_factor=self.cfg.NTK_by_parts_high_freq_factor,
                rotary_adjacent_pairs=self.cfg.rotary_adjacent_pairs,
            )
            self.rotary_sin = jax.lax.stop_gradient(rotary_sin)
            self.rotary_cos = jax.lax.stop_gradient(rotary_cos)
        else:
            self.hook_rot_k = None
            self.hook_rot_q = None
            self.rotary_sin = None
            self.rotary_cos = None

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

        if self.cfg.positional_embedding_type == "rotary":
            assert self.hook_rot_k is not None and self.hook_rot_q is not None, "Rotary hooks must be defined"
            assert self.cfg.rotary_dim is not None, "Rotary dim must be defined"
            q = self.hook_rot_q(
                self.apply_rotary(q, kv_cache_pos_offset, attention_mask, rotary_dim=self.cfg.rotary_dim)
            )
            k = self.hook_rot_k(
                self.apply_rotary(k, 0, attention_mask, rotary_dim=self.cfg.rotary_dim)
            )  # keys are cached so no offset

        # Promote precision to float32 if using 16-bit precision
        if q.dtype not in [jnp.float32, jnp.float64]:
            q = q.astype(jnp.float32)
        if k.dtype not in [jnp.float32, jnp.float64]:
            k = k.astype(jnp.float32)

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
        k: Float[jax.Array, "batch key_pos kv_head_index d_head"],
    ) -> Float[jax.Array, "batch head_index query_pos key_pos"]:
        if self.repeat_kv_heads is not None:
            k = einops.repeat(
                k,
                "batch key_pos kv_head_index d_head -> batch key_pos (kv_head_index repeat_kv_heads) d_head",
                repeat_kv_heads=self.repeat_kv_heads,
            )
        q_ = einops.rearrange(q, "batch query_pos head_index d_head -> batch head_index query_pos d_head")
        k_ = einops.rearrange(k, "batch key_pos head_index d_head -> batch head_index d_head key_pos")
        attn_scores = q_ @ k_ / self.attn_scale
        return attn_scores

    def calculate_z_scores(
        self,
        v: Float[jax.Array, "batch key_pos head_index d_head"],
        pattern: Float[jax.Array, "batch head_index query_pos key_pos"],
    ) -> Float[jax.Array, "batch query_pos head_index d_head"]:
        if self.repeat_kv_heads is not None:
            v = einops.repeat(
                v,
                "batch key_pos kv_head_index d_head -> batch key_pos (kv_head_index repeat_kv_heads) d_head",
                repeat_kv_heads=self.repeat_kv_heads,
            )
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

        return jnp.where(final_mask, attn_scores, -jnp.inf)

    @staticmethod
    def calculate_sin_cos_rotary(
        rotary_dim: int,
        n_ctx: int,
        base: int = 10000,
        dtype=jnp.float32,
        use_NTK_by_parts_rope: bool = False,
        NTK_by_parts_factor: float = 8.0,
        NTK_by_parts_low_freq_factor: float = 1.0,
        NTK_by_parts_high_freq_factor: float = 4.0,
        rotary_adjacent_pairs: bool = False,
    ) -> Tuple[Float[jax.Array, "n_ctx rotary_dim"], Float[jax.Array, "n_ctx rotary_dim"]]:
        """
        Calculate the sine and cosine waves to use in a rotary embedding.
        """
        pos = jnp.arange(n_ctx, dtype=dtype)
        dim = jnp.arange(rotary_dim // 2, dtype=dtype)

        if use_NTK_by_parts_rope:
            inv_freq = 1.0 / (base ** (jnp.arange(0, rotary_dim, 2, dtype=jnp.int32).astype(dtype) / rotary_dim))
            low_freq_wavelen = n_ctx / NTK_by_parts_low_freq_factor
            high_freq_wavelen = n_ctx / NTK_by_parts_high_freq_factor

            wavelen = 2 * jnp.pi / inv_freq
            inv_freq_llama = jnp.where(wavelen > low_freq_wavelen, inv_freq / NTK_by_parts_factor, inv_freq)
            smooth_factor = (n_ctx / wavelen - NTK_by_parts_low_freq_factor) / (
                NTK_by_parts_high_freq_factor - NTK_by_parts_low_freq_factor
            )
            smoothed_inv_freq = (
                1 - smooth_factor
            ) * inv_freq_llama / NTK_by_parts_factor + smooth_factor * inv_freq_llama
            is_medium_freq = ~(wavelen < high_freq_wavelen) & ~(wavelen > low_freq_wavelen)
            inv_freq_llama = jnp.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
            freq = 1 / inv_freq_llama
        else:
            freq = base ** (dim / (rotary_dim / 2))

        # Use einops to repeat frequencies
        if rotary_adjacent_pairs:
            freq = einops.repeat(freq, "d -> (d 2)")
        else:
            freq = einops.repeat(freq, "d -> (2 d)")

        # Create a n_ctx x rotary_dim tensor, where each column is an arithmetic sequence of angles in that frequency
        angles = pos[:, None] / freq[None, :]
        return jnp.sin(angles).astype(dtype), jnp.cos(angles).astype(dtype)

    def apply_rotary(
        self,
        x: Float[jax.Array, "batch pos head_index d_head"],
        past_kv_pos_offset=0,
        attention_mask: Optional[jnp.ndarray] = None,
        rotary_dim: int = 64,
    ) -> jnp.ndarray:
        """
        Apply rotary embeddings to the input tensor.
        """
        assert (
            self.rotary_sin is not None and self.rotary_cos is not None
        ), "Rotary sin and cos must be defined to apply rotary embeddings"

        # Only apply rotary to first rotary_dim dimensions (e.g., if rotary_dim=64 and d_head=256, only apply to first 1/4 of dimensions)
        x_pos = x.shape[1]
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        x_flip = self.rotate_every_two(x_rot)  # You need to define this function

        if attention_mask is None:
            rotary_cos_slice = self.rotary_cos[None, past_kv_pos_offset : past_kv_pos_offset + x_pos, None, :]
            rotary_sin_slice = self.rotary_sin[None, past_kv_pos_offset : past_kv_pos_offset + x_pos, None, :]
            x_rotated = x_rot * rotary_cos_slice + x_flip * rotary_sin_slice
        else:
            offset_position_ids = get_offset_position_ids(past_kv_pos_offset, attention_mask)
            mask_rotary_cos = self.rotary_cos[offset_position_ids, None, :]
            mask_rotary_sin = self.rotary_sin[offset_position_ids, None, :]
            x_rotated = x_rot * mask_rotary_cos + x_flip * mask_rotary_sin

        return jnp.concatenate([x_rotated, x_pass], axis=-1)

    def rotate_every_two(self, x: Float[jax.Array, "... rotary_dim"]) -> Float[jax.Array, "... rotary_dim"]:
        """
        Rotary helper function, splits x into blocks of size 2 along the final axis and maps [x0, x1] to [-x1, x0]

        The final axis of x must have even length.

        GPT-NeoX and GPT-J do rotary subtly differently, see calculate_sin_cos_rotary for details.
        """
        rot_x = x
        if self.cfg.rotary_adjacent_pairs:
            rot_x = rot_x.at[..., ::2].set(-x[..., 1::2])
            rot_x = rot_x.at[..., 1::2].set(x[..., ::2])
        else:
            n = x.shape[-1] // 2
            rot_x = rot_x.at[..., :n].set(-x[..., n:])
            rot_x = rot_x.at[..., n:].set(x[..., :n])

        return rot_x
