"""Hooked Transformer Embed Component.

This module contains all the component :class:`Embed`.
"""

from typing import Optional

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float, Int

from xlens.config import HookedTransformerConfig

from .layer_norm import LayerNorm


# Embed & Unembed
class Embed(eqx.Module):
    cfg: HookedTransformerConfig = eqx.field(static=True)
    W_E: Float[jax.Array, "d_vocab d_model"]
    ln: Optional[LayerNorm]

    def __init__(self, cfg: HookedTransformerConfig):
        self.cfg = cfg
        self.W_E = jnp.zeros((self.cfg.d_vocab, self.cfg.d_model))

        # Some models (e.g. Bloom) need post embedding layer norm
        self.ln = LayerNorm(self.cfg) if self.cfg.post_embedding_ln else None

    def __call__(self, tokens: Int[jax.Array, "batch pos"]) -> Float[jax.Array, "batch pos d_model"]:
        # If A has shape [a, b] and B has shape [c, d], then A[:, B] has shape [a, c, d]
        # B acts as a tensor of indices into the second dimension (so >=0 and <b)
        if self.cfg.post_embedding_ln:
            assert self.ln is not None
            return self.ln(self.W_E[tokens, :])
        return self.W_E[tokens, :]


def get_offset_position_ids(
    past_kv_pos_offset: int,
    attention_mask: Int[jax.Array, "batch offset_pos"],
) -> Int[jax.Array, "batch pos"]:
    """
    Returns the indices of non-padded tokens, offset by the position of the first attended token.
    """
    # shift the position ids so that the id at the the first attended token position becomes zero.
    # The position ids of the prepending pad tokens are shifted to -1.
    shifted_position_ids = attention_mask.cumsum(axis=1) - 1  # [batch, tokens_length]

    # Set the position ids of all prepending pad tokens to an arbitrary number (zero here)
    # just to avoid indexing errors.
    position_ids = jnp.where(attention_mask, shifted_position_ids, 0)  # [batch, tokens_length]
    return position_ids[:, past_kv_pos_offset:]


class PosEmbed(eqx.Module):
    cfg: HookedTransformerConfig = eqx.field(static=True)

    W_pos: Float[jax.Array, "n_ctx d_model"]

    def __init__(self, cfg: HookedTransformerConfig):
        self.cfg = cfg
        self.W_pos = jnp.zeros((self.cfg.n_ctx, self.cfg.d_model))

    def __call__(
        self,
        tokens: Int[jax.Array, "batch pos"],
        past_kv_pos_offset: int = 0,
        attention_mask: Optional[Int[jax.Array, "batch offset_pos"]] = None,
    ) -> Float[jax.Array, "batch pos d_model"]:
        """
        Forward pass for positional embeddings.

        Args:
            tokens (Int[jax.Array, "batch pos"]): Input tokens.
            past_kv_pos_offset (int, optional): The length of tokens in the past_kv_cache. Defaults to 0.
            attention_mask (Int[jax.Array, "batch pos"], optional): The attention mask for padded tokens.
                 Defaults to None.

        Returns:
            Float[jax.Array, "batch pos d_model"]: Absolute position embeddings.
        """

        tokens_length = tokens.shape[-1]

        if attention_mask is None:
            pos_embed = self.W_pos[past_kv_pos_offset : tokens_length + past_kv_pos_offset, :]  # [pos, d_model]
            batch_pos_embed = einops.repeat(pos_embed, "pos d_model -> batch pos d_model", batch=tokens.shape[0])

        else:
            # Separated from the no padding case for computational efficiency
            # (this code is a bit slower than the code above)

            offset_position_ids = get_offset_position_ids(past_kv_pos_offset, attention_mask)
            pos_embed = self.W_pos[offset_position_ids]  # [batch, pos, d_model]

            # Set the position embeddings to 0 for pad tokens (this is an arbitrary choice)
            padding_mask = ~attention_mask.astype(bool)  # [batch, pos]
            offset_padding_mask = jnp.expand_dims(
                padding_mask[:, past_kv_pos_offset : tokens_length + past_kv_pos_offset], -1
            )  # [batch, pos, 1]
            batch_pos_embed = jnp.where(offset_padding_mask, 0, pos_embed)

        return batch_pos_embed
