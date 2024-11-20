"""Hooked Transformer Unembed Component.

This module contains all the component :class:`Unembed`.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jaxtyping import Float

from xlens.config import HookedTransformerConfig


class Unembed(nnx.Module):
    cfg: HookedTransformerConfig

    W_U: nnx.Param[Float[jax.Array, "d_model d_vocab_out"]]

    def __init__(self, cfg: HookedTransformerConfig):
        self.cfg = cfg
        # Note that there's a separate variable for d_vocab_out and d_vocab (the input vocab size). For language tasks these are always the same, but for algorithmic tasks we may want them to be different.
        self.W_U = nnx.Param(jnp.zeros((self.cfg.d_model, self.cfg.d_vocab_out)))

    def __call__(self, residual: Float[jax.Array, "batch pos d_model"]) -> Float[jax.Array, "batch pos d_vocab_out"]:
        return residual @ self.W_U
