"""Hooked Transformer Layer Norm Component.

This module contains all the component :class:`LayerNorm`.
"""

from typing import Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float

from xlens.config import HookedTransformerConfig
from xlens.hooks import HookPoint


class LayerNorm(eqx.Module):
    cfg: HookedTransformerConfig = eqx.field(static=True)
    length: int = eqx.field(static=True)

    w: Float[jax.Array, " length"]
    b: Float[jax.Array, " length"]

    hook_scale: HookPoint
    hook_normalized: HookPoint

    def __init__(self, cfg: HookedTransformerConfig, length: Optional[int] = None):
        """
        LayerNorm with optional length parameter

        length (Optional[int]): If the dimension of the LayerNorm. If not provided, assumed to be d_model
        """
        self.cfg = cfg
        if length is None:
            self.length = self.cfg.d_model
        else:
            self.length = length

        self.w = jnp.ones(self.length)
        self.b = jnp.zeros(self.length)

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos, 1]
        # Hook_normalized is on the LN output
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def __call__(
        self,
        x: Union[
            Float[jax.Array, "batch pos d_model"],
            Float[jax.Array, "batch pos head_index d_model"],
        ],
    ) -> Union[
        Float[jax.Array, "batch pos d_model"],
        Float[jax.Array, "batch pos head_index d_model"],
    ]:
        x = x - x.mean(-1, keepdims=True)  # [batch, pos, length]

        scale: Float[jax.Array, "batch pos 1"] = self.hook_scale(
            jnp.sqrt((x**2).mean(-1, keepdims=True) + self.cfg.eps)
        )
        x = x / scale  # [batch, pos, length]
        return self.hook_normalized(x * self.w + self.b)


class LayerNormPre(eqx.Module):
    cfg: HookedTransformerConfig = eqx.field(static=True)

    hook_scale: HookPoint
    hook_normalized: HookPoint

    def __init__(self, cfg: HookedTransformerConfig):
        """LayerNormPre - the 'center and normalise' part of LayerNorm. Length is
        normally d_model, but is d_mlp for softmax. Not needed as a parameter. This
        should only be used in inference mode after folding in LayerNorm weights"""
        super().__init__()
        self.cfg = cfg

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos]
        # Hook Normalized captures LN output - here it's a vector with std 1 and mean 0
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def __call__(
        self,
        x: Union[
            Float[jax.Array, "batch pos d_model"],
            Float[jax.Array, "batch pos head_index d_model"],
        ],
    ) -> Union[
        Float[jax.Array, "batch pos d_model"],
        Float[jax.Array, "batch pos head_index d_model"],
    ]:
        x = x - x.mean(-1, keepdims=True)  # [batch, pos, length]
        scale: Float[jax.Array, "batch pos 1"] = self.hook_scale(
            jnp.sqrt((x**2).mean(-1, keepdims=True) + self.cfg.eps)
        )
        return self.hook_normalized(x / scale)


class RMSNorm(eqx.Module):
    cfg: HookedTransformerConfig = eqx.field(static=True)
    length: int = eqx.field(static=True)

    w: Float[jax.Array, " length"]

    hook_scale: HookPoint
    hook_normalized: HookPoint

    def __init__(self, cfg: HookedTransformerConfig, length: Optional[int] = None):
        """
        RMSNorm - LayerNorm without the centering and bias (RMS = Root Mean Square)

        length (Optional[int]): If the dimension of the RMSNorm. If not provided, assumed to be d_model
        """

        self.cfg = cfg
        if length is None:
            self.length = self.cfg.d_model
        else:
            self.length = length

        self.w = jnp.ones(self.length)

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos, 1]
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def __call__(self, x: Float[jax.Array, "batch pos length"]) -> Float[jax.Array, "batch pos length"]:
        scale: Float[jax.Array, "batch pos 1"] = self.hook_scale(
            jnp.sqrt((x**2).mean(-1, keepdims=True) + self.cfg.eps)
        )
        x = self.hook_normalized(x / scale)  # [batch, pos, length]
        return x * self.w


class RMSNormPre(eqx.Module):
    cfg: HookedTransformerConfig = eqx.field(static=True)

    hook_scale: HookPoint
    hook_normalized: HookPoint

    def __init__(self, cfg: HookedTransformerConfig):
        """RMSNormPre - LayerNormPre without the centering and bias (RMS = Root Mean Square)"""

        self.cfg = cfg

        # Adds a hook point for the normalisation scale factor
        self.hook_scale = HookPoint()  # [batch, pos]
        self.hook_normalized = HookPoint()  # [batch, pos, length]

    def __call__(self, x: Float[jax.Array, "batch pos length"]) -> Float[jax.Array, "batch pos length"]:
        scale: Float[jax.Array, "batch pos 1"] = self.hook_scale(
            jnp.sqrt((x**2).mean(-1, keepdims=True) + self.cfg.eps)
        )
        return self.hook_normalized(x / scale)  # [batch, pos, length]
