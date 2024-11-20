"""Hooked Transformer MLP Component.

This module contains all the component :class:`MLP`.
"""

from typing import Callable, Optional, Union

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jaxtyping import Float

from xlens.components.layer_norm import LayerNorm, LayerNormPre
from xlens.config import HookedTransformerConfig
from xlens.hooks.hook_point import HookPoint
from xlens.utilities.activation_functions import SUPPORTED_ACTIVATIONS


class MLP(nnx.Module):
    cfg: HookedTransformerConfig

    act_fn: Callable[[Float[jax.Array, "batch pos d_mlp"]], Float[jax.Array, "batch pos d_mlp"]]

    W_in: nnx.Param[Float[jax.Array, "d_model d_mlp"]]
    b_in: nnx.Param[Float[jax.Array, " d_mlp"]]

    W_out: nnx.Param[Float[jax.Array, "d_mlp d_model"]]
    b_out: nnx.Param[Float[jax.Array, " d_model"]]

    ln: Optional[Union[LayerNorm, LayerNormPre]]

    hook_pre: HookPoint
    hook_mid: Optional[HookPoint]
    hook_post: HookPoint

    def __init__(self, cfg: HookedTransformerConfig):
        self.cfg = cfg

        assert cfg.act_fn in SUPPORTED_ACTIVATIONS, f"Unsupported activation function: {cfg.act_fn}"
        act_fn = SUPPORTED_ACTIVATIONS.get(cfg.act_fn)
        if act_fn is None:
            raise ValueError(f"Unsupported activation function: {cfg.act_fn}")
        self.act_fn = act_fn

        self.W_in = nnx.Param(jnp.zeros((cfg.d_model, cfg.d_mlp)))
        self.b_in = nnx.Param(jnp.zeros(cfg.d_mlp))

        self.W_out = nnx.Param(jnp.zeros((cfg.d_mlp, cfg.d_model)))
        self.b_out = nnx.Param(jnp.zeros(cfg.d_model))

        self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        self.hook_post = HookPoint()  # [batch, pos, d_mlp]

        if self.cfg.is_layer_norm_activation():
            self.hook_mid = HookPoint()
            if self.cfg.normalization_type == "LN":
                self.ln = LayerNorm(self.cfg, self.cfg.d_mlp)
            else:
                self.ln = LayerNormPre(self.cfg)
        else:
            self.hook_mid = None
            self.ln = None

    def __call__(self, x: Float[jax.Array, "batch pos d_model"]) -> Float[jax.Array, "batch pos d_model"]:
        # There's no fused `addmm` here. May cause performance issues.
        pre_act = self.hook_pre(x @ self.W_in + self.b_in)

        if self.cfg.is_layer_norm_activation():
            assert (
                self.ln is not None and self.hook_mid is not None
            ), "LayerNorm and HookPoint must be set for layer norm activation"
            mid_act = self.hook_mid(self.act_fn(pre_act))  # [batch, pos, d_mlp]
            post_act = self.hook_post(self.ln(mid_act))
        else:
            post_act = self.hook_post(self.act_fn(pre_act))  # [batch, pos, d_mlp]
        return post_act @ self.W_out + self.b_out


class GatedMLP(nnx.Module):
    """
    The equation of a gated MLP:
    pre = x @ W_gate
    pre_linear = x @ W_in
    post = Gelu(pre) * (pre_linear) + b_in
    mlp_out = post @ W_out + b_out

    In one equation, mlp_out = (Gelu(x @ W_gate) * (x @ W_in) + b_in) @ W_out + b_out
    """

    cfg: HookedTransformerConfig

    act_fn: Callable[[Float[jax.Array, "batch pos d_mlp"]], Float[jax.Array, "batch pos d_mlp"]]

    W_in: nnx.Param[Float[jax.Array, "d_model d_mlp"]]
    b_in: nnx.Param[Float[jax.Array, " d_mlp"]]

    W_out: nnx.Param[Float[jax.Array, "d_mlp d_model"]]
    b_out: nnx.Param[Float[jax.Array, " d_model"]]

    W_gate: nnx.Param[Float[jax.Array, "d_model d_mlp"]]

    ln: Optional[Union[LayerNorm, LayerNormPre]]

    hook_pre: HookPoint
    hook_mid: Optional[HookPoint]
    hook_post: HookPoint
    hook_pre_linear: HookPoint

    def __init__(self, cfg: HookedTransformerConfig):
        self.cfg = cfg

        assert cfg.act_fn in SUPPORTED_ACTIVATIONS, f"Unsupported activation function: {cfg.act_fn}"
        act_fn = SUPPORTED_ACTIVATIONS.get(cfg.act_fn)
        if act_fn is None:
            raise ValueError(f"Unsupported activation function: {cfg.act_fn}")
        self.act_fn = act_fn

        self.W_in = nnx.Param(jnp.zeros((cfg.d_model, cfg.d_mlp)))
        self.b_in = nnx.Param(jnp.zeros(cfg.d_mlp))

        self.W_out = nnx.Param(jnp.zeros((cfg.d_mlp, cfg.d_model)))
        self.b_out = nnx.Param(jnp.zeros(cfg.d_model))

        self.W_gate = nnx.Param(jnp.zeros((cfg.d_model, cfg.d_mlp)))

        # hook on gate output but before act_fn
        self.hook_pre = HookPoint()  # [batch, pos, d_mlp]
        # hook on the linear component of the input
        self.hook_pre_linear = HookPoint()  # [batch, pos, d_mlp]
        # hook on act_fn(gate_output) * W_in(x) + b_in
        self.hook_post = HookPoint()  # [batch, pos, d_mlp]

        if self.cfg.is_layer_norm_activation():
            self.hook_mid = HookPoint()
            if self.cfg.normalization_type == "LN":
                self.ln = LayerNorm(self.cfg, self.cfg.d_mlp)
            else:
                self.ln = LayerNormPre(self.cfg)
        else:
            self.hook_mid = None
            self.ln = None

    def __call__(self, x: Float[jax.Array, "batch pos d_model"]) -> Float[jax.Array, "batch pos d_model"]:
        # Technically, all these einsums could be done with a single matmul, but this is more readable.
        pre_act = self.hook_pre(x @ self.W_gate)

        if self.cfg.is_layer_norm_activation() and self.hook_mid is not None and self.ln is not None:
            assert (
                self.ln is not None and self.hook_mid is not None
            ), "LayerNorm and HookPoint must be set for layer norm activation"
            mid_act = self.hook_mid(self.act_fn(pre_act))  # [batch, pos, d_mlp]
            post_act = self.hook_post(self.ln(mid_act))
        else:
            pre_linear = self.hook_pre_linear(x @ self.W_in)
            post_act = self.hook_post((self.act_fn(pre_act) * pre_linear) + self.b_in)  # [batch, pos, d_mlp]

        return post_act @ self.W_out + self.b_out
