from typing import Callable, Optional, Union

import equinox as eqx
import jax
from jaxtyping import Float, Int

from xlens.components import LayerNorm, LayerNormPre, RMSNorm, RMSNormPre
from xlens.components.attention import Attention
from xlens.components.mlp import MLP, GatedMLP
from xlens.config import HookedTransformerConfig
from xlens.hooks.hook_point import HookPoint

LayerNormLike = Union[LayerNorm, LayerNormPre, RMSNorm, RMSNormPre]


class TransformerBlock(eqx.Module):
    cfg: HookedTransformerConfig = eqx.field(static=True)

    ln1: Callable[[Float[jax.Array, "batch pos d_model"]], Float[jax.Array, "batch pos d_model"]]
    ln2: Optional[Callable[[Float[jax.Array, "batch pos d_model"]], Float[jax.Array, "batch pos d_model"]]]
    attn: Attention
    mlp: Optional[MLP | GatedMLP]

    hook_attn_in: HookPoint
    hook_q_input: HookPoint
    hook_k_input: HookPoint
    hook_v_input: HookPoint
    hook_mlp_in: HookPoint

    hook_attn_out: HookPoint
    hook_mlp_out: HookPoint

    hook_resid_pre: HookPoint
    hook_resid_mid: Optional[HookPoint]
    hook_resid_post: HookPoint

    def __init__(self, cfg: HookedTransformerConfig, block_index):
        self.cfg = cfg

        if cfg.normalization_type == "LN":
            normalization_layer: Callable[
                [HookedTransformerConfig],
                Callable[[Float[jax.Array, "batch pos d_model"]], Float[jax.Array, "batch pos d_model"]],
            ] = LayerNorm
        elif cfg.normalization_type == "LNPre":
            # We've folded in LayerNorm weights, so just need the center + scale parts
            normalization_layer = LayerNormPre
        elif cfg.normalization_type == "RMS":
            normalization_layer = RMSNorm
        elif cfg.normalization_type == "RMSPre":
            normalization_layer = RMSNormPre
        elif cfg.normalization_type is None:
            # This should just be the identity.
            # We need to make this a lambda so we can call it on the config, just like the others
            def normalization_layer(cfg):
                def identity(x: jax.Array):
                    return x

                return identity
        else:
            raise ValueError(f"Invalid normalization_type passed in: {cfg.normalization_type}")

        self.ln1 = normalization_layer(cfg)

        self.ln2 = normalization_layer(cfg) if not self.cfg.attn_only else None

        self.attn = Attention(self.cfg, "global", block_index)
        if not self.cfg.attn_only:
            self.mlp = GatedMLP(self.cfg) if self.cfg.gated_mlp else MLP(self.cfg)
        else:
            self.mlp = None

        self.hook_attn_in = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_q_input = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_k_input = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_v_input = HookPoint()  # [batch, pos, n_heads, d_model]
        self.hook_mlp_in = HookPoint()  # [batch, pos, d_model]

        self.hook_attn_out = HookPoint()  # [batch, pos, d_model]
        self.hook_mlp_out = HookPoint()  # [batch, pos, d_model]

        self.hook_resid_pre = HookPoint()  # [batch, pos, d_model]
        self.hook_resid_mid = HookPoint() if not self.cfg.attn_only else None  # [batch, pos, d_model]
        self.hook_resid_post = HookPoint()  # [batch, pos, d_model]

    def __call__(
        self,
        resid_pre: Float[jax.Array, "batch pos d_model"],
        attention_mask: Optional[Int[jax.Array, "batch offset_pos"]] = None,
    ) -> Float[jax.Array, "batch pos d_model"]:
        """A single Transformer block.

        Args:
            resid_pre (jax.Array): The residual stream - shape [batch, pos, d_model]
            past_kv_cache_entry (HookedTransformerKeyValueCache): A cache of previous keys and values, used only when generating text. Defaults to None.
            attention_mask (jax.Array, optional): The attention mask for padded tokens. Defaults to None.

        Returns:
            Float[jax.Array, "batch pos d_model"]: Our resulting tensor
        """
        resid_pre = self.hook_resid_pre(resid_pre)  # [batch, pos, d_model]

        attn_in = resid_pre

        query_input = attn_in
        key_input = attn_in
        value_input = attn_in

        attn_out = self.hook_attn_out(
            # hook the residual stream states that are used to calculate the
            # queries, keys and values, independently.
            # Then take the layer norm of these inputs, and pass these to the attention module.
            self.attn(
                query_input=self.ln1(query_input),
                key_input=self.ln1(key_input),
                value_input=self.ln1(value_input),
                attention_mask=attention_mask,
            )
        )  # [batch, pos, d_model]

        if not self.cfg.attn_only and not self.cfg.parallel_attn_mlp:
            assert (
                self.mlp is not None and self.ln2 is not None and self.hook_resid_mid is not None
            ), "MLP, LayerNorm2 and hook_resid_mid must be defined if attn_only is False"
            resid_mid = self.hook_resid_mid(resid_pre + attn_out)  # [batch, pos, d_model]
            mlp_in = self.hook_mlp_in(resid_mid)
            normalized_resid_mid = self.ln2(mlp_in)
            mlp_out = self.apply_mlp(normalized_resid_mid)
            resid_post = self.hook_resid_post(resid_mid + mlp_out)  # [batch, pos, d_model]
        elif self.cfg.parallel_attn_mlp:
            # Dumb thing done by GPT-J, both MLP and Attn read from resid_pre and write to resid_post, no resid_mid used.
            # In GPT-J, LN1 and LN2 are tied, in GPT-NeoX they aren't.
            normalized_resid_pre_2 = self.ln2(self.hook_mlp_in(resid_pre))
            mlp_out = self.apply_mlp(normalized_resid_pre_2)
            resid_post = self.hook_resid_post(resid_pre + attn_out + mlp_out)  # [batch, pos, d_model]
        else:
            resid_post = self.hook_resid_post(resid_pre + attn_out)  # [batch, pos, d_model]

        return resid_post

    def apply_mlp(
        self, normalized_resid: Float[jax.Array, "batch pos d_model"]
    ) -> Float[jax.Array, "batch pos d_model"]:
        """Centralized point where the MLP is applied to the forward pass

        Returns:
            Float[jax.Array, "batch pos d_model"]: Our resulting tensor
        """
        assert self.mlp is not None, "MLP must be defined if apply_mlp is called"
        mlp_out = self.mlp(normalized_resid)  # [batch, pos, d_model]
        return self.hook_mlp_out(mlp_out)
