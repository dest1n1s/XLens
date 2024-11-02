import logging
from typing import Optional, Union

import equinox as eqx
import jax
from jaxtyping import Float, Int

from xlens.components import Embed, LayerNorm, LayerNormPre, PosEmbed, RMSNorm, RMSNormPre, TransformerBlock, Unembed

from .config import HookedTransformerConfig
from .hooks import HookPoint

LayerNormLike = Union[LayerNorm, LayerNormPre, RMSNorm, RMSNormPre]


class HookedTransformer(eqx.Module):
    cfg: HookedTransformerConfig = eqx.field(static=True)

    embed: Embed
    pos_embed: PosEmbed
    blocks: list[TransformerBlock]
    ln_final: Optional[LayerNormLike] = None
    unembed: Unembed

    hook_embed: HookPoint
    hook_tokens: HookPoint
    hook_pos_embed: HookPoint

    def __init__(self, cfg: HookedTransformerConfig):
        self.cfg = cfg

        self.embed = Embed(cfg=cfg)
        self.pos_embed = PosEmbed(cfg=cfg)

        self.blocks = [TransformerBlock(cfg=cfg, block_index=i) for i in range(cfg.n_layers)]

        if self.cfg.normalization_type == "RMS":
            self.ln_final = RMSNorm(self.cfg)
        elif self.cfg.normalization_type == "RMSPre":
            self.ln_final = RMSNormPre(self.cfg)
        elif self.cfg.normalization_type == "LN":
            self.ln_final = LayerNorm(self.cfg)
        elif self.cfg.normalization_type == "LNPre":
            # We've folded in LayerNorm weights, so just need the center + scale parts
            self.ln_final = LayerNormPre(self.cfg)
        elif self.cfg.normalization_type is None:
            # If it's None, don't create either layer
            pass
        else:
            logging.warning("Invalid normalization_type passed in %s", self.cfg.normalization_type)
        self.unembed = Unembed(self.cfg)

        self.hook_embed = HookPoint()
        self.hook_tokens = HookPoint()
        self.hook_pos_embed = HookPoint()

    def __call__(
        self,
        input: Int[jax.Array, "batch pos"],
        attention_mask: Optional[jax.Array] = None,  # [batch pos]
    ) -> Float[jax.Array, "batch pos d_vocab"]:
        """Forward Pass.

        Input is either a batch of tokens ([batch, pos]) or a text string, a string is automatically
        tokenized to a batch of a single element. The prepend_bos flag only applies when inputting a
        text string.

        Note that loss is the standard "predict the next token" cross-entropy loss for GPT-2 style
        language models - if you want a custom loss function, the recommended behaviour is returning
        the logits and then applying your custom loss function.

        Args:
            attention_mask: Optional[jax.Array]: Override the attention mask used to ignore
                padded tokens. If start_at_layer is not None and (self.tokenizer.padding_side ==
                "left" or past_kv_cache is not None), this should be passed as the attention mask
                is not computed automatically. Defaults to None.
        """

        tokens = self.hook_tokens(input)  # [batch, pos]
        embed = self.hook_embed(self.embed(tokens))  # [batch, pos, d_model]
        pos_embed = self.hook_pos_embed(self.pos_embed(tokens, 0, attention_mask))  # [batch, pos, d_model]
        residual = embed + pos_embed

        for i, block in list(zip(range(self.cfg.n_layers), self.blocks)):
            # Note that each block includes skip connections, so we don't need
            # residual + block(residual)
            residual = block(
                residual,
                attention_mask=attention_mask,
            )  # [batch, pos, d_model]

        if self.cfg.normalization_type is not None:
            assert self.ln_final is not None, "ln_final should be set if normalization_type is set"
            residual = self.ln_final(residual)  # [batch, pos, d_model]
        logits = self.unembed(residual)  # [batch, pos, d_vocab]
        return logits
