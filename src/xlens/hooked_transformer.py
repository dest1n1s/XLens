import logging
from typing import Any, Callable, Optional, Union, cast

import flax.nnx as nnx
import jax
from jaxtyping import Float, Int

from xlens.components import (
    Embed,
    LayerNorm,
    LayerNormPre,
    PosEmbed,
    RMSNorm,
    RMSNormPre,
    TransformerBlock,
    Unembed,
)
from xlens.hooks import with_cache, with_hooks
from xlens.pretrained.convert import get_pretrained_model_config, get_pretrained_weights
from xlens.utils import load_pretrained_weights

from .config import HookedTransformerConfig
from .hooks import HookPoint

LayerNormLike = Union[LayerNorm, LayerNormPre, RMSNorm, RMSNormPre]


class HookedTransformer(nnx.Module):
    cfg: HookedTransformerConfig

    embed: Embed
    pos_embed: PosEmbed
    blocks: list[TransformerBlock]
    ln_final: Optional[LayerNormLike]
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
            if self.cfg.final_rms:
                self.ln_final = RMSNorm(self.cfg)
            else:
                self.ln_final = LayerNorm(self.cfg)
        elif self.cfg.normalization_type == "LNPre":
            # We've folded in LayerNorm weights, so just need the center + scale parts
            if self.cfg.final_rms:
                self.ln_final = RMSNormPre(self.cfg)
            else:
                self.ln_final = LayerNormPre(self.cfg)
        elif self.cfg.normalization_type is None:
            self.ln_final = None
        else:
            self.ln_final = None
            logging.warning("Invalid normalization_type passed in %s", self.cfg.normalization_type)
        self.unembed = Unembed(self.cfg)

        self.hook_embed = HookPoint()
        self.hook_tokens = HookPoint()
        self.hook_pos_embed = HookPoint()

    def __call__(
        self,
        input_ids: Int[jax.Array, "batch pos"],
        attention_mask: Optional[Int[jax.Array, "batch kv_pos"]] = None,
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

        tokens = self.hook_tokens(input_ids)  # [batch, pos]
        embed = self.hook_embed(self.embed(tokens))  # [batch, pos, d_model]
        self._check_kv_cache_consistency()  # Check that the KV cache is consistent
        past_kv_pos_offset = (
            0
            if self.blocks[0].attn.past_kv_cache.value is None
            else self.blocks[0].attn.past_kv_cache.value[0].shape[1]
        )
        pos_embed = self.hook_pos_embed(
            self.pos_embed(tokens, past_kv_pos_offset, attention_mask)
        )  # [batch, pos, d_model]
        residual = embed + pos_embed

        for block in self.blocks:
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

    def _check_kv_cache_consistency(self):
        """Check if the KV cache is consistent across blocks.

        This is to ensure that the KV cache is either:
        - None for all blocks
        - Non-None and has the same shape for all blocks
        """
        all_kv_cache_values = [block.attn.past_kv_cache.value for block in self.blocks]
        if any(kv_cache_value is not None for kv_cache_value in all_kv_cache_values):
            assert all(
                kv_cache_value is not None for kv_cache_value in all_kv_cache_values
            ), "All cached values must be non-None if any are set"
            first_cache = cast(tuple[jax.Array, jax.Array], all_kv_cache_values[0])
            first_k_shape = first_cache[0].shape
            for k_cache, v_cache in cast(list[tuple[jax.Array, jax.Array]], all_kv_cache_values):
                assert k_cache.shape == first_k_shape and v_cache.shape == first_k_shape, (
                    "All cached values must have the same shape. Found shapes %s and %s while first shape is %s"
                    % (k_cache.shape, v_cache.shape, first_k_shape)
                )

    def run_with_hooks(
        self,
        input_ids: Int[jax.Array, "batch pos"],
        attention_mask: Optional[jax.Array] = None,  # [batch pos]
        hooks: list[tuple[str, Callable[[Any], Any]]] = [],
    ) -> Float[jax.Array, "batch pos d_vocab"]:
        """Forward Pass with hooks.

        This is the same as the normal forward pass, but allows you to add hooks to the forward pass
        which can be used to extract intermediate values from the model.

        Args:
            attention_mask: Optional[jax.Array]: Override the attention mask used to ignore
                padded tokens. If start_at_layer is not None and (self.tokenizer.padding_side ==
                "left" or past_kv_cache is not None), this should be passed as the attention mask
                is not computed automatically. Defaults to None.
            hooks: list[tuple[str, Callable[[Any], Any]]]: A list of tuples, where the first element
                is the name of the hook, and the second element is a callable that takes in a value
                and returns a value. The callable should be a pure function, as it will be called
                multiple times. Defaults to [].
        """

        model = with_hooks(self, hooks)

        return model(input_ids, attention_mask=attention_mask)

    def run_with_cache(
        self,
        input_ids: Int[jax.Array, "batch pos"],
        attention_mask: Optional[jax.Array] = None,  # [batch pos]
        hook_names: list[str] = [],
    ) -> tuple[Float[jax.Array, "batch pos d_vocab"], dict[str, Any]]:
        """Forward Pass with cache.

        This is the same as the normal forward pass, but allows you to pass in a cache dictionary
        which can be used to store and retrieve intermediate values from the model.

        Args:
            attention_mask: Optional[jax.Array]: Override the attention mask used to ignore
                padded tokens. If start_at_layer is not None and (self.tokenizer.padding_side ==
                "left" or past_kv_cache is not None), this should be passed as the attention mask
                is not computed automatically. Defaults to None.
            hook_names: list[str]: A list of strings, where each string is the name of a hook point
        """

        model, cache = with_cache(self, hook_names)

        out = model(input_ids, attention_mask=attention_mask)

        return out, cache

    @classmethod
    def from_pretrained(cls, model_name: str, hf_model: Any = None) -> "HookedTransformer":
        """Load a pretrained model.

        Args:
            model_name: str: The name of the model to load.
            hf_model: Optionally, a HuggingFace model object. If provided, we will use
                these weights rather than reloading the model.
        """

        cfg = get_pretrained_model_config(model_name)
        weights = get_pretrained_weights(cfg, model_name, hf_model=hf_model)
        model = HookedTransformer(cfg)
        model = load_pretrained_weights(model, weights)
        return model
