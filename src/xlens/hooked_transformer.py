import logging
from functools import partial
from typing import Any, Callable, Optional, Self, TypeVar, Union

import flax.nnx as nnx
import jax
import jax.numpy as jnp
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
from xlens.hooks.utilities import retrieve_cache
from xlens.pretrained.convert import get_pretrained_model_config, get_pretrained_weights
from xlens.utilities.functional import functional
from xlens.utils import load_pretrained_weights

from .config import HookedTransformerConfig
from .hooks import HookPoint

U = TypeVar("U")
LayerNormLike = Union[LayerNorm, LayerNormPre, RMSNorm, RMSNormPre]
CarryType = tuple[int, Int[jax.Array, "batch generated_pos"], tuple[nnx.GraphDef[U], nnx.GraphState]]


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

    @functional
    def __call__(
        self,
        input_ids: Int[jax.Array, "batch pos"],
        attention_mask: Optional[Int[jax.Array, "batch pos"]] = None,
    ) -> tuple[Float[jax.Array, "batch pos d_vocab"], Self]:
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

        tokens, self.hook_tokens = self.hook_tokens(input_ids)  # [batch, pos]
        embed, self.embed = self.embed(tokens)  # [batch, pos, d_model]
        embed, self.hook_embed = self.hook_embed(embed)  # [batch, pos, d_model]
        self._check_kv_cache_consistency()  # Check that the KV cache is consistent
        past_kv_pos_offset = self.blocks[0].attn.past_kv_cache.length
        pos_embed, self.hook_pos_embed = self.hook_pos_embed(
            self.pos_embed(tokens, past_kv_pos_offset, attention_mask)
        )  # [batch, pos, d_model]
        residual = embed + pos_embed

        for i, block in enumerate(self.blocks):
            # Note that each block includes skip connections, so we don't need
            # residual + block(residual)
            residual, block = block(
                residual,
                attention_mask=attention_mask,
            )  # [batch, pos, d_model]
            self.blocks[i] = block

        if self.cfg.normalization_type is not None:
            assert self.ln_final is not None, "ln_final should be set if normalization_type is set"
            residual, self.ln_final = self.ln_final(residual)  # [batch, pos, d_model]

        logits = self.unembed(residual)  # [batch, pos, d_vocab]
        return logits, self

    # @checkify.checkify
    def _check_kv_cache_consistency(self):
        """Check if the KV cache is consistent across blocks.

        This is to ensure that the KV cache is either:
        - None for all blocks
        - Non-None and has the same shape for all blocks
        """
        # all_kv_cache_lengths = [block.attn.past_kv_cache.length for block in self.blocks]
        # checkify.check(
        #     jnp.all(
        #         jnp.array([kv_cache_length == all_kv_cache_lengths[0] for kv_cache_length in all_kv_cache_lengths])
        #     ),
        #     "All KV cache lengths must be the same",
        # )
        pass

    def run_with_hooks(
        self,
        input_ids: Int[jax.Array, "batch pos"],
        attention_mask: Optional[jax.Array] = None,  # [batch pos]
        hooks: list[tuple[str, Callable[[Any, Any], tuple[Any, Any]]]] = [],
    ) -> tuple[Float[jax.Array, "batch pos d_vocab"], Self]:
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
    ) -> tuple[Float[jax.Array, "batch pos d_vocab"], dict[str, Any], Self]:
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

        model = with_cache(self, hook_names)

        out, model = model(input_ids, attention_mask=attention_mask)
        cache = retrieve_cache(model, hook_names)

        return out, cache, model

    @classmethod
    @functional(transform=partial(jax.jit, static_argnums=(0, 1, 2)))
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

    @functional
    def generate(
        self,
        input_ids: Int[jax.Array, "batch pos"],
        eos_token_id: int,
        top_k: int = 5,
        top_p: float = 0.95,
        rng: Optional[jax.Array] = None,
    ) -> tuple[Float[jax.Array, "batch generated_pos"], "HookedTransformer"]:
        """Generate tokens from the model.

        Args:
            input_ids: Int[jax.Array, "batch pos"]: The input tokens to generate from.
            eos_token_id: int: The token id to use as an end-of-sequence token.
            top_k: int: The number of top tokens to consider for sampling.
            top_p: float: The cumulative probability threshold for top-p sampling.
            rng: Optional[jax.random.KeyArray]: A random number generator key. If not provided, a
                new key will be created.
        """

        if rng is None:
            rng = jax.random.PRNGKey(0)

        def sample_next_token(
            logits: Float[jax.Array, "batch d_vocab"], top_k: int, top_p: float, rng: jax.Array
        ) -> Int[jax.Array, " batch"]:
            # Get top k logits and indices
            top_logits, top_indices = jax.lax.top_k(logits, top_k)

            # Apply softmax to get probabilities
            probs = jax.nn.softmax(top_logits, axis=-1)

            # Apply top-p (nucleus) sampling
            sorted_probs = jnp.sort(probs, axis=-1, descending=True)
            cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
            probs = jnp.where(cumulative_probs > top_p, 0.0, probs)
            probs = probs / probs.sum(axis=-1, keepdims=True)

            # Sample from the filtered distribution
            next_token = jax.random.categorical(rng, jnp.log(probs))[:, None]

            return jnp.take_along_axis(top_indices, next_token, axis=1)

        @functional
        def cond_fn(carry: CarryType[Self]) -> jax.Array:
            i, current_ids, _ = carry
            return jnp.logical_and(i < self.cfg.n_ctx, ~jnp.any(current_ids[:, i] == eos_token_id))

        @functional
        def body_fn(
            carry: CarryType[Self],
        ) -> CarryType[Self]:
            i, current_ids, (graph_def, state) = carry
            model = nnx.merge(graph_def, state)
            # Get logits for the last token
            logits, model = model(current_ids[:, i - 1][:, None])
            next_token_logits = logits[:, -1, :]

            # Sample next token
            next_token = sample_next_token(next_token_logits, top_k, top_p, jax.random.fold_in(rng, i))

            # Append new token
            current_ids = jax.lax.dynamic_update_slice(current_ids, next_token, (0, i))
            return i + 1, current_ids, nnx.split(model)

        # First iteration should be separately handled
        logits, model = self(input_ids)
        next_token = sample_next_token(logits[:, -1, :], top_k, top_p, jax.random.fold_in(rng, 0))
        current_ids = jnp.zeros((input_ids.shape[0], self.cfg.n_ctx), dtype=jnp.int32)
        current_ids = jax.lax.dynamic_update_slice(current_ids, input_ids, (0, 0))
        current_ids = jax.lax.dynamic_update_slice(current_ids, next_token, (0, input_ids.shape[1]))

        # Initialize loop variables
        init_carry = (input_ids.shape[1] + 1, current_ids, nnx.split(model))

        # Run the generation loop
        _, generated_ids, (graph_def, state) = jax.lax.while_loop(cond_fn, body_fn, init_carry)
        model = nnx.merge(graph_def, state)

        return generated_ids, model
