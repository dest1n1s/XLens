"""Hooked Transformer Config.

Module with a dataclass for storing the configuration of a
:class:`transformer_lens.HookedTransformer` model.
"""

from __future__ import annotations

import logging
import math
import pprint
from dataclasses import dataclass
from typing import Any, Optional

from xlens.utilities.activation_functions import SUPPORTED_ACTIVATIONS


@dataclass
class HookedTransformerConfig:
    """Configuration class to store the configuration of a HookedTransformer model."""

    d_model: int
    """The dimensionality of the embeddings."""

    d_head: int
    """The dimensionality of each attention head."""

    n_layers: int
    """The number of transformer blocks (one block = one attn layer AND one MLP layer)."""

    n_ctx: int
    """The maximum sequence length."""

    d_vocab: int
    """The size of the vocabulary."""

    n_heads: int = -1
    """The number of attention heads. If not specified, will be set to d_model // d_head. (This is represented by a default value of -1)"""

    d_mlp: Optional[int] = None
    """The dimensionality of the feedforward mlp network. Defaults to 4 * d_model, and in an attn-only model is None."""

    act_fn: Optional[str] = None
    """The activation function to use. Always lowercase. Supports ['relu', 'gelu', 'silu', 'gelu_new', 'solu_ln', 'gelu_fast']. 
    Must be set unless using an attn-only model."""

    eps: float = 1e-5
    """The epsilon value to use for layer normalization."""

    use_attn_scale: bool = True
    """Whether to scale the attention weights by 1/sqrt(d_head)"""

    attn_scale: float = -1.0
    """The amount to divide attention scores by (if applicable). Defaults to sqrt(d_head)"""

    use_local_attn: bool = False
    """Whether to use local attention - ie each destination token can only attend to source tokens a certain distance back."""

    window_size: Optional[int] = None
    """The size of the window for local attention"""

    attn_types: Optional[list[str]] = None
    """The types of attention to use for local attention"""

    model_name: str = "custom"
    """The name of the model, used to load weights from HuggingFace or initialized to "custom" if not passed"""

    original_architecture: Optional[str] = None
    """The original architecture of the model"""

    init_mode: str = "gpt2"
    """The initialization mode to use for the weights. Only relevant for custom models, ignored for pre-trained. We now support 'gpt2', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal'. MuP support to come. Defaults to 'gpt2'."""

    normalization_type: Optional[str] = "LN"
    """The type of normalization to use. Options are None (no normalization), 'LN' (use LayerNorm, including weights & biases) and 'LNPre' (use LayerNorm, but no weights or biases), 'RMS' (use RMSNorm, including weights) and 'RMSPre' (use RMSNorm, but no weights or biases). Defaults to LN"""

    attention_dir: str = "causal"
    """Whether to use causal (aka unidirectional aka GPT-2 style) or bidirectional attention. Options are 'causal' and 'bidirectional'. Defaults to 'causal'"""

    attn_only: bool = False
    """Whether to only use attention layers, no feedforward layers. Defaults to False"""

    initializer_range: float = -1.0
    """The standard deviation of the normal used to initialise the weights, initialized to 0.8 / sqrt(d_model). If init_mode is 'xavier_uniform' or 'xavier_normal', this value is instead treated as the `gain` parameter for the weight initialisation (a constant factor to scale the weights by). Defaults to -1.0, which means not set."""

    scale_attn_by_inverse_layer_idx: bool = False
    """Whether to scale the attention weights by 1/(layer_id+1), used by Mistral (Stanford) models for numerical stability when training in FP16. Defaults to False."""

    positional_embedding_type: str = "standard"
    """The positional embedding used. Options are 'standard' (ie GPT-2 style, absolute, randomly initialized learned positional embeddings, directly added to the residual stream) and 'rotary' (described here: https://blog.eleuther.ai/rotary-embeddings/ ). Sinusoidal and Shortformer are not currently supported. Defaults to 'standard'."""

    final_rms: bool = False
    """Whether to replace the final normalization (just before the unembed) with RMSNorm (ie no centering or bias, just scaling + weights). Only included because of a dumb bug in my original SoLU code. Defaults to False."""

    d_vocab_out: int = -1
    """The size of the output vocabulary. Defaults to -1, which means not set. If not set, will be equal to d_vocab. Mainly useful for algorithmic tasks where the input and output vocabularies may be different."""

    parallel_attn_mlp: bool = False
    """Whether to parallelize the attention and MLP layers - a weird cursed thing done by GPT-J. Means that mlp_out=MLP(ln1(resid_pre)) and resid_post=resid_pre+attn_out+mlp_out. Defaults to False."""

    rotary_dim: Optional[int] = None
    """The dimensionality of the rotary embeddings, may be d_head in which case only the first rotary_dim dimensions of each head are rotated. Defaults to None, if positional_embedding_type=="rotary" post-init then sets it to d_head, i.e. "rotate all dimensions of the query and key"."""

    rotary_base: int = 10000
    """The base value for the rotary embeddings."""

    rotary_adjacent_pairs: bool = False
    """Whether to use adjacent pairs for the rotary embeddings."""

    gated_mlp: bool = False
    """Whether to use gated MLP layers."""

    n_key_value_heads: Optional[int] = None
    """The number of groups of heads that use the same key and value matrix. Only for models that use Grouped Query Attention."""

    post_embedding_ln: bool = False
    """Whether to apply layer normalization after embedding the tokens. Defaults to False."""

    use_NTK_by_parts_rope: bool = False
    """Whether to apply the "NTK-by-parts" method when using Rotary Positional Embedding. This method adjusts the interpolation based on frequency factors for different parts of the hidden dimensions. See Section 3.2 in https://arxiv.org/pdf/2309.00071 for details. Defaults to False."""

    NTK_by_parts_low_freq_factor: float = 1.0
    """The threshold applied to low-frequency hidden dimensions during interpolation when using the "NTK-by-parts" method. Defaults to 1.0."""

    NTK_by_parts_high_freq_factor: float = 4.0
    """The threshold applied to high-frequency hidden dimensions during interpolation in the "NTK-by-parts" method. Defaults to 4.0."""

    NTK_by_parts_factor: float = 8.0
    """The overall factor used in the "NTK-by-parts" method that affects the rate of change between low and high-frequency interpolation strategies. Defaults to 8.0."""

    def __post_init__(self):
        if self.n_heads == -1:
            self.n_heads = self.d_model // self.d_head

            if not self.d_model % (self.d_head) == 0:
                logging.warning(
                    "d_model %d is not divisible by d_head %d."
                    "n_heads was inferred to be %d, rounding down the ratio.",
                    self.d_model,
                    self.d_head,
                    self.n_heads,
                )

        if not self.attn_only:
            if self.d_mlp is None:
                # For some reason everyone hard codes in this hyper-parameter!
                self.d_mlp = self.d_model * 4
            assert self.act_fn is not None, "act_fn must be specified for non-attn-only models"
            assert self.act_fn in SUPPORTED_ACTIVATIONS, f"act_fn={self.act_fn} must be one of {SUPPORTED_ACTIVATIONS}"

        if self.initializer_range < 0 and self.init_mode == "gpt2":
            # Roughly copy the GPT-2 value, but proportional to sqrt(1/d_model)
            self.initializer_range = 0.8 / math.sqrt(self.d_model)

        if self.initializer_range < 0 and self.init_mode != "gpt2":
            # This is the gain parameter for the weight initialisation
            self.initializer_range = 1.0

        if self.d_vocab_out == -1:
            # d_vocab_out defaults to d_vocab, unless there's an algorithmic task
            self.d_vocab_out = self.d_vocab

        if self.positional_embedding_type == "rotary" and self.rotary_dim is None:
            self.rotary_dim = self.d_head

        if self.use_attn_scale and self.attn_scale == -1.0:
            self.attn_scale = math.sqrt(self.d_head)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> HookedTransformerConfig:
        """
        Instantiates a `HookedTransformerConfig` from a Python dictionary of
        parameters.
        """
        return cls(**config_dict)

    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        return "HookedTransformerConfig:\n" + pprint.pformat(self.to_dict())

    def is_layer_norm_activation(self) -> bool:
        return self.act_fn is not None and self.act_fn.endswith("_ln")
