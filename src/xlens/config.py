"""Hooked Transformer Config.

Module with a dataclass for storing the configuration of a
:class:`transformer_lens.HookedTransformer` model.
"""

from __future__ import annotations

import logging
import pprint
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

from xlens.utilities.activation_functions import SUPPORTED_ACTIVATIONS


@dataclass
class HookedTransformerConfig:
    """
    Configuration class to store the configuration of a HookedTransformer model.

    See further_comments.md for more details on the more complex arguments.

    Args:
        model_name (str): the name of the model, used to load
            weights from HuggingFace or initialized to "custom" if not passed
        d_model (int): The dimensionality of the embeddings.
        d_head (int): The dimensionality of each attention head.
        n_layers (int): The number of transformer blocks (one block = one attn layer AND one MLP layer).
        n_ctx (int): The maximum sequence length.
        n_heads (int): The number of attention heads. If not
            specified, will be set to d_model // d_head. (This is represented by a default value of -1)
        d_mlp (int, *optional*): The dimensionality of the feedforward mlp
            network. Defaults to 4 * d_model, and in an attn-only model is None.
        d_vocab (int): The size of the vocabulary. Defaults to -1, which means not set. If not set, will be
            automatically set from the tokenizer's vocab size.
        act_fn (str, *optional*): The activation function to use. Always
            lowercase. Supports ['relu', 'gelu', 'silu', 'gelu_new', 'solu_ln',
            'gelu_fast']. Must be set unless using an attn-only model.
        eps (float): The epsilon value to use for layer normalization. Defaults
            to 1e-5
        use_attn_scale (bool): whether to scale the attention weights by
            1/sqrt(d_head)
        attn_scale (float): The amount to divide attention scores by (if applicable). Defaults to
            sqrt(d_head)
        tokenizer_name (str, *optional*): the full name of the model, passed into
            HuggingFace to access the tokenizer. Only used when passing in
            custom config, if loading from pretrained then this is not needed.
        window_size (int, *optional*): the size of the window for local
            attention
        attn_types (List[str], *optional*): the types of attention to use for
            local attention
        init_mode (str): the initialization mode to use for the
            weights. Only relevant for custom models, ignored for pre-trained.
            We now support 'gpt2', 'xavier_uniform', 'xavier_normal', 'kaiming_uniform',
            'kaiming_normal'. MuP support to come. Defaults to 'gpt2'.
        normalization_type (str, *optional*): the type of normalization to use.
            Options are None (no normalization), 'LN' (use LayerNorm, including weights
            & biases) and 'LNPre' (use LayerNorm, but no weights or biases), 'RMS'
            (use RMSNorm, including weights) and 'RMSPre' (use RMSNorm, but no weights or biases).
            Defaults to LN
        attention_dir (str): Whether to use causal (aka unidirectional aka GPT-2
            style) or bidirectional attention. Options are 'causal' and
            'bidirectional'. Defaults to 'causal'
        attn_only (bool): Whether to only use attention layers, no feedforward
            layers. Defaults to False
        initializer_range (float): The standard deviation of the normal used to
            initialise the weights, initialized to 0.8 / sqrt(d_model). If init_mode is
            'xavier_uniform' or 'xavier_normal', this value is instead treated as the `gain` parameter for the weight
            initialisation (a constant factor to scale the weights by). Defaults to -1.0, which means not set.
        scale_attn_by_inverse_layer_idx (bool): Whether to scale the attention
            weights by 1/(layer_id+1), used by Mistral (Stanford) models for numerical stability when
            training in FP16. Defaults to False.
        d_vocab_out (int, *optional*): The size of the output vocabulary. Defaults to -1, which means not set. If not
            set, will be equal to d_vocab. Mainly useful for algorithmic tasks
            where the input and output vocabularies may be different.
        default_prepend_bos (bool, optional): Default behavior of whether to prepend the BOS token when the
            methods of HookedTransformer process input text to tokenize (only when input is a string).
            Defaults to True - even for models not explicitly trained with this, heads often use the
            first position as a resting position and accordingly lose information from the first token,
            so this empirically seems to give better results. To change the default behavior to False, pass in
            default_prepend_bos=False. Note that you can also locally override the default behavior by passing
            in prepend_bos=True/False when you call a method that processes the input string.
        tokenizer_prepends_bos (bool, *optional*): This flag is set by set_tokenizer. It is set to True only
            when the tokenizer automatically prepends the BOS token if initialized with add_bos_token=True.
            We need this information to dynamically control bos prepending.
        post_embedding_ln (bool): Whether to apply layer normalization after embedding the tokens. Defaults
            to False.
    """

    d_model: int
    d_head: int
    n_layers: int
    n_ctx: int
    n_heads: int = -1
    d_mlp: Optional[int] = None
    d_vocab: int = -1
    act_fn: Optional[str] = None
    eps: float = 1e-5
    use_attn_scale: bool = True
    attn_scale: float = -1.0
    model_name: str = "custom"
    original_architecture: Optional[str] = None
    tokenizer_name: Optional[str] = None
    window_size: Optional[int] = None
    attn_types: Optional[List] = None
    init_mode: str = "gpt2"
    normalization_type: Optional[str] = "LN"
    attention_dir: str = "causal"
    attn_only: bool = False
    initializer_range: float = -1.0
    scale_attn_by_inverse_layer_idx: bool = False
    d_vocab_out: int = -1
    default_prepend_bos: bool = True
    tokenizer_prepends_bos: Optional[bool] = None
    post_embedding_ln: bool = False

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
                self.d_mlp: int = self.d_model * 4
            assert self.act_fn is not None, "act_fn must be specified for non-attn-only models"
            assert self.act_fn in SUPPORTED_ACTIVATIONS, f"act_fn={self.act_fn} must be one of {SUPPORTED_ACTIVATIONS}"

        if self.initializer_range < 0 and self.init_mode == "gpt2":
            # Roughly copy the GPT-2 value, but proportional to sqrt(1/d_model)
            self.initializer_range = 0.8 / np.sqrt(self.d_model)

        if self.initializer_range < 0 and self.init_mode != "gpt2":
            # This is the gain parameter for the weight initialisation
            self.initializer_range = 1.0

        if self.d_vocab_out == -1:
            # d_vocab_out defaults to d_vocab, unless there's an algorithmic task
            # If d_vocab is not set, it'll be inferred from tokenizer_name or from a tokenizer
            # explicitly passed to HookedTransformer initialisation.
            self.d_vocab_out = self.d_vocab

        if self.use_attn_scale and self.attn_scale == -1.0:
            self.attn_scale = np.sqrt(self.d_head)

        assert self.default_prepend_bos in [
            True,
            False,
        ], f"padding_side must be either True or False, but {self.default_prepend_bos} is given"

    @classmethod
    def unwrap(cls, config: Union[Dict, "HookedTransformerConfig"]) -> HookedTransformerConfig:
        """
        Convenience function to avoid duplicate code from a common way config is passed to various components
        """
        return HookedTransformerConfig.from_dict(config) if isinstance(config, Dict) else config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> HookedTransformerConfig:
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
