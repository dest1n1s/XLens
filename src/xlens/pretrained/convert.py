"""Loading Pretrained Models Utilities.

This module contains functions for loading pretrained models from the Hugging Face Hub.
"""

from typing import Any

import jax

from xlens.config import HookedTransformerConfig
from xlens.pretrained.converters import (
    GPT2Converter,
    GPTNeoXConverter,
    LlamaConverter,
    MistralConverter,
    Qwen2Converter,
)
from xlens.pretrained.model_converter import HuggingFaceModelConverter

converter = HuggingFaceModelConverter(
    converters=[
        GPT2Converter(),
        Qwen2Converter(),
        LlamaConverter(),
        MistralConverter(),
        GPTNeoXConverter(),
    ]
)


def get_pretrained_model_config(model_name_or_path: str) -> HookedTransformerConfig:
    """Get the configuration for a pretrained model from Hugging Face.

    Args:
        model_name (str): The name of the model on Hugging Face Hub (e.g., 'gpt2', 'facebook/opt-125m')

    Returns:
        HookedTransformerConfig: Configuration object containing the model architecture details

    Raises:
        ValueError: If the model architecture is not supported
    """
    return converter.get_pretrained_model_config(model_name_or_path)


def get_pretrained_weights(
    cfg: HookedTransformerConfig, model_name_or_path: str, hf_model: Any = None
) -> dict[str, jax.Array]:
    """Load pretrained weights from a Hugging Face model and convert them to JAX arrays.

    Args:
        cfg (HookedTransformerConfig): Configuration object for the target model
        model_name (str): The name of the model on Hugging Face Hub (e.g., 'gpt2', 'facebook/opt-125m')
        hf_model (Any, optional): Pre-loaded Hugging Face model. If None, the model will be loaded
            from the Hub. Defaults to None.

    Returns:
        dict[str, jax.Array]: Dictionary mapping parameter names to their values as JAX arrays

    Raises:
        ValueError: If the model architecture is not supported or weights cannot be converted
    """
    return converter.get_pretrained_weights(cfg, model_name_or_path, hf_model=hf_model)
