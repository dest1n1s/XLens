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


def get_pretrained_model_config(model_name: str) -> HookedTransformerConfig:
    return converter.get_pretrained_model_config(model_name)


def get_pretrained_weights(cfg: HookedTransformerConfig, model_name: str, hf_model: Any = None) -> dict[str, jax.Array]:
    return converter.get_pretrained_weights(cfg, model_name, hf_model=hf_model)
