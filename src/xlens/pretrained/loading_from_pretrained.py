"""Loading Pretrained Models Utilities.

This module contains functions for loading pretrained models from the Hugging Face Hub.
"""

import logging
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from safetensors.flax import load_file as safe_load_file
from transformers import AutoConfig
from transformers.utils import SAFE_WEIGHTS_NAME, cached_file

from xlens.config import HookedTransformerConfig
from xlens.pretrained.convert_weight import (
    convert_gpt2_weights,
    convert_llama_weights,
    convert_mistral_weights,
    convert_neox_weights,
    convert_qwen2_weights,
)
from xlens.utils import flatten_dict

OFFICIAL_MODEL_NAMES = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "EleutherAI/pythia-14m",
    "EleutherAI/pythia-31m",
    "EleutherAI/pythia-70m",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-410m",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-1.4b",
    "EleutherAI/pythia-2.8b",
    "EleutherAI/pythia-6.9b",
    "EleutherAI/pythia-12b",
    "EleutherAI/pythia-70m-deduped",
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-1b-deduped",
    "EleutherAI/pythia-1.4b-deduped",
    "EleutherAI/pythia-2.8b-deduped",
    "EleutherAI/pythia-6.9b-deduped",
    "EleutherAI/pythia-12b-deduped",
    "EleutherAI/pythia-70m-v0",
    "EleutherAI/pythia-160m-v0",
    "EleutherAI/pythia-410m-v0",
    "EleutherAI/pythia-1b-v0",
    "EleutherAI/pythia-1.4b-v0",
    "EleutherAI/pythia-2.8b-v0",
    "EleutherAI/pythia-6.9b-v0",
    "EleutherAI/pythia-12b-v0",
    "EleutherAI/pythia-70m-deduped-v0",
    "EleutherAI/pythia-160m-deduped-v0",
    "EleutherAI/pythia-410m-deduped-v0",
    "EleutherAI/pythia-1b-deduped-v0",
    "EleutherAI/pythia-1.4b-deduped-v0",
    "EleutherAI/pythia-2.8b-deduped-v0",
    "EleutherAI/pythia-6.9b-deduped-v0",
    "EleutherAI/pythia-12b-deduped-v0",
    "EleutherAI/pythia-160m-seed1",
    "EleutherAI/pythia-160m-seed2",
    "EleutherAI/pythia-160m-seed3",
    "llama-7b-hf",
    "llama-13b-hf",
    "llama-30b-hf",
    "llama-65b-hf",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B",
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-70B",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mistral-Nemo-Base-2407",
    "Qwen/Qwen-1_8B",
    "Qwen/Qwen-7B",
    "Qwen/Qwen-14B",
    "Qwen/Qwen-1_8B-Chat",
    "Qwen/Qwen-7B-Chat",
    "Qwen/Qwen-14B-Chat",
    "Qwen/Qwen1.5-0.5B",
    "Qwen/Qwen1.5-0.5B-Chat",
    "Qwen/Qwen1.5-1.8B",
    "Qwen/Qwen1.5-1.8B-Chat",
    "Qwen/Qwen1.5-4B",
    "Qwen/Qwen1.5-4B-Chat",
    "Qwen/Qwen1.5-7B",
    "Qwen/Qwen1.5-7B-Chat",
    "Qwen/Qwen1.5-14B",
    "Qwen/Qwen1.5-14B-Chat",
    "Qwen/Qwen2-0.5B",
    "Qwen/Qwen2-0.5B-Instruct",
    "Qwen/Qwen2-1.5B",
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-7B",
    "Qwen/Qwen2-7B-Instruct",
]
"""Official model names for models on HuggingFace."""

# Model Aliases:
MODEL_ALIASES = {
    "EleutherAI/pythia-14m": [
        "pythia-14m",
    ],
    "EleutherAI/pythia-31m": [
        "pythia-31m",
    ],
    "EleutherAI/pythia-70m": [
        "pythia-70m",
        "pythia",
        "EleutherAI/pythia-19m",
        "pythia-19m",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-160m": [
        "pythia-160m",
        "EleutherAI/pythia-125m",
        "pythia-125m",  # EleutherAI renamed this model"
    ],
    "EleutherAI/pythia-410m": [
        "pythia-410m",
        "EleutherAI/pythia-350m",
        "pythia-350m",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-1b": [
        "pythia-1b",
        "EleutherAI/pythia-800m",
        "pythia-800m",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-1.4b": [
        "pythia-1.4b",
        "EleutherAI/pythia-1.3b",
        "pythia-1.3b",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-2.8b": [
        "pythia-2.8b",
        "EleutherAI/pythia-2.7b",
        "pythia-2.7b",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-6.9b": [
        "pythia-6.9b",
        "EleutherAI/pythia-6.7b",
        "pythia-6.7b",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-12b": [
        "pythia-12b",
        "EleutherAI/pythia-13b",
        "pythia-13b",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-70m-deduped": [
        "pythia-70m-deduped",
        "EleutherAI/pythia-19m-deduped",  # EleutherAI renamed this model
        "pythia-19m-deduped",
    ],
    "EleutherAI/pythia-160m-deduped": [
        "pythia-160m-deduped",
        "EleutherAI/pythia-125m-deduped",  # EleutherAI renamed this model
        "pythia-125m-deduped",
    ],
    "EleutherAI/pythia-410m-deduped": [
        "pythia-410m-deduped",
        "EleutherAI/pythia-350m-deduped",  # EleutherAI renamed this model
        "pythia-350m-deduped",
    ],
    "EleutherAI/pythia-1b-deduped": [
        "pythia-1b-deduped",
        "EleutherAI/pythia-800m-deduped",  # EleutherAI renamed this model
        "pythia-800m-deduped",
    ],
    "EleutherAI/pythia-1.4b-deduped": [
        "pythia-1.4b-deduped",
        "EleutherAI/pythia-1.3b-deduped",  # EleutherAI renamed this model
        "pythia-1.3b-deduped",
    ],
    "EleutherAI/pythia-2.8b-deduped": [
        "pythia-2.8b-deduped",
        "EleutherAI/pythia-2.7b-deduped",  # EleutherAI renamed this model
        "pythia-2.7b-deduped",
    ],
    "EleutherAI/pythia-6.9b-deduped": [
        "pythia-6.9b-deduped",
        "EleutherAI/pythia-6.7b-deduped",  # EleutherAI renamed this model
        "pythia-6.7b-deduped",
    ],
    "EleutherAI/pythia-12b-deduped": [
        "pythia-12b-deduped",
        "EleutherAI/pythia-13b-deduped",  # EleutherAI renamed this model
        "pythia-13b-deduped",
    ],
    "EleutherAI/pythia-70m-v0": [
        "pythia-70m-v0",
        "pythia-v0",
        "EleutherAI/pythia-19m-v0",
        "pythia-19m-v0",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-160m-v0": [
        "pythia-160m-v0",
        "EleutherAI/pythia-125m-v0",
        "pythia-125m-v0",  # EleutherAI renamed this model"
    ],
    "EleutherAI/pythia-410m-v0": [
        "pythia-410m-v0",
        "EleutherAI/pythia-350m-v0",
        "pythia-350m-v0",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-1b-v0": [
        "pythia-1b-v0",
        "EleutherAI/pythia-800m-v0",
        "pythia-800m-v0",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-1.4b-v0": [
        "pythia-1.4b-v0",
        "EleutherAI/pythia-1.3b-v0",
        "pythia-1.3b-v0",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-2.8b-v0": [
        "pythia-2.8b-v0",
        "EleutherAI/pythia-2.7b-v0",
        "pythia-2.7b-v0",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-6.9b-v0": [
        "pythia-6.9b-v0",
        "EleutherAI/pythia-6.7b-v0",
        "pythia-6.7b-v0",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-12b-v0": [
        "pythia-12b-v0",
        "EleutherAI/pythia-13b-v0",
        "pythia-13b-v0",  # EleutherAI renamed this model
    ],
    "EleutherAI/pythia-70m-deduped-v0": [
        "pythia-70m-deduped-v0",
        "EleutherAI/pythia-19m-deduped-v0",  # EleutherAI renamed this model
        "pythia-19m-deduped-v0",
    ],
    "EleutherAI/pythia-160m-deduped-v0": [
        "pythia-160m-deduped-v0",
        "EleutherAI/pythia-125m-deduped-v0",  # EleutherAI renamed this model
        "pythia-125m-deduped-v0",
    ],
    "EleutherAI/pythia-410m-deduped-v0": [
        "pythia-410m-deduped-v0",
        "EleutherAI/pythia-350m-deduped-v0",  # EleutherAI renamed this model
        "pythia-350m-deduped-v0",
    ],
    "EleutherAI/pythia-1b-deduped-v0": [
        "pythia-1b-deduped-v0",
        "EleutherAI/pythia-800m-deduped-v0",  # EleutherAI renamed this model
        "pythia-800m-deduped-v0",
    ],
    "EleutherAI/pythia-1.4b-deduped-v0": [
        "pythia-1.4b-deduped-v0",
        "EleutherAI/pythia-1.3b-deduped-v0",  # EleutherAI renamed this model
        "pythia-1.3b-deduped-v0",
    ],
    "EleutherAI/pythia-2.8b-deduped-v0": [
        "pythia-2.8b-deduped-v0",
        "EleutherAI/pythia-2.7b-deduped-v0",  # EleutherAI renamed this model
        "pythia-2.7b-deduped-v0",
    ],
    "EleutherAI/pythia-6.9b-deduped-v0": [
        "pythia-6.9b-deduped-v0",
        "EleutherAI/pythia-6.7b-deduped-v0",  # EleutherAI renamed this model
        "pythia-6.7b-deduped-v0",
    ],
    "EleutherAI/pythia-12b-deduped-v0": [
        "pythia-12b-deduped-v0",
        "EleutherAI/pythia-13b-deduped-v0",  # EleutherAI renamed this model
        "pythia-13b-deduped-v0",
    ],
    "EleutherAI/pythia-160m-seed1": [
        "pythia-160m-seed1",
        "EleutherAI/pythia-125m-seed1",
        "pythia-125m-seed1",  # EleutherAI renamed this model"
    ],
    "EleutherAI/pythia-160m-seed2": [
        "pythia-160m-seed2",
        "EleutherAI/pythia-125m-seed2",
        "pythia-125m-seed2",  # EleutherAI renamed this model"
    ],
    "EleutherAI/pythia-160m-seed3": [
        "pythia-160m-seed3",
        "EleutherAI/pythia-125m-seed3",
        "pythia-125m-seed3",  # EleutherAI renamed this model"
    ],
    "mistralai/Mistral-7B-v0.1": ["mistral-7b"],
    "mistralai/Mistral-7B-Instruct-v0.1": ["mistral-7b-instruct"],
    "mistralai/Mistral-Nemo-Base-2407": ["mistral-nemo-base-2407"],
    "gpt2": ["gpt2-small"],
    "llama-7b-hf": ["llama-7b"],
    "llama-13b-hf": ["llama-13b"],
    "llama-30b-hf": ["llama-30b"],
    "llama-65b-hf": ["llama-65b"],
    "meta-llama/Llama-2-7b-hf": ["Llama-2-7b", "meta-llama/Llama-2-7b-hf"],
    "meta-llama/Llama-2-7b-chat-hf": [
        "Llama-2-7b-chat",
        "meta-llama/Llama-2-7b-chat-hf",
    ],
    "meta-llama/Llama-2-13b-hf": ["Llama-2-13b", "meta-llama/Llama-2-13b-hf"],
    "meta-llama/Llama-2-13b-chat-hf": [
        "Llama-2-13b-chat",
        "meta-llama/Llama-2-13b-chat-hf",
    ],
    "meta-llama/Llama-2-70b-chat-hf": ["Llama-2-70b-chat", "meta-llama-2-70b-chat-hf"],
    "Qwen/Qwen-1_8B": ["qwen-1.8b"],
    "Qwen/Qwen-7B": ["qwen-7b"],
    "Qwen/Qwen-14B": ["qwen-14b"],
    "Qwen/Qwen-1_8B-Chat": ["qwen-1.8b-chat"],
    "Qwen/Qwen-7B-Chat": ["qwen-7b-chat"],
    "Qwen/Qwen-14B-Chat": ["qwen-14b-chat"],
    "Qwen/Qwen1.5-0.5B": ["qwen1.5-0.5b"],
    "Qwen/Qwen1.5-0.5B-Chat": ["qwen1.5-0.5b-chat"],
    "Qwen/Qwen1.5-1.8B": ["qwen1.5-1.8b"],
    "Qwen/Qwen1.5-1.8B-Chat": ["qwen1.5-1.8b-chat"],
    "Qwen/Qwen1.5-4B": ["qwen1.5-4b"],
    "Qwen/Qwen1.5-4B-Chat": ["qwen1.5-4b-chat"],
    "Qwen/Qwen1.5-7B": ["qwen1.5-7b"],
    "Qwen/Qwen1.5-7B-Chat": ["qwen1.5-7b-chat"],
    "Qwen/Qwen1.5-14B": ["qwen1.5-14b"],
    "Qwen/Qwen1.5-14B-Chat": ["qwen1.5-14b-chat"],
}
"""Model aliases for models on HuggingFace."""

NON_HF_HOSTED_MODEL_NAMES = [
    "llama-7b-hf",
    "llama-13b-hf",
    "llama-30b-hf",
    "llama-65b-hf",
]
"""Official model names for models not hosted on HuggingFace."""

# Sets a default model alias, by convention the first one in the model alias table, else the official name if it has no aliases
DEFAULT_MODEL_ALIASES = [MODEL_ALIASES[name][0] if name in MODEL_ALIASES else name for name in OFFICIAL_MODEL_NAMES]

NEED_REMOTE_CODE_MODELS = ("Qwen/Qwen-",)


def make_model_alias_map():
    """
    Converts OFFICIAL_MODEL_NAMES (the list of actual model names on
    HuggingFace) and MODEL_ALIASES (a dictionary mapping official model names to
    aliases) into a dictionary mapping all aliases to the official model name.
    """
    model_alias_map = {}
    for official_model_name in OFFICIAL_MODEL_NAMES:
        aliases = MODEL_ALIASES.get(official_model_name, [])
        for alias in aliases:
            model_alias_map[alias.lower()] = official_model_name
        model_alias_map[official_model_name.lower()] = official_model_name
    return model_alias_map


def get_official_model_name(model_name: str):
    """
    Returns the official model name for a given model name (or alias).
    """
    model_alias_map = make_model_alias_map()
    official_model_name = model_alias_map.get(model_name.lower(), None)
    if official_model_name is None:
        raise ValueError(f"{model_name} not found. Valid official model names (excl aliases): {OFFICIAL_MODEL_NAMES}")
    return official_model_name


def convert_hf_model_config(model_name: str, **kwargs):
    """
    Returns the model config for a HuggingFace model, converted to a dictionary
    in the HookedTransformerConfig format.

    Takes the official_model_name as an input.
    """
    # In case the user passed in an alias
    if (Path(model_name) / "config.json").exists():
        logging.info("Loading model config from local directory")
        official_model_name = model_name
    else:
        official_model_name = get_official_model_name(model_name)

    # Load HuggingFace model config
    if "llama" in official_model_name.lower():
        architecture = "LlamaForCausalLM"
    else:
        huggingface_token = os.environ.get("HF_TOKEN", None)
        hf_config = AutoConfig.from_pretrained(
            official_model_name,
            token=huggingface_token,
            **kwargs,
        )
        architecture = hf_config.architectures[0]

    if official_model_name.startswith(("llama-7b", "meta-llama/Llama-2-7b")):  # same architecture for LLaMA and Llama-2
        cfg_dict: dict[str, Any] = {
            "d_model": 4096,
            "d_head": 4096 // 32,
            "n_heads": 32,
            "d_mlp": 11008,
            "n_layers": 32,
            "n_ctx": 2048 if official_model_name.startswith("llama-7b") else 4096,
            "eps": 1e-6 if official_model_name.startswith("llama-7b") else 1e-5,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 4096 // 32,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif official_model_name.startswith("codellama"):  # same architecture CodeLlama and Llama-2
        cfg_dict = {
            "d_model": 4096,
            "d_head": 4096 // 32,
            "n_heads": 32,
            "d_mlp": 11008,
            "n_layers": 32,
            "n_ctx": 4096,
            "eps": 1e-5,
            "d_vocab": 32016,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": 4096 // 32,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 1000000,
        }
        if "python" in official_model_name.lower():
            # The vocab size of python version of CodeLlama-7b is 32000
            cfg_dict["d_vocab"] = 32000
    elif official_model_name.startswith(
        ("llama-13b", "meta-llama/Llama-2-13b")
    ):  # same architecture for LLaMA and Llama-2
        cfg_dict = {
            "d_model": 5120,
            "d_head": 5120 // 40,
            "n_heads": 40,
            "d_mlp": 13824,
            "n_layers": 40,
            "n_ctx": 2048 if official_model_name.startswith("llama-13b") else 4096,
            "eps": 1e-6 if official_model_name.startswith("llama-13b") else 1e-5,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 5120 // 40,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "llama-30b" in official_model_name:
        cfg_dict = {
            "d_model": 6656,
            "d_head": 6656 // 52,
            "n_heads": 52,
            "d_mlp": 17920,
            "n_layers": 60,
            "n_ctx": 2048,
            "eps": 1e-6,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 6656 // 52,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "llama-65b" in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 8192 // 64,
            "n_heads": 64,
            "d_mlp": 22016,
            "n_layers": 80,
            "n_ctx": 2048,
            "eps": 1e-6,
            "d_vocab": 32000,
            "act_fn": "silu",
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": 8192 // 64,
            "rotary_adjacent_pairs": False,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "Llama-2-70b" in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 128,
            "n_heads": 64,
            "d_mlp": 28672,
            "n_layers": 80,
            "n_ctx": 4096,
            "eps": 1e-5,
            "d_vocab": 32000,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif "Meta-Llama-3-8B" in official_model_name:
        cfg_dict = {
            "d_model": 4096,
            "d_head": 128,
            "n_heads": 32,
            "d_mlp": 14336,
            "n_layers": 32,
            "n_ctx": 8192,
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
        }
    elif "Meta-Llama-3-70B" in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 128,
            "n_heads": 64,
            "d_mlp": 28672,
            "n_layers": 80,
            "n_ctx": 8192,
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
        }
    elif "Llama-3.2-1B" in official_model_name:
        cfg_dict = {
            "d_model": 2048,
            "d_head": 64,
            "n_heads": 32,
            "d_mlp": 8192,
            "n_layers": 16,
            "n_ctx": 2048,  # capped due to memory issues
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 64,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
            "use_NTK_by_parts_rope": True,
            "NTK_by_parts_low_freq_factor": 1.0,
            "NTK_by_parts_high_freq_factor": 4.0,
            "NTK_by_parts_factor": 32.0,
        }
    elif "Llama-3.2-3B" in official_model_name:
        cfg_dict = {
            "d_model": 3072,
            "d_head": 128,
            "n_heads": 24,
            "d_mlp": 8192,
            "n_layers": 28,
            "n_ctx": 2048,  # capped due to memory issues
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
            "use_NTK_by_parts_rope": True,
            "NTK_by_parts_low_freq_factor": 1.0,
            "NTK_by_parts_high_freq_factor": 4.0,
            "NTK_by_parts_factor": 32.0,
        }
    elif "Llama-3.1-8B" in official_model_name:
        cfg_dict = {
            "d_model": 4096,
            "d_head": 128,
            "n_heads": 32,
            "d_mlp": 14336,
            "n_layers": 32,
            "n_ctx": 2048,  # capped due to memory issues
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
            "use_NTK_by_parts_rope": True,
            "NTK_by_parts_low_freq_factor": 1.0,
            "NTK_by_parts_high_freq_factor": 4.0,
            "NTK_by_parts_factor": 8.0,
        }
    elif "Llama-3.1-70B" in official_model_name:
        cfg_dict = {
            "d_model": 8192,
            "d_head": 128,
            "n_heads": 64,
            "d_mlp": 28672,
            "n_layers": 80,
            "n_ctx": 2048,  # capped due to memory issues
            "eps": 1e-5,
            "d_vocab": 128256,
            "act_fn": "silu",
            "n_key_value_heads": 8,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": 128,
            "final_rms": True,
            "gated_mlp": True,
            "rotary_base": 500000.0,
            "use_NTK_by_parts_rope": True,
            "NTK_by_parts_low_freq_factor": 1.0,
            "NTK_by_parts_high_freq_factor": 4.0,
            "NTK_by_parts_factor": 8.0,
        }
    elif architecture == "GPT2LMHeadModel":
        cfg_dict = {
            "d_model": hf_config.n_embd,
            "d_head": hf_config.n_embd // hf_config.n_head,
            "n_heads": hf_config.n_head,
            "d_mlp": hf_config.n_embd * 4,
            "n_layers": hf_config.n_layer,
            "n_ctx": hf_config.n_ctx,
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "scale_attn_by_inverse_layer_idx": hf_config.scale_attn_by_inverse_layer_idx,
            "normalization_type": "LN",
        }
    elif architecture == "OPTForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.ffn_dim,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": 1e-5,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.activation_function,
            "use_attn_scale": True,
            "scale_attn_by_inverse_layer_idx": False,
            "normalization_type": "LN",
        }
    elif architecture == "GPTNeoXForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.layer_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "use_attn_scale": True,
            "scale_attn_by_inverse_layer_idx": False,
            "parallel_attn_mlp": True,
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "normalization_type": "LN",
        }
        rotary_pct = hf_config.rotary_pct
        cfg_dict["rotary_dim"] = round(rotary_pct * cfg_dict["d_head"])
    elif architecture == "MistralForCausalLM":
        use_local_attn = True if hf_config.sliding_window else False
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.head_dim
            if hasattr(hf_config, "head_dim") and hf_config.head_dim > 0
            else hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": 2048,  # Capped due to memory issues
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "window_size": hf_config.sliding_window,  # None if no sliding window was used
            "attn_types": ["local"] * hf_config.num_hidden_layers if use_local_attn else None,
            "eps": hf_config.rms_norm_eps,
            "rotary_base": hf_config.rope_theta,
            "n_key_value_heads": hf_config.num_key_value_heads,
            "use_local_attn": use_local_attn,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "gated_mlp": True,
        }
    elif architecture == "LlamaForCausalLM":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": hf_config.max_position_embeddings,
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "n_key_value_heads": (
                hf_config.num_key_value_heads
                if hf_config.num_key_value_heads != hf_config.num_attention_heads
                else None
            ),
            # This is done because the current implementation of GQA will use Grouped-Query Attention if
            # n_key_value_heads is not None, but hf_config.num_key_value_heads is sometimes specified as
            # the same as hf_config.num_attention_heads, in which case GQA should not be used.
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif architecture == "QWenLMHeadModel":
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "d_mlp": hf_config.intermediate_size // 2,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": 2048,  # Capped bc the actual ctx length is 30k and the attn mask would be too big
            "eps": hf_config.layer_norm_epsilon,
            "d_vocab": hf_config.vocab_size,
            "act_fn": "silu",
            "use_attn_scale": hf_config.scale_attn_weights,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_dim": hf_config.kv_channels,
            "rotary_adjacent_pairs": False,
            "final_rms": True,
            "gated_mlp": True,
        }
    elif architecture == "Qwen2ForCausalLM":
        # Note that Qwen1.5 models have architecture type Qwen2ForCausalLM.
        cfg_dict = {
            "d_model": hf_config.hidden_size,
            "d_head": hf_config.hidden_size // hf_config.num_attention_heads,
            "n_heads": hf_config.num_attention_heads,
            "n_key_value_heads": hf_config.num_key_value_heads,
            "d_mlp": hf_config.intermediate_size,
            "n_layers": hf_config.num_hidden_layers,
            "n_ctx": 2048,  # Capped bc the actual ctx length is 30k and the attn mask would be too big
            "eps": hf_config.rms_norm_eps,
            "d_vocab": hf_config.vocab_size,
            "act_fn": hf_config.hidden_act,
            "use_attn_scale": True,
            "initializer_range": hf_config.initializer_range,
            "normalization_type": "RMS",
            "positional_embedding_type": "rotary",
            "rotary_base": hf_config.rope_theta,
            "rotary_adjacent_pairs": False,
            "rotary_dim": hf_config.hidden_size // hf_config.num_attention_heads,
            "final_rms": True,
            "gated_mlp": True,
        }
    else:
        raise NotImplementedError(f"{architecture} is not currently supported.")
    # All of these models use LayerNorm
    cfg_dict["original_architecture"] = architecture
    if kwargs.get("trust_remote_code", False):
        cfg_dict["trust_remote_code"] = True
    return cfg_dict


def get_pretrained_model_config(
    model_name: str,
    default_prepend_bos: bool = True,
    **kwargs,
):
    """Returns the pretrained model config as an HookedTransformerConfig object.

    There are two types of pretrained models: HuggingFace models (where
    AutoModel and AutoConfig work), and models trained by me (NeelNanda) which
    aren't as integrated with HuggingFace infrastructure.

    Args:
        model_name: The name of the model from HuggingFace. Currently does not
            support models trained by [NeelNanda, ArthurConmy, Baidicoot].
        default_prepend_bos (bool, optional): Default behavior of whether to prepend the BOS token when the
            methods of HookedTransformer process input text to tokenize (only when input is a string).
            Defaults to True - even for models not explicitly trained with this, heads often use the
            first position as a resting position and accordingly lose information from the first token,
            so this empirically seems to give better results. To change the default behavior to False, pass in
            default_prepend_bos=False. Note that you can also locally override the default behavior by passing
            in prepend_bos=True/False when you call a method that processes the input string.

    """
    if Path(model_name).exists():
        # If the model_name is a path, it's a local model
        cfg_dict = convert_hf_model_config(model_name, **kwargs)
        official_model_name = model_name
    else:
        official_model_name = get_official_model_name(model_name)
    if official_model_name.startswith(NEED_REMOTE_CODE_MODELS) and not kwargs.get("trust_remote_code", False):
        logging.warning(f"Loading model {official_model_name} requires setting trust_remote_code=True")
        kwargs["trust_remote_code"] = True
    cfg_dict = convert_hf_model_config(official_model_name, **kwargs)
    cfg_dict["model_name"] = official_model_name.split("/")[-1]
    cfg_dict["default_prepend_bos"] = default_prepend_bos
    cfg = HookedTransformerConfig.from_dict(cfg_dict)
    return cfg


def get_hf_pretrained_weight(
    pretrained_model_name_or_path: str,
) -> dict[str, jax.Array]:
    is_local = os.path.isdir(pretrained_model_name_or_path)
    if is_local:
        if os.path.isfile(os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_NAME)):
            resolved_archive_file = os.path.join(pretrained_model_name_or_path, SAFE_WEIGHTS_NAME)
        else:
            raise EnvironmentError(
                f"Error: Unable to find file {SAFE_WEIGHTS_NAME} in directory {pretrained_model_name_or_path}."
            )
    else:
        resolved_archive_file = cached_file(pretrained_model_name_or_path, SAFE_WEIGHTS_NAME, token=True)
        if resolved_archive_file is None:
            raise EnvironmentError(
                f"Error: Unable to find file {SAFE_WEIGHTS_NAME} in directory {pretrained_model_name_or_path}."
            )

    state_dict = safe_load_file(resolved_archive_file)
    return state_dict


def get_pretrained_state_dict(
    official_model_name: str,
    cfg: HookedTransformerConfig,
    hf_model=None,
    **kwargs,
) -> dict[str, jax.Array]:
    """
    Loads in the model weights for a pretrained model, and processes them to
    have the HookedTransformer parameter names and shapes. Supports checkpointed
    models (and expects the checkpoint info to be stored in the config object)

    hf_model: Optionally, a HuggingFace model object. If provided, we will use
        these weights rather than reloading the model.
    kwargs: Other optional arguments passed to HuggingFace's from_pretrained.
        Also given to other HuggingFace functions when compatible.
    """

    if Path(official_model_name).exists():
        official_model_name = str(Path(official_model_name).resolve())
        logging.info(f"Loading model from local path {official_model_name}")
    else:
        official_model_name = get_official_model_name(official_model_name)
    if official_model_name.startswith(NEED_REMOTE_CODE_MODELS) and not kwargs.get("trust_remote_code", False):
        logging.warning(f"Loading model {official_model_name} state dict requires setting trust_remote_code=True")
        kwargs["trust_remote_code"] = True

    if hf_model is not None:
        params: dict[str, jax.Array] = {k: jnp.array(v) for k, v in flatten_dict(hf_model.state_dict()).values()}
    else:
        try:
            params = get_hf_pretrained_weight(official_model_name)
        except EnvironmentError:
            logging.warning(
                "Cannot load weights from non-sharded .safetensors file. Attempting to load using transformers."
            )
            huggingface_token = os.environ.get("HF_TOKEN", None)
            if official_model_name in NON_HF_HOSTED_MODEL_NAMES:
                raise NotImplementedError("Model not hosted on HuggingFace, must pass in hf_model")
            from transformers import AutoModelForCausalLM

            hf_model = AutoModelForCausalLM.from_pretrained(official_model_name, token=huggingface_token, **kwargs)
            params = {k: jnp.array(v) for k, v in flatten_dict(hf_model.state_dict()).values()}

    if cfg.original_architecture == "GPT2LMHeadModel":
        state_dict = convert_gpt2_weights(params, cfg)
    elif cfg.original_architecture == "LlamaForCausalLM":
        state_dict = convert_llama_weights(params, cfg)
    elif cfg.original_architecture == "Qwen2ForCausalLM":
        state_dict = convert_qwen2_weights(params, cfg)
    elif cfg.original_architecture == "GPTNeoXForCausalLM":
        state_dict = convert_neox_weights(params, cfg)
    elif cfg.original_architecture == "MistralForCausalLM":
        state_dict = convert_mistral_weights(params, cfg)
    else:
        raise ValueError(
            f"Loading weights from the architecture is not currently supported: {cfg.original_architecture}, generated from model name {cfg.model_name}. Feel free to open an issue on GitHub to request this feature."
        )

    return state_dict
