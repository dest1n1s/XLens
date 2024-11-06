import functools
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Optional, cast

import jax
import jax.numpy as jnp
from safetensors.flax import load_file as safe_load_file
from transformers import AutoConfig
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    cached_file,
    has_file,
)

from xlens.config import HookedTransformerConfig
from xlens.utils import flatten_dict


class ModelConverter(ABC):
    @abstractmethod
    def can_convert(self, model_name_or_path: str) -> bool:
        """Check if the model converter can convert the given model."""
        pass

    @abstractmethod
    def get_pretrained_model_config(self, model_name_or_path: str, **kwargs: Any) -> HookedTransformerConfig:
        """Get the model configuration for the given model name.

        Args:
            model_name_or_path: Name or path of the model
            **kwargs: Additional arguments passed to configuration loading

        Returns:
            HookedTransformerConfig for the model
        """
        pass

    @abstractmethod
    def get_pretrained_weights(
        self, cfg: HookedTransformerConfig, model_name_or_path: str, **kwargs: Any
    ) -> dict[str, jax.Array]:
        """Get the pretrained weights for the given model.

        Args:
            cfg: Model configuration
            model_name_or_path: Name or path of the model
            **kwargs: Additional arguments passed to weight loading

        Returns:
            Dictionary mapping parameter names to weight arrays
        """
        pass


class HuggingFaceModelConverterSingle(ModelConverter):
    def __init__(
        self,
        model_names: list[str],
        model_alias_map: Optional[dict[str, list[str]]] = None,
        model_architecture: Optional[str] = None,
    ):
        self.model_names = model_names
        self.model_alias_map = model_alias_map or {}
        # Reverse the alias map, to allow for O(1) lookup from alias to model name. The reverse map should contain
        # the original model name as an alias.
        self.rev_alias_map = {
            alias: model_name for model_name, aliases in self.model_alias_map.items() for alias in aliases
        } | {model_name: model_name for model_name in self.model_names}

        self.model_architecture = model_architecture

    def can_convert(self, model_name_or_path: str) -> bool:
        if os.path.isdir(model_name_or_path):
            if os.path.exists(os.path.join(model_name_or_path, "config.json")):
                hf_cfg = AutoConfig.from_pretrained(model_name_or_path, token=True)
                architecture: str = hf_cfg.architectures[0]
                return architecture == self.model_architecture
            else:
                return False
        else:
            return model_name_or_path in self.rev_alias_map

    @abstractmethod
    def convert_hf_model_config(self, hf_cfg: AutoConfig) -> HookedTransformerConfig:
        """Convert a HuggingFace model config to a HookedTransformerConfig.

        Args:
            hf_cfg: HuggingFace model configuration object

        Returns:
            Equivalent configuration in HookedTransformerConfig format
        """
        pass

    def get_pretrained_model_config(self, model_name_or_path: str, **kwargs: Any) -> HookedTransformerConfig:
        model_name_or_path = (
            model_name_or_path if os.path.isdir(model_name_or_path) else self.rev_alias_map[model_name_or_path]
        )
        hf_cfg = AutoConfig.from_pretrained(model_name_or_path, token=True)
        return self.convert_hf_model_config(hf_cfg)

    @abstractmethod
    def convert_hf_weights(
        self, hf_weights: dict[str, jax.Array], cfg: HookedTransformerConfig
    ) -> dict[str, jax.Array]:
        """Convert a HuggingFace model weights to a HookedTransformer weights.

        Args:
            hf_weights: HuggingFace model weights
            cfg: Model configuration
        """
        pass

    def _get_safe_weights_files(self, safe_weight_index: Any) -> list[str]:
        weight_map = safe_weight_index["weight_map"]
        return [weight_map[key] for key in weight_map]

    def get_pretrained_weights(
        self, cfg: HookedTransformerConfig, model_name_or_path: str, **kwargs: Any
    ) -> dict[str, jax.Array]:
        if os.path.isdir(model_name_or_path):
            if os.path.isfile(os.path.join(model_name_or_path, SAFE_WEIGHTS_NAME)):
                resolved_archive_files = [os.path.join(model_name_or_path, SAFE_WEIGHTS_NAME)]
            elif os.path.isfile(os.path.join(model_name_or_path, SAFE_WEIGHTS_INDEX_NAME)):
                with open(os.path.join(model_name_or_path, SAFE_WEIGHTS_INDEX_NAME), "r") as f:
                    safe_weight_index = json.load(f)
                resolved_archive_files = [
                    os.path.join(model_name_or_path, file_name)
                    for file_name in self._get_safe_weights_files(safe_weight_index)
                ]
            else:
                raise EnvironmentError(
                    f"Error: Unable to find file {SAFE_WEIGHTS_NAME} in directory {model_name_or_path}."
                )
        else:
            if has_file(model_name_or_path, SAFE_WEIGHTS_NAME):
                resolved_archive_files = [cached_file(model_name_or_path, SAFE_WEIGHTS_NAME, token=True)]
                assert all(resolved_archive_files)
                resolved_archive_files = cast(list[str], resolved_archive_files)
            elif has_file(model_name_or_path, SAFE_WEIGHTS_INDEX_NAME):
                safe_weights_index_file = cached_file(model_name_or_path, SAFE_WEIGHTS_INDEX_NAME, token=True)
                assert safe_weights_index_file is not None
                with open(safe_weights_index_file, "r") as f:
                    safe_weight_index = json.load(f)
                resolved_archive_files = [
                    cached_file(model_name_or_path, file_name, token=True)
                    for file_name in self._get_safe_weights_files(safe_weight_index)
                ]
                assert all(resolved_archive_files)
                resolved_archive_files = cast(list[str], resolved_archive_files)
            else:
                resolved_archive_files = None
        if resolved_archive_files is None:
            logging.warning(
                "Cannot load weights from non-sharded .safetensors file. Attempting to load using transformers."
            )
            from transformers import AutoModelForCausalLM

            hf_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, token=True, **kwargs)
            params: dict[str, jax.Array] = {k: jnp.array(v) for k, v in flatten_dict(hf_model.state_dict()).items()}
        else:
            params = {}
            for resolved_archive_file in resolved_archive_files:
                params.update(safe_load_file(resolved_archive_file))
        return self.convert_hf_weights(params, cfg)


class HuggingFaceModelConverter(ModelConverter):
    """Convert a HuggingFace model to a HookedTransformer model. This class is a wrapper around a list of
    HuggingFaceModelConverterSingle objects, which can convert different types of models.
    """

    def __init__(self, converters: list[HuggingFaceModelConverterSingle]):
        self.converters = converters

    @property
    def model_names(self) -> list[str]:
        return [model_name for converter in self.converters for model_name in converter.model_names]

    @property
    def model_alias_map(self) -> dict[str, list[str]]:
        return functools.reduce(lambda a, b: {**a, **b}, [converter.model_alias_map for converter in self.converters])

    @property
    def rev_alias_map(self) -> dict[str, str]:
        return functools.reduce(lambda a, b: {**a, **b}, [converter.rev_alias_map for converter in self.converters])

    @property
    def name_converter_map(self) -> dict[str, HuggingFaceModelConverterSingle]:
        return {model_name: converter for converter in self.converters for model_name in converter.model_names}

    @property
    def architecture_converter_map(self) -> dict[str, HuggingFaceModelConverterSingle]:
        return {
            converter.model_architecture: converter
            for converter in self.converters
            if converter.model_architecture is not None
        }

    @property
    def model_architectures(self) -> list[str]:
        return [
            converter.model_architecture for converter in self.converters if converter.model_architecture is not None
        ]

    def can_convert(self, model_name_or_path: str) -> bool:
        if os.path.isdir(model_name_or_path):
            if os.path.exists(os.path.join(model_name_or_path, "config.json")):
                architecture: Any = AutoConfig.from_pretrained(model_name_or_path, token=True).architectures[0]
                return architecture in self.model_architectures
            else:
                return False
        else:
            return model_name_or_path in self.rev_alias_map

    def get_pretrained_model_config(self, model_name_or_path: str, **kwargs: Any) -> HookedTransformerConfig:
        if os.path.isdir(model_name_or_path):
            hf_cfg = AutoConfig.from_pretrained(model_name_or_path, token=True)
            architecture = hf_cfg.architectures[0]
            if architecture in self.architecture_converter_map:
                return self.architecture_converter_map[architecture].convert_hf_model_config(hf_cfg)
            else:
                raise ValueError(f"No converter found for model {model_name_or_path}")
        else:
            return self.name_converter_map[model_name_or_path].get_pretrained_model_config(model_name_or_path, **kwargs)

    def get_pretrained_weights(
        self, cfg: HookedTransformerConfig, model_name_or_path: str, **kwargs: Any
    ) -> dict[str, jax.Array]:
        if cfg.original_architecture in self.architecture_converter_map:
            return self.architecture_converter_map[cfg.original_architecture].get_pretrained_weights(
                cfg, model_name_or_path, **kwargs
            )
        else:
            raise ValueError(f"No converter found for model {model_name_or_path}")
