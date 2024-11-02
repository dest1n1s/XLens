"""Activation Functions.

Utilities for interacting with all supported activation functions.
"""

from typing import Callable, Dict

import jax
import jax.numpy as jnp
from jaxtyping import Float

# Convenient type for the format of each activation function
ActivationFunction = Callable[..., jax.Array]


def gelu_new(input: Float[jax.Array, "batch pos d_mlp"]) -> Float[jax.Array, "batch pos d_mlp"]:
    # Implementation of GeLU used by GPT2 - subtly different from PyTorch's
    return 0.5 * input * (1.0 + jnp.tanh(jnp.sqrt(2.0 / jnp.pi) * (input + 0.044715 * jnp.pow(input, 3.0))))


def gelu_fast(input: Float[jax.Array, "batch pos d_mlp"]) -> Float[jax.Array, "batch pos d_mlp"]:
    return 0.5 * input * (1.0 + jnp.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))


def solu(input: Float[jax.Array, "batch pos d_mlp"]) -> Float[jax.Array, "batch pos d_mlp"]:
    """
    SoLU activation function as described by
    https://transformer-circuits.pub/2022/solu/index.html.

    LayerNorm implemented by the MLP class.
    """
    return input * jax.nn.softmax(input, axis=-1)


# All currently supported activation functions. To add a new function, simply
# put the name of the function as the key, and the value as the actual callable.
SUPPORTED_ACTIVATIONS: Dict[str, ActivationFunction] = {
    "solu": solu,
    "solu_ln": solu,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "silu": jax.nn.silu,
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
}
