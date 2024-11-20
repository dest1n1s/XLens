from typing import TypeVar

import flax.nnx as nnx
import jax
from flax.traverse_util import unflatten_dict

T = TypeVar("T")
U = TypeVar("U", bound=nnx.Module)


def load_pretrained_weights(
    model: U,
    pretrained_weights: dict[str, jax.Array],
) -> U:
    """Load pretrained weights into a model.

    Args:
        model: An nnx.Module.
        pretrained_weights: A dictionary of pretrained weights.
    """

    graph_def, state = nnx.split(model)

    state.replace_by_pure_dict(unflatten_dict(pretrained_weights, sep="."))

    return nnx.merge(graph_def, state)
