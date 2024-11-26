from typing import Any, Callable

import flax.nnx as nnx
from typing_extensions import TypeVar

from xlens.utilities.functional import functional
from xlens.utilities.traverse import get_nested_attr

from .hook_point import HookPoint

U = TypeVar("U", bound=nnx.Module)


@functional
def with_hooks(tree: U, hooks: list[tuple[str, Callable[[Any, Any], tuple[Any, Any]]]] = []) -> U:
    """Set hooks on a tree of objects.

    Args:
        tree: U: The tree of objects to set hooks on.
        hooks: list[tuple[str, Callable[[Any], Any]]]: A list of tuples, where the first element
            is the name of the hook, and the second element is a callable that takes in a value
            and returns a value. The callable should be a pure function, as it will be called
            multiple times. Defaults to [].
    """

    for hook_name, hook_fn in hooks:
        hook_point = get_nested_attr(tree, hook_name)
        assert isinstance(hook_point, HookPoint), f"Attribute {hook_name} is not a HookPoint"
        hook_point.hooks = hook_point.hooks + [hook_fn]

    return tree


def with_cache(tree: U, hook_names: list[str] = []) -> U:
    """Set hooks on a tree of objects.

    This function uses the state of the hook point to cache the values.

    Args:
        tree: U: The tree of objects to set hooks on.
        hook_names: list[str]: A list of strings, where each string is the name of a hook point
    """

    def hook_fn(name: str):
        def _hook_fn(x: Any, state: Any):
            assert state is None, "State to cache should be None"
            return x, x

        return _hook_fn

    tree = with_hooks(tree, [(name, hook_fn(name)) for name in hook_names])

    return tree


def retrieve_cache(tree: Any, hook_names: list[str] = []) -> dict[str, Any]:
    return {name: get_nested_attr(tree, name).state.value for name in hook_names}
