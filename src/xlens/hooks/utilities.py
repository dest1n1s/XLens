from typing import Any, Callable

from typing_extensions import TypeVar

from xlens.utils import get_nested_component, set_nested_component

from .hook_point import HookPoint

U = TypeVar("U")


def with_hooks(tree: U, hooks: list[tuple[str, Callable[[Any], Any]]] = []) -> U:
    """Set hooks on a tree of objects.

    Args:
        tree: U: The tree of objects to set hooks on.
        hooks: list[tuple[str, Callable[[Any], Any]]]: A list of tuples, where the first element
            is the name of the hook, and the second element is a callable that takes in a value
            and returns a value. The callable should be a pure function, as it will be called
            multiple times. Defaults to [].
    """

    for hook_name, hook_fn in hooks:
        hook_point = get_nested_component(tree, hook_name, HookPoint)
        hook_point = hook_point.append_hook(hook_fn)
        tree = set_nested_component(tree, hook_name, hook_point, HookPoint)

    return tree


def with_cache(tree: U, hook_names: list[str] = []) -> tuple[U, dict[str, Any]]:
    """Set hooks on a tree of objects.

    Warning: This is not a pure function. Each time the tree is called, the cache will be updated.
            Do JIT outside the full scope of the cache.

    Args:
        tree: U: The tree of objects to set hooks on.
        hook_names: list[str]: A list of strings, where each string is the name of a hook point
    """

    cache = {}

    def hook_fn(name: str):
        def _hook_fn(x: Any):
            cache[name] = x
            return x

        return _hook_fn

    tree = with_hooks(tree, [(name, hook_fn(name)) for name in hook_names])

    return tree, cache
