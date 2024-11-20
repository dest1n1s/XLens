from typing import Any, TypeVar

from flax import nnx

T = TypeVar("T", bound=nnx.Module)


def get_nested_attr(module: nnx.Module, path: str | list[str]) -> Any:
    """
    Retrieves a nested attribute from an module based on a path.

    For example, if `path` is "a.b.c" or ["a", "b", "c"], this function will return `module.a.b.c`.

    Args:
        module (nnx.Module): The module from which to retrieve the attribute.
        path (Union[str, List[str]]): A dot-separated string or list of strings representing the attribute hierarchy.

    Returns:
        Any: The value of the nested attribute.
    """

    def path_parts_to_path(path_parts: tuple[Any, ...]) -> str:
        return ".".join(str(part) for part in path_parts)

    objs = [obj for path_parts, obj in nnx.iter_graph(module) if path_parts_to_path(path_parts) == path]
    assert len(objs) == 1, f"Expected exactly one object, got {len(objs)}"
    return objs[0]
