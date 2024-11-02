from typing import TypeVar

import jax

T = TypeVar("T")
U = TypeVar("U")


def transformer_lens_compatible_path_str(key_path: jax.tree_util.KeyPath) -> str:
    def _transform_key_entry(entry: jax.tree_util.KeyEntry) -> str:
        if isinstance(entry, jax.tree_util.SequenceKey):
            return str(entry.idx)
        if isinstance(entry, jax.tree_util.GetAttrKey):
            return entry.name
        if isinstance(entry, jax.tree_util.DictKey):
            return str(entry.key)
        if isinstance(entry, jax.tree_util.FlattenedIndexKey):
            return str(entry.key)
        raise ValueError(f"Unsupported key entry type: {type(entry)}")

    return ".".join(map(_transform_key_entry, key_path))


def get_nested_component(
    tree: U,
    path: str,
    component_type: type[T] | None = None,
    transformer_lens_compatible: bool = True,
) -> T:
    """Get a nested component from a tree.

    Args:
        tree: A PyTree.
        path: A string representing the path to the component.
        component_type: The type of the component to retrieve. If None, the component is a leaf node.
        transformer_lens_compatible: Whether the path is compatible with TransformerLens format.
    """

    res, _ = jax.tree_util.tree_flatten_with_path(
        tree,
        is_leaf=None if component_type is None else lambda x: isinstance(x, component_type),
    )

    def filter_path(key_path: jax.tree_util.KeyPath):
        return path in [jax.tree_util.keystr(key_path)] + (
            [transformer_lens_compatible_path_str(key_path)] if transformer_lens_compatible else []
        )

    res = [x for key_path, x in res if filter_path(key_path)]
    assert len(res) == 1, f"Expected 1 component, got {len(res)} components."
    return res[0]


def set_nested_component(
    tree: U,
    path: str,
    component: T,
    component_type: type[T] | None = None,
    transformer_lens_compatible: bool = True,
) -> U:
    """Set a nested component in a tree out-of-place.

    Args:
        tree: A PyTree.
        path: A string representing the path to the component.
        component: The component to set.
        component_type: The type of the component to retrieve. If None, the component is a leaf node.
        transformer_lens_compatible: Whether the path is compatible with TransformerLens format.
    """

    res, tree_def = jax.tree_util.tree_flatten_with_path(
        tree,
        is_leaf=None if component_type is None else lambda x: isinstance(x, component_type),
    )

    def filter_path(key_path: jax.tree_util.KeyPath):
        return path in [jax.tree_util.keystr(key_path)] + (
            [transformer_lens_compatible_path_str(key_path)] if transformer_lens_compatible else []
        )

    res = [component if filter_path(key_path) else x for key_path, x in res]

    return jax.tree_util.tree_unflatten(tree_def, res)


def load_pretrained_weights(
    model: U,
    pretrained_weights: dict[str, T],
    transformer_lens_compatible: bool = True,
) -> U:
    """Load pretrained weights into a model.

    Args:
        model: A PyTree.
        pretrained_weights: A dictionary of pretrained weights.
        transformer_lens_compatible: Whether the path is compatible with TransformerLens format.
    """

    res, tree_def = jax.tree_util.tree_flatten_with_path(model)

    res = [
        pretrained_weights.get(transformer_lens_compatible_path_str(key_path), x)
        if transformer_lens_compatible_path_str(key_path) in pretrained_weights
        else x
        for key_path, x in res
    ]

    return jax.tree_util.tree_unflatten(tree_def, res)
