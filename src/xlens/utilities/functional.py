from functools import partial, wraps
from typing import Any, Callable, Optional, TypeVar, overload

from flax import nnx

F = TypeVar("F", bound=Callable[..., Any])


def _split_fn(ctx: Any, path: Any, prefix: Any, leaf: Any) -> Any:
    return nnx.NodeStates.from_split(*nnx.split(leaf))


def _merge_fn(ctx: Any, path: Any, prefix: Any, states: nnx.NodeStates) -> Any:
    assert isinstance(states, nnx.NodeStates), "Expected NodeStates"
    return nnx.merge(states.graphdef, *states.states)


@overload
def functional(*, transform: Optional[Callable[[Any], Any]] = None) -> Callable[[F], F]: ...
@overload
def functional(f: F, *, transform: Optional[Callable[[Any], Any]] = None) -> F: ...


def functional(f: F | None = None, *, transform: Optional[Callable[[Any], Any]] = None) -> Callable[[F], F] | F:
    if f is None:
        return partial(functional, transform=transform)

    def inner(*pure_args: Any, **pure_kwargs: Any) -> Any:
        args, kwargs = nnx.from_tree((pure_args, pure_kwargs), merge_fn=_merge_fn)
        out = f(*args, **kwargs)
        pure_out = nnx.to_tree(out, split_fn=_split_fn)
        return pure_out

    inner = transform(inner) if transform is not None else inner

    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        pure_args, pure_kwargs = nnx.to_tree((args, kwargs), split_fn=_split_fn)
        pure_out = inner(*pure_args, **pure_kwargs)
        return nnx.from_tree(pure_out, merge_fn=_merge_fn)

    wrapper.inner = inner  # type: ignore

    return wrapper
