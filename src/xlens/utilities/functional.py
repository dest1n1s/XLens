from functools import partial, wraps
from typing import Any, Callable, Optional, TypeVar, overload

from flax import nnx

F = TypeVar("F", bound=Callable[..., Any])


@overload
def functional(*, transform: Optional[Callable[[Any], Any]] = None) -> Callable[[F], F]: ...
@overload
def functional(f: F, *, transform: Optional[Callable[[Any], Any]] = None) -> F: ...


def functional(f: F | None = None, *, transform: Optional[Callable[[Any], Any]] = None) -> Callable[[F], F] | F:
    if f is None:
        return partial(functional, transform=transform)

    @wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        def inner(pure_args: Any, pure_kwargs: Any) -> Any:
            args, kwargs = nnx.from_tree((pure_args, pure_kwargs))
            out = f(*args, **kwargs)
            pure_out = nnx.to_tree(out)
            return pure_out

        inner = transform(inner) if transform is not None else inner

        pure_args, pure_kwargs = nnx.to_tree((args, kwargs))
        pure_out = inner(pure_args, pure_kwargs)
        return nnx.from_tree(pure_out)

    return wrapper
