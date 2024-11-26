from typing import Any, Callable, Generic, Self

import flax.nnx as nnx
import jax
from typing_extensions import TypeVar

from xlens.utilities.functional import functional

T = TypeVar("T", default=jax.Array)


class HookPoint(nnx.Module, Generic[T]):
    hooks: list[Callable[[T, Any], tuple[T, Any]]]
    state: nnx.Variable[Any]

    def __init__(self, hooks: list[Callable[[T, Any], tuple[T, Any]]] = [], state: Any = None):
        self.hooks = hooks
        self.state = nnx.Variable(state)

    @functional
    def __call__(self, x: T) -> tuple[T, Self]:
        for hook in self.hooks:
            x, self.state.value = hook(x, self.state.value)
        return x, self

    def append_hook(self, hook: Callable[[T, Any], tuple[T, Any]]) -> "HookPoint[T]":
        return HookPoint(self.hooks + [hook], self.state)

    def prepend_hook(self, hook: Callable[[T, Any], tuple[T, Any]]) -> "HookPoint[T]":
        return HookPoint([hook] + self.hooks, self.state)

    def clear_hooks(self) -> "HookPoint[T]":
        return HookPoint([])
