from typing import Callable, Generic

import flax.nnx as nnx
import jax
from typing_extensions import TypeVar

T = TypeVar("T", default=jax.Array)


class HookPoint(nnx.Module, Generic[T]):
    def __init__(self, hooks: list[Callable[[T], T]] = []):
        self.hooks = hooks

    def __call__(self, x: T) -> T:
        for hook in self.hooks:
            x = hook(x)
        return x

    def append_hook(self, hook: Callable[[T], T]) -> "HookPoint[T]":
        return HookPoint(self.hooks + [hook])

    def prepend_hook(self, hook: Callable[[T], T]) -> "HookPoint[T]":
        return HookPoint([hook] + self.hooks)

    def clear_hooks(self) -> "HookPoint[T]":
        return HookPoint([])
