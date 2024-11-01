from dataclasses import replace
from typing import Callable, Self, TypeVar

import equinox as eqx

T = TypeVar("T")


class HookPoint(eqx.Module):
    hooks: list[Callable[[T], T]] = eqx.field(static=True, default_factory=list)

    def __call__(self, x: T) -> T:
        for hook in self.hooks:
            x = hook(x)
        return x

    def append_hook(self, hook: Callable[[T], T]) -> Self:
        return replace(self, hooks=self.hooks + [hook])

    def prepend_hook(self, hook: Callable[[T], T]) -> Self:
        return replace(self, hooks=[hook] + self.hooks)

    def clear_hooks(self) -> Self:
        return replace(self, hooks=[])
