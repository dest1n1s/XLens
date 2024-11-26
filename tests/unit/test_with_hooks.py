from typing import Self

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from xlens import HookPoint, with_cache, with_hooks
from xlens.hooks.utilities import retrieve_cache
from xlens.utilities.functional import functional


class ModuleA(nnx.Module):
    hook_mid: HookPoint

    def __init__(self, hook_mid: HookPoint):
        self.hook_mid = hook_mid

    @functional
    def __call__(self, x: jax.Array) -> tuple[jax.Array, Self]:
        x, self.hook_mid = self.hook_mid(x * 2)
        return x * 2, self


def test_with_hooks():
    a = ModuleA(HookPoint())
    a_with_hooks = with_hooks(a, [("hook_mid", lambda x, state: (x + 1, state))])
    y_with_hooks = a_with_hooks(jnp.array(1.0))
    assert jnp.allclose(y_with_hooks, 6.0)
    y = a(jnp.array(1.0))
    assert jnp.allclose(y, 4.0)


def test_with_cache():
    a = ModuleA(HookPoint())
    a = with_cache(a, ["hook_mid"])
    y, a = a(jnp.array(1.0))
    cache = retrieve_cache(a, ["hook_mid"])

    assert jnp.allclose(y, 4.0)
    assert "hook_mid" in cache
    assert jnp.allclose(cache["hook_mid"], 2.0)


if __name__ == "__main__":
    test_with_cache()
