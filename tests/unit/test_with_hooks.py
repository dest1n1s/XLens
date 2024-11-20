import flax.nnx as nnx
import jax
import jax.numpy as jnp

from xlens import HookPoint, with_cache, with_hooks


class ModuleA(nnx.Module):
    hook_mid: HookPoint

    def __init__(self, hook_mid: HookPoint):
        self.hook_mid = hook_mid

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.hook_mid(x * 2) * 2


def test_with_hooks():
    a = ModuleA(HookPoint())
    a_with_hooks = with_hooks(a, [("hook_mid", lambda x: x + 1)])
    y_with_hooks = a_with_hooks(jnp.array(1.0))
    assert jnp.allclose(y_with_hooks, 6.0)
    y = a(jnp.array(1.0))
    assert jnp.allclose(y, 4.0)


def test_with_cache():
    a = ModuleA(HookPoint())
    a, cache = with_cache(a, ["hook_mid"])
    y = a(jnp.array(1.0))

    assert jnp.allclose(y, 4.0)
    assert "hook_mid" in cache
    assert jnp.allclose(cache["hook_mid"], 2.0)
