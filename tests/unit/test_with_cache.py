import equinox as eqx
import jax.numpy as jnp

from xlens import HookPoint, with_cache


class ModuleA(eqx.Module):
    hook_mid: HookPoint

    def __call__(self, x):
        return self.hook_mid(x * 2) * 2


def test_with_cache():
    a = ModuleA(HookPoint())
    a, cache = with_cache(a, ["hook_mid"])
    y = a(jnp.array(1.0))

    assert jnp.allclose(y, 4.0)
    assert "hook_mid" in cache
    assert jnp.allclose(cache["hook_mid"], 2.0)
