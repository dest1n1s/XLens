import jax.numpy as jnp

from xlens import HookPoint


def test_hook_point():
    hook_point = HookPoint()
    x = jnp.array(1.0)

    # Test original value
    y = hook_point(x)
    assert jnp.allclose(x, y), f"{x} != {y}"

    # Test append_hook and prepend_hook
    hook_point = hook_point.append_hook(lambda x: x + 1)
    y = hook_point(x)
    assert jnp.allclose(x + 1, y), f"{x + 1} != {y}"

    hook_point = hook_point.append_hook(lambda x: x * 2)
    y = hook_point(x)
    assert jnp.allclose((x + 1) * 2, y), f"{(x + 1) * 2} != {y}"

    hook_point = hook_point.prepend_hook(lambda x: x + 1)
    y = hook_point(x)
    assert jnp.allclose((x + 2) * 2, y), f"{(x + 2) * 2} != {y}"

    # Test clear_hooks
    hook_point = hook_point.clear_hooks()
    y = hook_point(x)
    assert jnp.allclose(x, y), f"{x} != {y}"
