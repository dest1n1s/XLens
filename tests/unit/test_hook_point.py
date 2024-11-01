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


if __name__ == "__main__":
    import equinox as eqx
    import jax

    class ModuleA(eqx.Module):
        hook_point: HookPoint

        def __forward__(self, x):
            return self.hook_point(x)

    class ModuleB(eqx.Module):
        module_as: list[ModuleA]

        def __forward__(self, x):
            for module_a in self.module_as:
                x = module_a(x)
            return x

    module = ModuleB(
        module_as=[
            ModuleA(hook_point=HookPoint().append_hook(lambda x: x + 1)),
            ModuleA(hook_point=HookPoint().append_hook(lambda x: x * 2)),
        ]
    )

    tree_def = jax.tree.structure(module)
    print(tree_def)

    def f(path, x, *args):
        print(jax.tree_util.keystr(path), x, args)
        return x

    res, tree_def = jax.tree_util.tree_flatten_with_path(module, is_leaf=lambda x: isinstance(x, HookPoint))
    print(tree_def)
    for path, x in res:
        print(jax.tree_util.keystr(path), x)
