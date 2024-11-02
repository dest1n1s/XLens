import equinox as eqx

from xlens import HookPoint, get_nested_component, set_nested_component


class ModuleA(eqx.Module):
    hook_point: HookPoint

    def __call__(self, x):
        return self.hook_point(x)


class ModuleB(eqx.Module):
    module_as: list[ModuleA]

    def __call__(self, x):
        for module_a in self.module_as:
            x = module_a(x)
        return x


def test_get_nested_component():
    module = ModuleB(
        module_as=[
            ModuleA(hook_point=HookPoint().append_hook(lambda x: x + 1)),
            ModuleA(hook_point=HookPoint().append_hook(lambda x: x * 2)),
        ]
    )

    # Get nested component with TransformerLens compatible path
    assert module.module_as[0].hook_point is get_nested_component(
        module, "module_as.0.hook_point", HookPoint
    ), f"{module.module_as[0].hook_point} != {get_nested_component(module, 'module_as.0.hook_point', HookPoint)}"

    # Get nested component with Jax PyTree path
    assert module.module_as[0].hook_point is get_nested_component(
        module, ".module_as[0].hook_point", HookPoint
    ), f"{module.module_as[0].hook_point} != {get_nested_component(module, '.module_as[0].hook_point', HookPoint)}"


def test_set_nested_component():
    module = ModuleB(
        module_as=[
            ModuleA(hook_point=HookPoint().append_hook(lambda x: x + 1)),
            ModuleA(hook_point=HookPoint().append_hook(lambda x: x * 2)),
        ]
    )

    new_hook_point = HookPoint().append_hook(lambda x: x * 3)

    # Set nested component with TransformerLens compatible path
    module_modified = set_nested_component(module, "module_as.0.hook_point", new_hook_point, HookPoint)

    assert (
        module_modified.module_as[0].hook_point is new_hook_point
    ), f"{module_modified.module_as[0].hook_point} != {new_hook_point}"

    assert (
        module.module_as[1].hook_point is module_modified.module_as[1].hook_point
    ), f"{module.module_as[1].hook_point} != {module_modified.module_as[1].hook_point}"

    assert (
        module.module_as[0].hook_point is not module_modified.module_as[0].hook_point
    ), f"{module.module_as[0].hook_point} == {module_modified.module_as[0].hook_point}"

    # Set nested component with Jax PyTree path
    module_modified = set_nested_component(module, ".module_as[0].hook_point", new_hook_point, HookPoint)

    assert (
        module_modified.module_as[0].hook_point is new_hook_point
    ), f"{module_modified.module_as[0].hook_point} != {new_hook_point}"
