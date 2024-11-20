import flax.nnx as nnx
import jax

from xlens import HookPoint
from xlens.utilities.traverse import get_nested_attr


class ModuleA(nnx.Module):
    hook_point: HookPoint

    def __init__(self, hook_point: HookPoint):
        self.hook_point = hook_point

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.hook_point(x)


class ModuleB(nnx.Module):
    module_as: list[ModuleA]

    def __init__(self, module_as: list[ModuleA]):
        self.module_as = module_as

    def __call__(self, x: jax.Array) -> jax.Array:
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

    # Get nested attribute
    print(get_nested_attr(module, "module_as.0.hook_point"))
    assert module.module_as[0].hook_point is get_nested_attr(
        module, "module_as.0.hook_point"
    ), f"{module.module_as[0].hook_point} != {get_nested_attr(module, 'module_as.0.hook_point')}"


# def test_set_nested_component():
#     module = ModuleB(
#         module_as=[
#             ModuleA(hook_point=HookPoint().append_hook(lambda x: x + 1)),
#             ModuleA(hook_point=HookPoint().append_hook(lambda x: x * 2)),
#         ]
#     )

#     new_hook_point = HookPoint().append_hook(lambda x: x * 3)

#     # Set nested component with TransformerLens compatible path
#     module_modified = replace_nested_attr(module, "module_as.0.hook_point", new_hook_point)

#     assert (
#         module_modified.module_as[0].hook_point is new_hook_point
#     ), f"{module_modified.module_as[0].hook_point} != {new_hook_point}"

#     assert (
#         module.module_as[1].hook_point is module_modified.module_as[1].hook_point
#     ), f"{module.module_as[1].hook_point} != {module_modified.module_as[1].hook_point}"

#     assert (
#         module.module_as[0].hook_point is not module_modified.module_as[0].hook_point
#     ), f"{module.module_as[0].hook_point} == {module_modified.module_as[0].hook_point}"
