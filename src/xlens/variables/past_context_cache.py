from typing import Generic, Optional, TypeVar, Union

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Float, Int, PyTree

T = TypeVar(
    "T",
    bound=PyTree[
        Optional[
            Union[
                Float[jax.Array, "batch kv_pos ..."],
                Int[jax.Array, "batch kv_pos ..."],
            ]
        ]
    ],
)


class PastContextCache(
    Generic[T],
    nnx.Variable[
        tuple[
            int,
            Optional[T],
        ]
    ],
):
    max_length: int

    def __init__(
        self,
        max_length: int,
    ):
        super().__init__(value=(0, None))
        self.max_length = max_length

    def append(
        self,
        value: T,
        length: int,
    ) -> T:
        def _init_cache(x: Optional[jax.Array]) -> Optional[jax.Array]:
            if x is None:
                return None

            return jnp.zeros_like(x, shape=x.shape[:1] + (self.max_length,) + x.shape[2:])

        def _update_cache(x: Optional[jax.Array], v: Optional[jax.Array]) -> Optional[jax.Array]:
            if x is None or v is None:
                assert x is None and v is None, "x and v must both be None or both be arrays"
                return None

            return jax.lax.dynamic_update_slice(x, v, (0, self.value[0], *[0] * (len(x.shape) - 2)))

        if self.value[1] is None:
            self.value = (
                0,
                jax.tree.map(_init_cache, value),
            )

        cached = jax.tree.map(
            _update_cache,
            self.value[1],
            value,
        )

        self.value = (length + self.value[0], cached)
        return cached

    @property
    def length(self) -> int:
        return self.value[0]
