from typing import Optional

import jax
from flax import nnx
from jaxtyping import Float


class KVCache(
    nnx.Variable[Optional[tuple[Float[jax.Array, "batch kv_pos d_model"], Float[jax.Array, "batch kv_pos d_model"]]]]
):
    def __init__(
        self,
        value: Optional[
            tuple[Float[jax.Array, "batch kv_pos d_model"], Float[jax.Array, "batch kv_pos d_model"]]
        ] = None,
    ):
        super().__init__(value)

    @property
    def k(self) -> Optional[Float[jax.Array, "batch kv_pos d_model"]]:
        return self.value[0] if self.value is not None else None

    @property
    def v(self) -> Optional[Float[jax.Array, "batch kv_pos d_model"]]:
        return self.value[1] if self.value is not None else None
