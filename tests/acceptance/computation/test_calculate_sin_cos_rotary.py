import math
from typing import Tuple

import einops
import jax.numpy as jnp
import pytest
from jaxtyping import Float

from xlens.components import Attention

pytest.importorskip("torch")

import torch  # noqa: E402


# Test the consistency with the torch implementation, copied from [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/components/abstract_attention.py#L465).
def torch_calculate_sin_cos_rotary(
    rotary_dim: int,
    n_ctx: int,
    base: int = 10000,
    dtype: torch.dtype = torch.float32,
    use_NTK_by_parts_rope: bool = False,
    NTK_by_parts_factor: float = 8.0,
    NTK_by_parts_low_freq_factor: float = 1.0,
    NTK_by_parts_high_freq_factor: float = 4.0,
    rotary_adjacent_pairs: bool = False,
) -> Tuple[Float[torch.Tensor, "n_ctx rotary_dim"], Float[torch.Tensor, "n_ctx rotary_dim"]]:
    """
    Calculate the sine and cosine waves to use in a rotary embedding. See https://blog.eleuther.ai/rotary-embeddings/ for details

    Note: For some inexplicable reason, in GPT-J each ADJACENT pair of elements in k and q are rotated, in GPT-NeoX the pair of elements at k and k+n//2 are rotated (ie folding the full length in half, and then looking at pairs accordingly). I have absolutely no clue why, it should be completely equivalent.
    To resolve this, I've coded it to default to the GPT-J mode, but to explicitly check whether it's GPT-NeoX and then do the GPT-NeoX thing if it is.
    """
    high_precision = torch.float32 if dtype != torch.float64 else torch.float64
    pos = torch.arange(n_ctx, dtype=high_precision)
    dim = torch.arange(rotary_dim // 2, dtype=high_precision)

    # Llama-3.1 uses NTK-by-Parts Rotary Embedding introduced in Section 3.2 in https://arxiv.org/pdf/2309.00071
    # Implementation copied from https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/modeling_rope_utils.py#L310
    if use_NTK_by_parts_rope:
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.int64).float() / rotary_dim))
        factor = NTK_by_parts_factor
        low_freq_factor = NTK_by_parts_low_freq_factor
        high_freq_factor = NTK_by_parts_high_freq_factor
        old_context_len = n_ctx

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        wavelen = 2 * math.pi / inv_freq
        inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
        smooth_factor = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
        is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
        inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)
        freq = 1 / inv_freq_llama
    else:
        freq = base ** (dim / (rotary_dim / 2))
    if rotary_adjacent_pairs:
        freq = einops.repeat(freq, "d -> (d 2)")
    else:
        freq = einops.repeat(freq, "d -> (2 d)")
    # Create a n_ctx x rotary_dim tensor, where each column is an arithmetic sequence of angles in that frequency
    angles = pos[:, None] / freq[None, :]
    return torch.sin(angles).to(dtype), torch.cos(angles).to(dtype)


@torch.no_grad()
def test_calculate_sin_cos_rotary():
    rotary_sin, rotary_cos = Attention.calculate_sin_cos_rotary(64, 128)
    assert rotary_sin.shape == (128, 64)
    assert rotary_cos.shape == (128, 64)

    rotary_sin_torch, rotary_cos_torch = torch_calculate_sin_cos_rotary(64, 128)
    assert jnp.allclose(rotary_sin, jnp.array(rotary_sin_torch), atol=1e-6)
    assert jnp.allclose(rotary_cos, jnp.array(rotary_cos_torch), atol=1e-6)
