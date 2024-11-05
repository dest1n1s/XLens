from typing import Any

import einops
import jax
import jax.numpy as jnp

from xlens.config import HookedTransformerConfig
from xlens.pretrained.model_converter import HuggingFaceModelConverterSingle


class GPT2Converter(HuggingFaceModelConverterSingle):
    def __init__(self):
        super().__init__(
            model_names=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "distilgpt2"],
            model_alias_map={
                "gpt2": ["gpt2-small"],
            },
            model_architecture="GPT2LMHeadModel",
        )

    def convert_hf_model_config(self, hf_cfg: Any) -> HookedTransformerConfig:
        return HookedTransformerConfig(
            d_model=hf_cfg.n_embd,
            n_layers=hf_cfg.n_layer,
            n_heads=hf_cfg.n_head,
            d_head=hf_cfg.n_embd // hf_cfg.n_head,
            d_mlp=hf_cfg.n_embd * 4,
            d_vocab=hf_cfg.vocab_size,
            n_ctx=hf_cfg.n_positions,
            act_fn=hf_cfg.activation_function,
            normalization_type="LN",
            scale_attn_by_inverse_layer_idx=hf_cfg.scale_attn_by_inverse_layer_idx,
            use_attn_scale=True,
            original_architecture="GPT2LMHeadModel",
        )

    def convert_hf_weights(
        self, hf_weights: dict[str, jax.Array], cfg: HookedTransformerConfig
    ) -> dict[str, jax.Array]:
        if not any(k.startswith("transformer.") for k in hf_weights.keys()):
            hf_weights = {f"transformer.{k}": v for k, v in hf_weights.items()} | {
                "lm_head.weight": hf_weights["wte.weight"]
            }
        state_dict: dict[str, jax.Array] = {}

        state_dict["embed.W_E"] = hf_weights["transformer.wte.weight"]
        assert state_dict["embed.W_E"].shape == (cfg.d_vocab, cfg.d_model)
        state_dict["pos_embed.W_pos"] = hf_weights["transformer.wpe.weight"]
        assert state_dict["pos_embed.W_pos"].shape == (cfg.n_ctx, cfg.d_model)

        for l in range(cfg.n_layers):
            state_dict[f"blocks.{l}.ln1.w"] = hf_weights[f"transformer.h.{l}.ln_1.weight"]
            assert state_dict[f"blocks.{l}.ln1.w"].shape == (cfg.d_model,)
            state_dict[f"blocks.{l}.ln1.b"] = hf_weights[f"transformer.h.{l}.ln_1.bias"]
            assert state_dict[f"blocks.{l}.ln1.b"].shape == (cfg.d_model,)

            # In GPT-2, q,k,v are produced by one big linear map, whose output is
            # concat([q, k, v])
            W = hf_weights[f"transformer.h.{l}.attn.c_attn.weight"]
            assert W.shape == (cfg.d_model, 3 * cfg.d_model)
            W_Q, W_K, W_V = jnp.split(W, 3, axis=1)
            W_Q = einops.rearrange(W_Q, "m (i h)->i m h", i=cfg.n_heads)
            W_K = einops.rearrange(W_K, "m (i h)->i m h", i=cfg.n_heads)
            W_V = einops.rearrange(W_V, "m (i h)->i m h", i=cfg.n_heads)

            state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
            state_dict[f"blocks.{l}.attn.W_K"] = W_K
            state_dict[f"blocks.{l}.attn.W_V"] = W_V

            qkv_bias = hf_weights[f"transformer.h.{l}.attn.c_attn.bias"]
            assert qkv_bias.shape == (3 * cfg.d_model,)
            qkv_bias = einops.rearrange(
                qkv_bias,
                "(qkv index head)->qkv index head",
                qkv=3,
                index=cfg.n_heads,
                head=cfg.d_head,
            )
            state_dict[f"blocks.{l}.attn.b_Q"] = qkv_bias[0]
            state_dict[f"blocks.{l}.attn.b_K"] = qkv_bias[1]
            state_dict[f"blocks.{l}.attn.b_V"] = qkv_bias[2]

            W_O = hf_weights[f"transformer.h.{l}.attn.c_proj.weight"]
            assert W_O.shape == (cfg.d_model, cfg.d_model)
            W_O = einops.rearrange(W_O, "(i h) m->i h m", i=cfg.n_heads)
            state_dict[f"blocks.{l}.attn.W_O"] = W_O
            state_dict[f"blocks.{l}.attn.b_O"] = hf_weights[f"transformer.h.{l}.attn.c_proj.bias"]
            assert state_dict[f"blocks.{l}.attn.b_O"].shape == (cfg.d_model,)

            state_dict[f"blocks.{l}.ln2.w"] = hf_weights[f"transformer.h.{l}.ln_2.weight"]
            assert state_dict[f"blocks.{l}.ln2.w"].shape == (cfg.d_model,)
            state_dict[f"blocks.{l}.ln2.b"] = hf_weights[f"transformer.h.{l}.ln_2.bias"]
            assert state_dict[f"blocks.{l}.ln2.b"].shape == (cfg.d_model,)

            state_dict[f"blocks.{l}.mlp.W_in"] = hf_weights[f"transformer.h.{l}.mlp.c_fc.weight"]
            assert state_dict[f"blocks.{l}.mlp.W_in"].shape == (cfg.d_model, cfg.d_mlp)
            state_dict[f"blocks.{l}.mlp.b_in"] = hf_weights[f"transformer.h.{l}.mlp.c_fc.bias"]
            assert state_dict[f"blocks.{l}.mlp.b_in"].shape == (cfg.d_mlp,)

            state_dict[f"blocks.{l}.mlp.W_out"] = hf_weights[f"transformer.h.{l}.mlp.c_proj.weight"]
            assert state_dict[f"blocks.{l}.mlp.W_out"].shape == (cfg.d_mlp, cfg.d_model)
            state_dict[f"blocks.{l}.mlp.b_out"] = hf_weights[f"transformer.h.{l}.mlp.c_proj.bias"]
            assert state_dict[f"blocks.{l}.mlp.b_out"].shape == (cfg.d_model,)

        state_dict["unembed.W_U"] = hf_weights["lm_head.weight"].T
        assert state_dict["unembed.W_U"].shape == (cfg.d_model, cfg.d_vocab)

        state_dict["ln_final.w"] = hf_weights["transformer.ln_f.weight"]
        assert state_dict["ln_final.w"].shape == (cfg.d_model,)
        state_dict["ln_final.b"] = hf_weights["transformer.ln_f.bias"]
        assert state_dict["ln_final.b"].shape == (cfg.d_model,)

        return state_dict
