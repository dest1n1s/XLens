from typing import Any

import einops
import jax
import jax.numpy as jnp

from xlens.config import HookedTransformerConfig
from xlens.pretrained.model_converter import HuggingFaceModelConverterSingle


class ChameleonConverter(HuggingFaceModelConverterSingle):
    def __init__(self):
        super().__init__(
            model_names=[
                "facebook/chameleon-7b",  # Add actual Chameleon model names here
            ],
            model_alias_map={
                "facebook/chameleon-7b": ["chameleon-7b"],  # Add appropriate aliases
            },
            model_architecture="ChameleonForCausalLM",
        )

    def convert_hf_model_config(self, hf_cfg: Any) -> HookedTransformerConfig:
        return HookedTransformerConfig(
            d_model=hf_cfg.hidden_size,
            d_head=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
            n_heads=hf_cfg.num_attention_heads,
            d_mlp=hf_cfg.intermediate_size,
            n_layers=hf_cfg.num_hidden_layers,
            n_ctx=hf_cfg.max_position_embeddings,
            d_vocab=hf_cfg.vocab_size,
            act_fn=hf_cfg.hidden_act,
            n_key_value_heads=(
                hf_cfg.num_key_value_heads if hf_cfg.num_key_value_heads != hf_cfg.num_attention_heads else None
            ),
            normalization_type="RMS",
            positional_embedding_type="rotary",
            gated_mlp=True,
            original_architecture="ChameleonForCausalLM",
        )

    def convert_hf_weights(
        self, hf_weights: dict[str, jax.Array], cfg: HookedTransformerConfig
    ) -> dict[str, jax.Array]:
        if not any(k.startswith("model.") for k in hf_weights.keys()):
            hf_weights = {f"model.{k}": v for k, v in hf_weights.items()}
        if "lm_head.weight" not in hf_weights:
            hf_weights = {**hf_weights, "lm_head.weight": hf_weights["model.embed_tokens.weight"]}

        state_dict: dict[str, jax.Array] = {}
        state_dict["embed.W_E"] = hf_weights["model.embed_tokens.weight"]

        n_kv_heads = cfg.n_key_value_heads if cfg.n_key_value_heads is not None else cfg.n_heads

        for l in range(cfg.n_layers):
            state_dict[f"blocks.{l}.ln1.w"] = hf_weights[f"model.layers.{l}.input_layernorm.weight"]

            W_Q = hf_weights[f"model.layers.{l}.self_attn.q_proj.weight"]
            W_K = hf_weights[f"model.layers.{l}.self_attn.k_proj.weight"]
            W_V = hf_weights[f"model.layers.{l}.self_attn.v_proj.weight"]

            W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
            W_K = einops.rearrange(W_K, "(n h) m->n m h", n=n_kv_heads)
            W_V = einops.rearrange(W_V, "(n h) m->n m h", n=n_kv_heads)

            state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
            state_dict[f"blocks.{l}.attn.W_K"] = W_K
            state_dict[f"blocks.{l}.attn.W_V"] = W_V

            state_dict[f"blocks.{l}.attn.b_Q"] = jnp.zeros((cfg.n_heads, cfg.d_head))
            state_dict[f"blocks.{l}.attn.b_K"] = jnp.zeros((n_kv_heads, cfg.d_head))
            state_dict[f"blocks.{l}.attn.b_V"] = jnp.zeros((n_kv_heads, cfg.d_head))

            # Add layernorm weights for Q and K
            state_dict[f"blocks.{l}.attn.ln_q.w"] = hf_weights[f"model.layers.{l}.self_attn.q_norm.weight"]
            state_dict[f"blocks.{l}.attn.ln_q.b"] = hf_weights[f"model.layers.{l}.self_attn.q_norm.bias"]
            state_dict[f"blocks.{l}.attn.ln_k.w"] = hf_weights[f"model.layers.{l}.self_attn.k_norm.weight"]
            state_dict[f"blocks.{l}.attn.ln_k.b"] = hf_weights[f"model.layers.{l}.self_attn.k_norm.bias"]

            W_O = hf_weights[f"model.layers.{l}.self_attn.o_proj.weight"]
            W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)

            state_dict[f"blocks.{l}.attn.W_O"] = W_O
            state_dict[f"blocks.{l}.attn.b_O"] = jnp.zeros(cfg.d_model)

            state_dict[f"blocks.{l}.ln2.w"] = hf_weights[f"model.layers.{l}.post_attention_layernorm.weight"]

            state_dict[f"blocks.{l}.mlp.W_in"] = hf_weights[f"model.layers.{l}.mlp.up_proj.weight"].T
            state_dict[f"blocks.{l}.mlp.W_gate"] = hf_weights[f"model.layers.{l}.mlp.gate_proj.weight"].T
            state_dict[f"blocks.{l}.mlp.W_out"] = hf_weights[f"model.layers.{l}.mlp.down_proj.weight"].T

            state_dict[f"blocks.{l}.mlp.b_in"] = jnp.zeros(cfg.d_mlp)
            state_dict[f"blocks.{l}.mlp.b_out"] = jnp.zeros(cfg.d_model)

        state_dict["ln_final.w"] = hf_weights["model.norm.weight"]
        state_dict["unembed.W_U"] = hf_weights["lm_head.weight"].T
        state_dict["unembed.b_U"] = jnp.zeros(cfg.d_vocab)

        return state_dict
