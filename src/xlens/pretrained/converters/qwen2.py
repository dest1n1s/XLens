from typing import Any

import einops
import jax
import jax.numpy as jnp

from xlens.config import HookedTransformerConfig
from xlens.pretrained.model_converter import HuggingFaceModelConverterSingle


class Qwen2Converter(HuggingFaceModelConverterSingle):
    def __init__(self):
        super().__init__(
            model_names=[
                "Qwen/Qwen1.5-0.5B",
                "Qwen/Qwen1.5-1.8B",
                "Qwen/Qwen1.5-4B",
                "Qwen/Qwen1.5-7B",
                "Qwen/Qwen1.5-14B",
                "Qwen/Qwen1.5-72B",
                "Qwen/Qwen2-0.5B",
                "Qwen/Qwen2-1.8B",
                "Qwen/Qwen2-4B",
                "Qwen/Qwen2-7B",
                "Qwen/Qwen2-13B",
                "Qwen/Qwen2-72B",
                "Qwen/Qwen2-0.5B-Instruct",
                "Qwen/Qwen2-1.8B-Instruct",
                "Qwen/Qwen2-4B-Instruct",
                "Qwen/Qwen2-7B-Instruct",
                "Qwen/Qwen2-13B-Instruct",
                "Qwen/Qwen2-72B-Instruct",
            ],
            model_alias_map={
                "Qwen/Qwen1.5-0.5B": ["qwen1.5-0.5b", "qwen15-0.5b"],
                "Qwen/Qwen1.5-1.8B": ["qwen1.5-1.8b", "qwen15-1.8b"],
                "Qwen/Qwen1.5-4B": ["qwen1.5-4b", "qwen15-4b"],
                "Qwen/Qwen1.5-7B": ["qwen1.5-7b", "qwen15-7b"],
                "Qwen/Qwen1.5-14B": ["qwen1.5-14b", "qwen15-14b"],
                "Qwen/Qwen1.5-72B": ["qwen1.5-72b", "qwen15-72b"],
                "Qwen/Qwen2-0.5B": ["qwen2-0.5b"],
                "Qwen/Qwen2-1.8B": ["qwen2-1.8b"],
                "Qwen/Qwen2-4B": ["qwen2-4b"],
                "Qwen/Qwen2-7B": ["qwen2-7b"],
                "Qwen/Qwen2-13B": ["qwen2-13b"],
                "Qwen/Qwen2-72B": ["qwen2-72b"],
            },
            model_architecture="Qwen2ForCausalLM",
        )

    def convert_hf_model_config(self, hf_cfg: Any) -> HookedTransformerConfig:
        return HookedTransformerConfig(
            d_model=hf_cfg.hidden_size,
            d_head=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
            n_heads=hf_cfg.num_attention_heads,
            n_key_value_heads=hf_cfg.num_key_value_heads,
            d_mlp=hf_cfg.intermediate_size,
            n_layers=hf_cfg.num_hidden_layers,
            n_ctx=2048,  # Capped bc the actual ctx length is 30k and the attn mask would be too big
            eps=hf_cfg.rms_norm_eps,
            d_vocab=hf_cfg.vocab_size,
            act_fn=hf_cfg.hidden_act,
            use_attn_scale=True,
            initializer_range=hf_cfg.initializer_range,
            normalization_type="RMS",
            positional_embedding_type="rotary",
            rotary_base=hf_cfg.rope_theta,
            rotary_adjacent_pairs=False,
            rotary_dim=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
            final_rms=True,
            gated_mlp=True,
            original_architecture="Qwen2ForCausalLM",
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

        assert cfg.d_mlp is not None  # keep mypy happy

        for l in range(cfg.n_layers):
            state_dict[f"blocks.{l}.ln1.w"] = hf_weights[f"model.layers.{l}.input_layernorm.weight"]

            W_Q = hf_weights[f"model.layers.{l}.self_attn.q_proj.weight"]
            W_K = hf_weights[f"model.layers.{l}.self_attn.k_proj.weight"]
            W_V = hf_weights[f"model.layers.{l}.self_attn.v_proj.weight"]
            W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
            W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_key_value_heads)
            W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_key_value_heads)
            state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
            state_dict[f"blocks.{l}.attn.W_K"] = W_K
            state_dict[f"blocks.{l}.attn.W_V"] = W_V

            b_Q = hf_weights[f"model.layers.{l}.self_attn.q_proj.bias"]
            b_Q = einops.rearrange(
                b_Q,
                "(n_head d_head) -> n_head d_head",
                n_head=cfg.n_heads,
            )

            b_K = hf_weights[f"model.layers.{l}.self_attn.k_proj.bias"]
            b_K = einops.rearrange(
                b_K,
                "(n_head d_head) -> n_head d_head",
                n_head=cfg.n_key_value_heads,
            )

            b_V = hf_weights[f"model.layers.{l}.self_attn.v_proj.bias"]
            b_V = einops.rearrange(
                b_V,
                "(n_head d_head) -> n_head d_head",
                n_head=cfg.n_key_value_heads,
            )

            state_dict[f"blocks.{l}.attn.b_Q"] = b_Q
            state_dict[f"blocks.{l}.attn.b_K"] = b_K
            state_dict[f"blocks.{l}.attn.b_V"] = b_V

            W_O = hf_weights[f"model.layers.{l}.self_attn.o_proj.weight"]
            W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
            state_dict[f"blocks.{l}.attn.W_O"] = W_O

            state_dict[f"blocks.{l}.attn.b_O"] = jnp.zeros(cfg.d_model)

            state_dict[f"blocks.{l}.ln2.w"] = hf_weights[f"model.layers.{l}.post_attention_layernorm.weight"]

            state_dict[f"blocks.{l}.mlp.W_in"] = hf_weights[f"model.layers.{l}.mlp.up_proj.weight"].T
            state_dict[f"blocks.{l}.mlp.W_gate"] = hf_weights[f"model.layers.{l}.mlp.gate_proj.weight"].T
            state_dict[f"blocks.{l}.mlp.b_in"] = jnp.zeros(cfg.d_mlp)

            state_dict[f"blocks.{l}.mlp.W_out"] = hf_weights[f"model.layers.{l}.mlp.down_proj.weight"].T
            state_dict[f"blocks.{l}.mlp.b_out"] = jnp.zeros(cfg.d_model)

        state_dict["ln_final.w"] = hf_weights["model.norm.weight"]

        state_dict["unembed.W_U"] = hf_weights["lm_head.weight"].T

        return state_dict
