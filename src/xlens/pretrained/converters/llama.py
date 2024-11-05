from typing import Any

import einops
import jax
import jax.numpy as jnp

from xlens.config import HookedTransformerConfig
from xlens.pretrained.model_converter import HuggingFaceModelConverterSingle


class LlamaConverter(HuggingFaceModelConverterSingle):
    def __init__(self):
        super().__init__(
            model_names=[
                "llama-7b-hf",
                "llama-13b-hf",
                "llama-30b-hf",
                "llama-65b-hf",
                "meta-llama/Llama-2-7b-hf",
                "meta-llama/Llama-2-7b-chat-hf",
                "meta-llama/Llama-2-13b-hf",
                "meta-llama/Llama-2-13b-chat-hf",
                "meta-llama/Llama-2-70b-chat-hf",
                "meta-llama/Meta-Llama-3-8B",
                "meta-llama/Meta-Llama-3-8B-Instruct",
                "meta-llama/Meta-Llama-3-70B",
                "meta-llama/Meta-Llama-3-70B-Instruct",
                "meta-llama/Llama-3.2-1B",
                "meta-llama/Llama-3.2-3B",
                "meta-llama/Llama-3.2-1B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct",
                "meta-llama/Llama-3.1-70B",
                "meta-llama/Llama-3.1-8B",
                "meta-llama/Llama-3.1-8B-Instruct",
                "meta-llama/Llama-3.1-70B-Instruct",
                # Add other Llama model variants here
            ],
            model_alias_map={
                "llama-7b-hf": ["llama-7b"],
                "llama-13b-hf": ["llama-13b"],
                "llama-30b-hf": ["llama-30b"],
                "llama-65b-hf": ["llama-65b"],
                "meta-llama/Llama-2-7b-hf": ["Llama-2-7b"],
                "meta-llama/Llama-2-7b-chat-hf": ["Llama-2-7b-chat"],
                "meta-llama/Llama-2-13b-hf": ["Llama-2-13b"],
                "meta-llama/Llama-2-13b-chat-hf": ["Llama-2-13b-chat"],
                "meta-llama/Llama-2-70b-chat-hf": ["Llama-2-70b-chat", "meta-llama-2-70b-chat-hf"],
                "meta-llama/Meta-Llama-3-8B": ["Meta-Llama-3-8B"],
                "meta-llama/Meta-Llama-3-8B-Instruct": ["Meta-Llama-3-8B-Instruct"],
                "meta-llama/Meta-Llama-3-70B": ["Meta-Llama-3-70B"],
                "meta-llama/Meta-Llama-3-70B-Instruct": ["Meta-Llama-3-70B-Instruct"],
                "meta-llama/Llama-3.2-1B": ["Llama-3.2-1B"],
                "meta-llama/Llama-3.2-3B": ["Llama-3.2-3B"],
                "meta-llama/Llama-3.2-1B-Instruct": ["Llama-3.2-1B-Instruct"],
                "meta-llama/Llama-3.2-3B-Instruct": ["Llama-3.2-3B-Instruct"],
                "meta-llama/Llama-3.1-70B": ["Llama-3.1-70B"],
                "meta-llama/Llama-3.1-8B": ["Llama-3.1-8B"],
                "meta-llama/Llama-3.1-8B-Instruct": ["Llama-3.1-8B-Instruct"],
                "meta-llama/Llama-3.1-70B-Instruct": ["Llama-3.1-70B-Instruct"],
            },
            model_architecture="LlamaForCausalLM",
        )

    def convert_hf_model_config(self, hf_cfg: Any) -> HookedTransformerConfig:
        if hasattr(hf_cfg, "rope_scaling") and hf_cfg.rope_scaling is not None:
            ntk_cfg: dict[str, Any] = {
                "use_NTK_by_parts_rope": True,
                "NTK_by_parts_low_freq_factor": hf_cfg.rope_scaling["low_freq_factor"],
                "NTK_by_parts_high_freq_factor": hf_cfg.rope_scaling["high_freq_factor"],
                "NTK_by_parts_factor": hf_cfg.rope_scaling["factor"],
            }
        else:
            ntk_cfg = {}
        return HookedTransformerConfig(
            d_model=hf_cfg.hidden_size,
            d_head=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
            n_heads=hf_cfg.num_attention_heads,
            d_mlp=hf_cfg.intermediate_size,
            n_layers=hf_cfg.num_hidden_layers,
            n_ctx=min(hf_cfg.max_position_embeddings, 2048),  # capped due to memory issues
            eps=hf_cfg.rms_norm_eps,
            d_vocab=hf_cfg.vocab_size,
            act_fn=hf_cfg.hidden_act,
            n_key_value_heads=(
                hf_cfg.num_key_value_heads if hf_cfg.num_key_value_heads != hf_cfg.num_attention_heads else None
            ),
            normalization_type="RMS",
            positional_embedding_type="rotary",
            rotary_dim=hf_cfg.hidden_size // hf_cfg.num_attention_heads,
            rotary_base=hf_cfg.rope_theta if hasattr(hf_cfg, "rope_theta") else 10000,
            rotary_adjacent_pairs=False,
            **ntk_cfg,
            final_rms=True,
            gated_mlp=True,
            original_architecture="LlamaForCausalLM",
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

        # Some models with the Llama architecture use Grouped Query Attention.
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

        return state_dict
