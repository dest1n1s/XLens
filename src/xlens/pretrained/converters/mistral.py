from typing import Any

import einops
import jax
import jax.numpy as jnp

from xlens.config import HookedTransformerConfig
from xlens.pretrained.model_converter import HuggingFaceModelConverterSingle


class MistralConverter(HuggingFaceModelConverterSingle):
    def __init__(self):
        super().__init__(
            model_names=[
                "mistralai/Mistral-7B-v0.1",
                "mistralai/Mistral-7B-Instruct-v0.1",
                "mistralai/Mistral-Nemo-Base-2407",
            ],
            model_alias_map={
                "mistralai/Mistral-7B-v0.1": ["mistral-7b"],
                "mistralai/Mistral-7B-Instruct-v0.1": ["mistral-7b-instruct"],
                "mistralai/Mistral-Nemo-Base-2407": ["mistral-nemo-base-2407"],
            },
            model_architecture="MistralForCausalLM",
        )

    def convert_hf_model_config(self, hf_cfg: Any) -> HookedTransformerConfig:
        use_local_attn = True if hf_cfg.sliding_window else False
        return HookedTransformerConfig(
            d_model=hf_cfg.hidden_size,
            d_head=hf_cfg.head_dim
            if hasattr(hf_cfg, "head_dim") and hf_cfg.head_dim > 0
            else hf_cfg.hidden_size // hf_cfg.num_attention_heads,
            n_heads=hf_cfg.num_attention_heads,
            d_mlp=hf_cfg.intermediate_size,
            n_layers=hf_cfg.num_hidden_layers,
            n_ctx=2048,  # Capped due to memory issues
            d_vocab=hf_cfg.vocab_size,
            act_fn=hf_cfg.hidden_act,
            window_size=hf_cfg.sliding_window,  # None if no sliding window was used
            attn_types=["local"] * hf_cfg.num_hidden_layers if use_local_attn else None,
            eps=hf_cfg.rms_norm_eps,
            rotary_base=hf_cfg.rope_theta,
            n_key_value_heads=hf_cfg.num_key_value_heads,
            use_local_attn=use_local_attn,
            normalization_type="RMS",
            positional_embedding_type="rotary",
            gated_mlp=True,
            original_architecture="MistralForCausalLM",
        )

    def convert_hf_weights(
        self, hf_weights: dict[str, jax.Array], cfg: HookedTransformerConfig
    ) -> dict[str, jax.Array]:
        if not any(k.startswith("model.") for k in hf_weights.keys()):
            hf_weights = {f"model.{k}": v for k, v in hf_weights.items()}
        if "lm_head.weight" not in hf_weights:
            hf_weights = {**hf_weights, "lm_head.weight": hf_weights["model.embed_tokens.weight"]}
        state_dict = {}

        state_dict["embed.W_E"] = hf_weights["model.embed_tokens.weight"]

        assert cfg.n_key_value_heads is not None  # keep mypy happy
        assert cfg.d_mlp is not None  # keep mypy happy

        # Mistral has no biases anywhere
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

            state_dict[f"blocks.{l}.attn.b_Q"] = jnp.zeros((cfg.n_heads, cfg.d_head))
            state_dict[f"blocks.{l}.attn.b_K"] = jnp.zeros((cfg.n_key_value_heads, cfg.d_head))
            state_dict[f"blocks.{l}.attn.b_V"] = jnp.zeros((cfg.n_key_value_heads, cfg.d_head))

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
