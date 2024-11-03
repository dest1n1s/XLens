import einops
import jax
import jax.numpy as jnp

from xlens.config import HookedTransformerConfig


def convert_mistral_weights(params: dict[str, jax.Array], cfg: HookedTransformerConfig):
    if not any(k.startswith("model.") for k in params.keys()):
        params = {f"model.{k}": v for k, v in params.items()}
    if "lm_head.weight" not in params:
        params = {**params, "lm_head.weight": params["model.embed_tokens.weight"]}
    state_dict = {}

    state_dict["embed.W_E"] = params["model.embed_tokens.weight"]

    assert cfg.n_key_value_heads is not None  # keep mypy happy
    assert cfg.d_mlp is not None  # keep mypy happy

    # Mistral has no biases anywhere
    for l in range(cfg.n_layers):
        state_dict[f"blocks.{l}.ln1.w"] = params[f"model.layers.{l}.input_layernorm.weight"]

        W_Q = params[f"model.layers.{l}.self_attn.q_proj.weight"]
        W_K = params[f"model.layers.{l}.self_attn.k_proj.weight"]
        W_V = params[f"model.layers.{l}.self_attn.v_proj.weight"]
        W_Q = einops.rearrange(W_Q, "(n h) m->n m h", n=cfg.n_heads)
        W_K = einops.rearrange(W_K, "(n h) m->n m h", n=cfg.n_key_value_heads)
        W_V = einops.rearrange(W_V, "(n h) m->n m h", n=cfg.n_key_value_heads)
        state_dict[f"blocks.{l}.attn.W_Q"] = W_Q
        state_dict[f"blocks.{l}.attn._W_K"] = W_K
        state_dict[f"blocks.{l}.attn._W_V"] = W_V

        state_dict[f"blocks.{l}.attn.b_Q"] = jnp.zeros((cfg.n_heads, cfg.d_head))
        state_dict[f"blocks.{l}.attn.b_K"] = jnp.zeros((cfg.n_key_value_heads, cfg.d_head))
        state_dict[f"blocks.{l}.attn.b_V"] = jnp.zeros((cfg.n_key_value_heads, cfg.d_head))

        W_O = params[f"model.layers.{l}.self_attn.o_proj.weight"]
        W_O = einops.rearrange(W_O, "m (n h)->n h m", n=cfg.n_heads)
        state_dict[f"blocks.{l}.attn.W_O"] = W_O

        state_dict[f"blocks.{l}.attn.b_O"] = jnp.zeros(cfg.d_model)

        state_dict[f"blocks.{l}.ln2.w"] = params[f"model.layers.{l}.post_attention_layernorm.weight"]

        state_dict[f"blocks.{l}.mlp.W_in"] = params[f"model.layers.{l}.mlp.up_proj.weight"].T
        state_dict[f"blocks.{l}.mlp.W_gate"] = params[f"model.layers.{l}.mlp.gate_proj.weight"].T
        state_dict[f"blocks.{l}.mlp.b_in"] = jnp.zeros(cfg.d_mlp)

        state_dict[f"blocks.{l}.mlp.W_out"] = params[f"model.layers.{l}.mlp.down_proj.weight"].T
        state_dict[f"blocks.{l}.mlp.b_out"] = jnp.zeros(cfg.d_model)

    state_dict["ln_final.w"] = params["model.norm.weight"]

    state_dict["unembed.W_U"] = params["lm_head.weight"].T

    return state_dict
