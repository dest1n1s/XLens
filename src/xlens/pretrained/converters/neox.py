from typing import Any

import einops
import jax

from xlens.config import HookedTransformerConfig
from xlens.pretrained.model_converter import HuggingFaceModelConverterSingle


class GPTNeoXConverter(HuggingFaceModelConverterSingle):
    def __init__(self):
        super().__init__(
            model_names=[
                "EleutherAI/pythia-14m",
                "EleutherAI/pythia-31m",
                "EleutherAI/pythia-70m",
                "EleutherAI/pythia-160m",
                "EleutherAI/pythia-410m",
                "EleutherAI/pythia-1b",
                "EleutherAI/pythia-1.4b",
                "EleutherAI/pythia-2.8b",
                "EleutherAI/pythia-6.9b",
                "EleutherAI/pythia-12b",
                "EleutherAI/pythia-70m-deduped",
                "EleutherAI/pythia-160m-deduped",
                "EleutherAI/pythia-410m-deduped",
                "EleutherAI/pythia-1b-deduped",
                "EleutherAI/pythia-1.4b-deduped",
                "EleutherAI/pythia-2.8b-deduped",
                "EleutherAI/pythia-6.9b-deduped",
                "EleutherAI/pythia-12b-deduped",
                "EleutherAI/pythia-70m-v0",
                "EleutherAI/pythia-160m-v0",
                "EleutherAI/pythia-410m-v0",
                "EleutherAI/pythia-1b-v0",
                "EleutherAI/pythia-1.4b-v0",
                "EleutherAI/pythia-2.8b-v0",
                "EleutherAI/pythia-6.9b-v0",
                "EleutherAI/pythia-12b-v0",
                "EleutherAI/pythia-70m-deduped-v0",
                "EleutherAI/pythia-160m-deduped-v0",
                "EleutherAI/pythia-410m-deduped-v0",
                "EleutherAI/pythia-1b-deduped-v0",
                "EleutherAI/pythia-1.4b-deduped-v0",
                "EleutherAI/pythia-2.8b-deduped-v0",
                "EleutherAI/pythia-6.9b-deduped-v0",
                "EleutherAI/pythia-12b-deduped-v0",
                "EleutherAI/pythia-160m-seed1",
                "EleutherAI/pythia-160m-seed2",
                "EleutherAI/pythia-160m-seed3",
            ],
            model_alias_map={
                "EleutherAI/pythia-70m": ["pythia-70m", "pythia", "EleutherAI/pythia-19m", "pythia-19m"],
                "EleutherAI/pythia-160m": ["pythia-160m", "EleutherAI/pythia-125m", "pythia-125m"],
                "EleutherAI/pythia-410m": ["pythia-410m", "EleutherAI/pythia-350m", "pythia-350m"],
                "EleutherAI/pythia-1b": ["pythia-1b", "EleutherAI/pythia-800m", "pythia-800m"],
                "EleutherAI/pythia-1.4b": ["pythia-1.4b", "EleutherAI/pythia-1.3b", "pythia-1.3b"],
                "EleutherAI/pythia-2.8b": ["pythia-2.8b", "EleutherAI/pythia-2.7b", "pythia-2.7b"],
                "EleutherAI/pythia-6.9b": ["pythia-6.9b", "EleutherAI/pythia-6.7b", "pythia-6.7b"],
                "EleutherAI/pythia-12b": ["pythia-12b", "EleutherAI/pythia-13b", "pythia-13b"],
            },
            model_architecture="GPTNeoXForCausalLM",
        )

    def convert_hf_model_config(self, hf_cfg: Any) -> HookedTransformerConfig:
        cfg_dict = {
            "d_model": hf_cfg.hidden_size,
            "d_head": hf_cfg.hidden_size // hf_cfg.num_attention_heads,
            "n_heads": hf_cfg.num_attention_heads,
            "d_mlp": hf_cfg.intermediate_size,
            "n_layers": hf_cfg.num_hidden_layers,
            "n_ctx": hf_cfg.max_position_embeddings,
            "eps": hf_cfg.layer_norm_eps,
            "d_vocab": hf_cfg.vocab_size,
            "act_fn": hf_cfg.hidden_act,
            "use_attn_scale": True,
            "scale_attn_by_inverse_layer_idx": False,
            "parallel_attn_mlp": True,
            "positional_embedding_type": "rotary",
            "rotary_adjacent_pairs": False,
            "normalization_type": "LN",
            "original_architecture": "GPTNeoXForCausalLM",
        }
        rotary_pct = hf_cfg.rotary_pct
        cfg_dict["rotary_dim"] = round(rotary_pct * cfg_dict["d_head"])
        return HookedTransformerConfig.from_dict(cfg_dict)

    def convert_hf_weights(
        self, hf_weights: dict[str, jax.Array], cfg: HookedTransformerConfig
    ) -> dict[str, jax.Array]:
        if not any(k.startswith("gpt_neox.") for k in hf_weights.keys()):
            hf_weights = {f"gpt_neox.{k}": v for k, v in hf_weights.items()}
        if "embed_out.weight" not in hf_weights:
            hf_weights = {**hf_weights, "embed_out.weight": hf_weights["gpt_neox.embed_in.weight"]}

        state_dict = {}

        state_dict["embed.W_E"] = hf_weights["gpt_neox.embed_in.weight"]

        for l in range(cfg.n_layers):
            state_dict[f"blocks.{l}.ln1.w"] = hf_weights[f"gpt_neox.layers.{l}.input_layernorm.weight"]
            state_dict[f"blocks.{l}.ln1.b"] = hf_weights[f"gpt_neox.layers.{l}.input_layernorm.bias"]

            # For some inexplicable reason, NeoX both uses the concatenated QKV
            # matmul of GPT-2 (afaict this has a neglible performance impact) AND
            # has the flattened axis in the DIFFERENT order of (head_index qkv
            # d_head) - this took me an hour to debug...
            W = hf_weights[f"gpt_neox.layers.{l}.attention.query_key_value.weight"]
            W = einops.rearrange(W, "(i qkv h) m->qkv i m h", i=cfg.n_heads, qkv=3)

            state_dict[f"blocks.{l}.attn.W_Q"] = W[0]
            state_dict[f"blocks.{l}.attn.W_K"] = W[1]
            state_dict[f"blocks.{l}.attn.W_V"] = W[2]

            qkv_bias = hf_weights[f"gpt_neox.layers.{l}.attention.query_key_value.bias"]
            qkv_bias = einops.rearrange(
                qkv_bias,
                "(index qkv head)->qkv index head",
                qkv=3,
                index=cfg.n_heads,
                head=cfg.d_head,
            )

            state_dict[f"blocks.{l}.attn.b_Q"] = qkv_bias[0]
            state_dict[f"blocks.{l}.attn.b_K"] = qkv_bias[1]
            state_dict[f"blocks.{l}.attn.b_V"] = qkv_bias[2]

            W_O = hf_weights[f"gpt_neox.layers.{l}.attention.dense.weight"]
            W_O = einops.rearrange(W_O, "m (i h)->i h m", i=cfg.n_heads)
            state_dict[f"blocks.{l}.attn.W_O"] = W_O
            state_dict[f"blocks.{l}.attn.b_O"] = hf_weights[f"gpt_neox.layers.{l}.attention.dense.bias"]

            state_dict[f"blocks.{l}.ln2.w"] = hf_weights[f"gpt_neox.layers.{l}.post_attention_layernorm.weight"]
            state_dict[f"blocks.{l}.ln2.b"] = hf_weights[f"gpt_neox.layers.{l}.post_attention_layernorm.bias"]

            state_dict[f"blocks.{l}.mlp.W_in"] = hf_weights[f"gpt_neox.layers.{l}.mlp.dense_h_to_4h.weight"].T
            state_dict[f"blocks.{l}.mlp.b_in"] = hf_weights[f"gpt_neox.layers.{l}.mlp.dense_h_to_4h.bias"]

            state_dict[f"blocks.{l}.mlp.W_out"] = hf_weights[f"gpt_neox.layers.{l}.mlp.dense_4h_to_h.weight"].T
            state_dict[f"blocks.{l}.mlp.b_out"] = hf_weights[f"gpt_neox.layers.{l}.mlp.dense_4h_to_h.bias"]

        state_dict["ln_final.w"] = hf_weights["gpt_neox.final_layer_norm.weight"]
        state_dict["ln_final.b"] = hf_weights["gpt_neox.final_layer_norm.bias"]

        state_dict["unembed.W_U"] = hf_weights["embed_out.weight"].T
        return state_dict
