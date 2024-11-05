import jax
import jax.numpy as jnp
import pytest

from xlens import HookedTransformer

pytest.importorskip("torch")

import torch  # noqa: E402
from transformers import AutoTokenizer, MistralForCausalLM  # noqa: E402

jax.config.update("jax_default_matmul_precision", "highest")


@torch.no_grad()
def test_mistral_computation():
    hf_model = MistralForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1", torch_dtype=torch.float32, attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    hf_model.eval()

    hf_input: torch.Tensor = tokenizer("Hello, my dog is cute!", return_tensors="pt")["input_ids"]
    hf_output = hf_model(hf_input, output_hidden_states=True)
    hf_logits = hf_output.logits
    hf_hidden_states = hf_output.hidden_states

    del hf_model
    torch.cuda.empty_cache()

    model = HookedTransformer.from_pretrained("mistralai/Mistral-7B-v0.1")

    input = jnp.array(hf_input)
    logits, cache = model.run_with_cache(input, hook_names=[f"blocks.{i}.hook_resid_pre" for i in range(12)])

    for i in range(12):
        print(
            f"Block {i} Residual Pre Difference: ",
            jnp.linalg.norm(jnp.array(hf_hidden_states[i]) - cache[f"blocks.{i}.hook_resid_pre"]),
        )

    print("Logits Difference: ", jnp.linalg.norm(logits - jnp.array(hf_logits)))

    hf_probs = torch.nn.functional.softmax(hf_logits, dim=-1)
    probs = jax.nn.softmax(logits, axis=-1)

    print("Probs Difference: ", jnp.linalg.norm(probs - jnp.array(hf_probs)))

    assert jnp.allclose(probs, jnp.array(hf_probs), atol=1e-3)


# @torch.no_grad()
# def test_mistral_computation_tl():
#     import transformer_lens as tl

#     tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
#     tl_model = tl.HookedTransformer.from_pretrained("mistralai/Mistral-7B-v0.1", tokenizer=tokenizer, fold_ln=False)
#     tl_model.eval()

#     tl_input = tokenizer("Hello, my dog is cute!", return_tensors="pt")["input_ids"]
#     tl_logits, tl_cache = tl_model.run_with_cache(
#         tl_input, names_filter=[f"blocks.{i}.hook_resid_pre" for i in range(12)]
#     )

#     model = HookedTransformer.from_pretrained("mistralai/Mistral-7B-v0.1")

#     input = jnp.array(tl_input)
#     logits, cache = model.run_with_cache(input, hook_names=[f"blocks.{i}.hook_resid_pre" for i in range(12)])

#     for i in range(12):
#         print(
#             f"Block {i} Residual Pre Difference: ",
#             jnp.linalg.norm(jnp.array(tl_cache[f"blocks.{i}.hook_resid_pre"]) - cache[f"blocks.{i}.hook_resid_pre"]),
#         )

#     print("Logits Difference: ", jnp.linalg.norm(logits - jnp.array(tl_logits)))

#     tl_probs = torch.nn.functional.softmax(tl_logits, dim=-1)
#     probs = jax.nn.softmax(logits, axis=-1)

#     print("Probs Difference: ", jnp.linalg.norm(probs - jnp.array(tl_probs)))

#     assert jnp.allclose(probs, jnp.array(tl_probs), atol=1e-3)


if __name__ == "__main__":
    test_mistral_computation()
