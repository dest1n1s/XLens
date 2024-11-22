import timeit

import jax
import pytest
from transformers import GPT2Tokenizer

from xlens import HookedTransformer

jax.config.update("jax_default_matmul_precision", "highest")


def test_gpt2_computation_speed():
    model = HookedTransformer.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input = tokenizer("Hello, my dog is cute", return_tensors="np")["input_ids"]

    def forward(input):
        return model(input)[0]

    print(f"JAX not jitted: {timeit.timeit(lambda: forward(input), number=10) / 10} seconds")
    jitted_forward = jax.jit(forward)
    print(f"JAX jitting: {timeit.timeit(lambda: jitted_forward(input), number=1) / 1} seconds")
    print(f"JAX jitted: {timeit.timeit(lambda: jitted_forward(input), number=10) / 10} seconds")


pytest.importorskip("torch")

import torch  # noqa: E402
import transformer_lens as tl  # noqa: E402
from transformers import GPT2LMHeadModel  # noqa: E402

torch.set_float32_matmul_precision("high")


@torch.no_grad()
def test_gpt2_computation_speed_hf():
    hf_model: GPT2LMHeadModel = GPT2LMHeadModel.from_pretrained("gpt2").cuda()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    input = tokenizer("Hello, my dog is cute", return_tensors="pt")["input_ids"].cuda()

    @torch.compile
    def forward(input):
        return hf_model(input)

    torch.cuda.synchronize()
    print(f"Torch HF Compiling: {timeit.timeit(lambda: forward(input), number=1) / 1} seconds")
    torch.cuda.synchronize()
    print(f"Torch HF: {timeit.timeit(lambda: forward(input), number=100) / 100} seconds")


@torch.no_grad()
def test_gpt2_computation_speed_tl():
    model = tl.HookedTransformer.from_pretrained("gpt2").cuda()
    input = model.to_tokens("Hello, my dog is cute")

    # @torch.compile
    def forward(input):
        return model.forward(input)

    torch.cuda.synchronize()
    print(f"TransformerLens Compiling: {timeit.timeit(lambda: forward(input), number=1) / 1} seconds")
    torch.cuda.synchronize()
    print(f"TransformerLens: {timeit.timeit(lambda: forward(input), number=100) / 100} seconds")


if __name__ == "__main__":
    test_gpt2_computation_speed()
    test_gpt2_computation_speed_hf()
    test_gpt2_computation_speed_tl()
