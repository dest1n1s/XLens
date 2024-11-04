import pytest

from xlens import HookedTransformer
from xlens.pretrained import get_pretrained_model_config, get_pretrained_state_dict


def test_get_pretrained_state_dict():
    cfg = get_pretrained_model_config("gpt2")
    state_dict = get_pretrained_state_dict("gpt2", cfg)
    assert "unembed.W_U" in state_dict


def test_from_pretrained():
    model = HookedTransformer.from_pretrained("gpt2")
    assert model is not None


pytest.importorskip("torch")

from transformers import GPT2LMHeadModel  # noqa: E402


def test_from_pretrained_from_hf_model():
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = HookedTransformer.from_pretrained("gpt2", hf_model=hf_model)
    assert model is not None
