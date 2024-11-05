import pytest

from xlens import HookedTransformer
from xlens.pretrained import get_pretrained_model_config, get_pretrained_weights


def test_get_pretrained_weights():
    cfg = get_pretrained_model_config("gpt2")
    weights = get_pretrained_weights(cfg, "gpt2")
    assert "unembed.W_U" in weights


def test_from_pretrained():
    model = HookedTransformer.from_pretrained("gpt2")
    assert model is not None


pytest.importorskip("torch")

from transformers import GPT2LMHeadModel  # noqa: E402


def test_from_pretrained_from_hf_model():
    hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
    model = HookedTransformer.from_pretrained("gpt2", hf_model=hf_model)
    assert model is not None
