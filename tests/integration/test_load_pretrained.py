from xlens.pretrained import get_pretrained_model_config, get_pretrained_state_dict


def test_get_pretrained_state_dict():
    cfg = get_pretrained_model_config("gpt2")
    state_dict = get_pretrained_state_dict("gpt2", cfg)
    assert "unembed.W_U" in state_dict
