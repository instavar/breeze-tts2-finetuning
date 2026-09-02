from types import SimpleNamespace

from training.model_loading import configure_eager_attention


def test_configure_eager_attention_handles_mixed_nested_config_shapes():
    backbone = {"model_type": "qwen3"}
    depth_decoder = SimpleNamespace(use_cache=True)
    text_encoder = SimpleNamespace(preferred_attn_implementation="flash_attention_2")
    config = SimpleNamespace(
        use_cache=True,
        backbone_config=backbone,
        depth_decoder_config=depth_decoder,
        text_encoder_config=text_encoder,
    )

    configure_eager_attention(config)

    assert config._attn_implementation == "eager"
    assert config.use_cache is False
    assert backbone["_attn_implementation"] == "eager"
    assert backbone["preferred_attn_implementation"] == "eager"
    assert backbone["use_cache"] is False
    assert depth_decoder._attn_implementation == "eager"
    assert depth_decoder.use_cache is False
    assert text_encoder._attn_implementation == "eager"
    assert text_encoder.preferred_attn_implementation == "eager"
