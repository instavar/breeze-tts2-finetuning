from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer

from models.breeze import BreezeForConditionalGeneration
from models.breeze_config import BreezeConfig


def configure_eager_attention(config):
    """Propagate eager attention through object and dictionary sub-configs."""

    config._attn_implementation = "eager"
    config.use_cache = False
    for nested_name in (
        "backbone_config",
        "depth_decoder_config",
        "text_encoder_config",
    ):
        nested = getattr(config, nested_name, None)
        if nested is None:
            continue
        if isinstance(nested, dict):
            nested["_attn_implementation"] = "eager"
            nested["preferred_attn_implementation"] = "eager"
            nested["use_cache"] = False
            continue
        nested._attn_implementation = "eager"
        if hasattr(nested, "preferred_attn_implementation"):
            nested.preferred_attn_implementation = "eager"
        if hasattr(nested, "use_cache"):
            nested.use_cache = False
    return config


def load_eager_config(model_root: str | Path) -> BreezeConfig:
    """Load Breeze while propagating eager attention through every nested model."""

    config = BreezeConfig.from_pretrained(model_root)
    configure_eager_attention(config)
    return config


def load_training_model(
    model_root: str | Path,
    *,
    device: str,
) -> BreezeForConditionalGeneration:
    config = load_eager_config(model_root)
    model = BreezeForConditionalGeneration.from_pretrained(
        model_root,
        config=config,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.config.use_cache = False
    return model


def load_tokenizer(model_root: str | Path):
    # Transformers 4.57.3 treats any local 4.57.3 model config as a possible
    # Mistral tokenizer. Breeze uses GemmaTokenizerFast with one whitespace
    # Split pre-tokenizer, so the Mistral sequence-index patch is inapplicable.
    return AutoTokenizer.from_pretrained(model_root, fix_mistral_regex=False)
