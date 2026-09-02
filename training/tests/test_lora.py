from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from training.lora import (
    LoRALinear,
    inject_lora,
    load_adapter,
    merge_lora,
    save_adapter,
)


class Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(4, 4, bias=False)
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(4, 8, bias=False)


class ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone_model = nn.Module()
        self.backbone_model.layers = nn.ModuleList([Block()])
        self.depth_decoder = nn.Module()
        self.depth_decoder.model = nn.Module()
        self.depth_decoder.model.layers = nn.ModuleList([Block()])
        self.depth_decoder.model.inputs_embeds_projector = nn.Linear(4, 4, bias=False)
        self.text_encoder_proj = nn.Linear(4, 4, bias=False)
        self.lm_head = nn.Linear(4, 7, bias=False)
        self.config = SimpleNamespace()


def test_backbone_only_injects_no_other_family() -> None:
    model = ToyModel()
    families = inject_lora(model, variant="backbone_only", rank=2, alpha=4, seed=7)
    assert set(families.values()) == {"backbone"}
    module = model.backbone_model.layers[0].self_attn.q_proj
    assert isinstance(module, LoRALinear)
    assert module.in_features == module.base.in_features
    assert module.out_features == module.base.out_features
    assert module.weight is module.base.weight
    assert module.bias is module.base.bias
    assert module.lora_A.device == module.base.weight.device
    assert module.lora_B.device == module.base.weight.device
    assert isinstance(model.depth_decoder.model.layers[0].self_attn.q_proj, nn.Linear)
    assert all(
        parameter.requires_grad
        for module in model.modules()
        if isinstance(module, LoRALinear)
        for parameter in (module.lora_A, module.lora_B)
    )


def test_projection_variant_includes_all_requested_families() -> None:
    model = ToyModel()
    families = inject_lora(
        model,
        variant="backbone_depth_projection",
        rank=2,
        alpha=4,
        seed=7,
    )
    assert set(families.values()) == {
        "backbone",
        "depth_decoder",
        "depth_projection",
        "output_projection",
        "text_projection",
    }


def test_adapter_round_trip_and_merge_preserve_output(tmp_path) -> None:
    torch.manual_seed(3)
    inputs = torch.randn(2, 4)
    model = ToyModel()
    base_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    inject_lora(model, variant="backbone_only", rank=2, alpha=4, seed=7)
    module = model.backbone_model.layers[0].self_attn.q_proj
    with torch.no_grad():
        module.lora_B.fill_(0.25)
    expected = module(inputs)
    assert torch.allclose(
        expected,
        module(inputs_embeds=inputs, position_ids=torch.arange(4)),
        atol=1e-6,
    )

    adapter_path = tmp_path / "adapter.safetensors"
    hashes = save_adapter(model, adapter_path)
    reloaded = ToyModel()
    reloaded.load_state_dict(base_state)
    inject_lora(reloaded, variant="backbone_only", rank=2, alpha=4, seed=7)
    assert load_adapter(reloaded, adapter_path) == hashes
    reloaded_module = reloaded.backbone_model.layers[0].self_attn.q_proj
    assert torch.allclose(expected, reloaded_module(inputs), atol=1e-6)

    merged_count = merge_lora(reloaded)
    assert merged_count == 2
    assert torch.allclose(
        expected,
        reloaded.backbone_model.layers[0].self_attn.q_proj(inputs),
        atol=1e-6,
    )
