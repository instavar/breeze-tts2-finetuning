from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

TRANSFORMER_LINEAR_SUFFIXES = (
    ".self_attn.q_proj",
    ".self_attn.k_proj",
    ".self_attn.v_proj",
    ".self_attn.o_proj",
    ".mlp.gate_proj",
    ".mlp.up_proj",
    ".mlp.down_proj",
)
PROJECTION_TARGETS = {
    "text_encoder_proj": "text_projection",
    "depth_decoder.model.inputs_embeds_projector": "depth_projection",
    "lm_head": "output_projection",
}
VARIANTS = (
    "backbone_only",
    "backbone_depth",
    "backbone_depth_projection",
)


class LoRALinear(nn.Module):
    """Small dependency-free LoRA wrapper for a frozen linear layer."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        seed: int,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        bound = 1.0 / math.sqrt(base.in_features)
        lora_a = torch.empty((rank, base.in_features), dtype=torch.float32)
        lora_a.uniform_(-bound, bound, generator=generator)
        self.lora_A = nn.Parameter(
            lora_a.to(device=base.weight.device, dtype=base.weight.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                (base.out_features, rank),
                dtype=base.weight.dtype,
                device=base.weight.device,
            )
        )

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base.bias

    def forward(
        self,
        inputs: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del position_ids
        if inputs is None:
            inputs = inputs_embeds
        elif inputs_embeds is not None:
            raise ValueError("provide inputs or inputs_embeds, not both")
        if inputs is None:
            raise ValueError("LoRALinear requires inputs or inputs_embeds")
        update = torch.nn.functional.linear(inputs, self.lora_A)
        update = torch.nn.functional.linear(update, self.lora_B)
        return self.base(inputs) + update * self.scaling

    def merged_linear(self) -> nn.Linear:
        merged = self.base
        with torch.no_grad():
            delta = torch.matmul(self.lora_B, self.lora_A) * self.scaling
            merged.weight.add_(delta.to(dtype=merged.weight.dtype))
        return merged


def target_family(module_name: str, variant: str) -> str | None:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported LoRA variant: {variant}")
    if module_name.startswith("backbone_model.layers.") and module_name.endswith(
        TRANSFORMER_LINEAR_SUFFIXES
    ):
        return "backbone"
    if (
        variant in ("backbone_depth", "backbone_depth_projection")
        and module_name.startswith("depth_decoder.model.layers.")
        and module_name.endswith(TRANSFORMER_LINEAR_SUFFIXES)
    ):
        return "depth_decoder"
    if variant == "backbone_depth_projection":
        return PROJECTION_TARGETS.get(module_name)
    return None


def _parent_and_child(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    *,
    variant: str,
    rank: int,
    alpha: float,
    seed: int,
) -> dict[str, str]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    targets: list[tuple[str, nn.Linear, str]] = []
    for name, module in model.named_modules():
        family = target_family(name, variant)
        if family is not None:
            if not isinstance(module, nn.Linear):
                raise TypeError(f"LoRA target is not linear: {name}")
            targets.append((name, module, family))
    if not targets:
        raise RuntimeError(f"no LoRA targets found for {variant}")

    families: dict[str, str] = {}
    for index, (name, module, family) in enumerate(targets):
        parent, child = _parent_and_child(model, name)
        setattr(
            parent,
            child,
            LoRALinear(
                module,
                rank=rank,
                alpha=alpha,
                seed=seed + index,
            ),
        )
        families[name] = family
    return families


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        state[f"{name}.lora_A"] = module.lora_A.detach().cpu().contiguous()
        state[f"{name}.lora_B"] = module.lora_B.detach().cpu().contiguous()
    if not state:
        raise RuntimeError("model has no LoRA adapter state")
    return state


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def adapter_hashes(model: nn.Module) -> dict[str, str]:
    return {name: tensor_sha256(value) for name, value in adapter_state(model).items()}


def save_adapter(model: nn.Module, path: str | Path) -> dict[str, str]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = adapter_state(model)
    save_file(state, path)
    return {name: tensor_sha256(value) for name, value in state.items()}


def load_adapter(model: nn.Module, path: str | Path) -> dict[str, str]:
    expected = adapter_state(model)
    loaded = load_file(path)
    if set(loaded) != set(expected):
        missing = sorted(set(expected) - set(loaded))
        unexpected = sorted(set(loaded) - set(expected))
        raise ValueError(
            f"adapter keys differ; missing={missing} unexpected={unexpected}"
        )
    modules = dict(model.named_modules())
    with torch.no_grad():
        for key, value in loaded.items():
            module_name, parameter_name = key.rsplit(".", 1)
            module = modules[module_name]
            if not isinstance(module, LoRALinear):
                raise TypeError(f"adapter target is not LoRALinear: {module_name}")
            target = getattr(module, parameter_name)
            target.copy_(value.to(device=target.device, dtype=target.dtype))
    return adapter_hashes(model)


def merge_lora(model: nn.Module) -> int:
    names = [
        name for name, module in model.named_modules() if isinstance(module, LoRALinear)
    ]
    for name in names:
        parent, child = _parent_and_child(model, name)
        module = getattr(parent, child)
        setattr(parent, child, module.merged_linear())
    return len(names)


def trainable_parameter_receipt(
    model: nn.Module, families: dict[str, str]
) -> dict[str, Any]:
    modules = dict(model.named_modules())
    by_family: dict[str, dict[str, int]] = {}
    for name, family in families.items():
        module = modules[name]
        row = by_family.setdefault(family, {"modules": 0, "parameters": 0})
        row["modules"] += 1
        row["parameters"] += sum(
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        )
    total = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "by_family": by_family}
