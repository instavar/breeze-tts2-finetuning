from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.lora import (
    VARIANTS,
    adapter_hashes,
    inject_lora,
    merge_lora,
    save_adapter,
    trainable_parameter_receipt,
)
from training.model_loading import load_training_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one matched Breeze LoRA variant")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--example", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def loss_receipt(outputs) -> dict[str, float]:
    losses = {
        "total": float(outputs.loss.detach().float().cpu().item()),
        "backbone": float(outputs.backbone_loss.detach().float().cpu().item()),
        "depth_decoder": float(
            outputs.depth_decoder_loss.detach().float().cpu().item()
        ),
    }
    if not all(math.isfinite(value) for value in losses.values()):
        raise RuntimeError(f"non-finite loss: {losses}")
    return losses


def forward_loss(model, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    with torch.no_grad():
        outputs = model(**batch, use_cache=False, return_dict=True)
    return loss_receipt(outputs)


def gradient_receipt(model, families: dict[str, str]) -> dict[str, dict[str, Any]]:
    modules = dict(model.named_modules())
    receipt: dict[str, dict[str, Any]] = {}
    for name, family in families.items():
        module = modules[name]
        row = receipt.setdefault(
            family,
            {
                "adapter_tensors": 0,
                "gradient_tensors": 0,
                "finite_gradient_tensors": 0,
                "nonzero_gradient_tensors": 0,
                "max_abs": 0.0,
            },
        )
        for parameter in (module.lora_A, module.lora_B):
            row["adapter_tensors"] += 1
            gradient = parameter.grad
            if gradient is None:
                continue
            row["gradient_tensors"] += 1
            finite = bool(torch.isfinite(gradient).all().item())
            if finite:
                row["finite_gradient_tensors"] += 1
            if bool(torch.count_nonzero(gradient).item()):
                row["nonzero_gradient_tensors"] += 1
            row["max_abs"] = max(
                row["max_abs"], float(gradient.detach().float().abs().max().item())
            )
    return receipt


def assert_gradient_receipt(receipt: dict[str, dict[str, Any]]) -> None:
    for family, row in receipt.items():
        if row["gradient_tensors"] != row["finite_gradient_tensors"]:
            raise RuntimeError(f"{family} produced non-finite LoRA gradients: {row}")
        if row["nonzero_gradient_tensors"] == 0:
            raise RuntimeError(f"{family} produced no nonzero LoRA gradients: {row}")


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("the LoRA study requires an available CUDA device")
    if args.steps <= 0 or args.learning_rate <= 0:
        raise ValueError("steps and learning rate must be positive")
    adapter_path = args.output_root / "adapter.safetensors"
    receipt_path = args.output_root / "receipt.json"
    for path in (adapter_path, receipt_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)

    source_root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty_paths = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--short"], text=True
    ).splitlines()

    example = torch.load(args.example, map_location="cpu", weights_only=True)
    batch = {name: tensor.to(args.device) for name, tensor in example.items()}
    model = load_training_model(args.model_root, device=args.device)
    families = inject_lora(
        model,
        variant=args.variant,
        rank=args.rank,
        alpha=args.alpha,
        seed=args.seed,
    )
    parameters = trainable_parameter_receipt(model, families)
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        momentum=0.0,
        foreach=False,
    )

    model.eval()
    initial = forward_loss(model, batch)
    history: list[dict[str, float]] = []
    first_gradients = None
    started_at = time.time()
    model.train()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch, use_cache=False, return_dict=True)
        step_loss = loss_receipt(outputs)
        outputs.loss.backward()
        if step == 0:
            first_gradients = gradient_receipt(model, families)
            assert_gradient_receipt(first_gradients)
        optimizer.step()
        history.append({"step": step + 1, **step_loss})

    model.eval()
    adapted = forward_loss(model, batch)
    hashes = save_adapter(model, adapter_path)
    if hashes != adapter_hashes(model):
        raise RuntimeError("saved adapter hashes changed unexpectedly")
    merged_count = merge_lora(model)
    if merged_count != len(families):
        raise RuntimeError(
            f"merged module count differs: expected={len(families)} actual={merged_count}"
        )
    merged = forward_loss(model, batch)
    for key in adapted:
        if abs(adapted[key] - merged[key]) > 0.02:
            raise RuntimeError(
                f"merged {key} loss differs from adapter path: "
                f"adapter={adapted[key]} merged={merged[key]}"
            )

    receipt = {
        "schema_version": 1,
        "status": "adapter_trained_saved_and_in_memory_merge_verified",
        "variant": args.variant,
        "source": {
            "repository": str(source_root),
            "revision": revision,
            "dirty_paths": dirty_paths,
            "model_root": str(args.model_root),
            "example": str(args.example),
        },
        "configuration": {
            "rank": args.rank,
            "alpha": args.alpha,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "optimizer": "torch.optim.SGD",
            "seed": args.seed,
            "dtype": "bfloat16",
            "attention_implementation": "eager",
        },
        "targets": {
            "module_count": len(families),
            "families": families,
            "trainable_parameters": parameters,
            "not_directly_adapted": [
                "depth_decoder.codebooks_head",
                "codec_model",
                "text_encoder",
            ],
        },
        "loss": {
            "initial": initial,
            "training_history": history,
            "adapted": adapted,
            "merged": merged,
            "change": {key: adapted[key] - initial[key] for key in initial},
        },
        "first_step_gradients": first_gradients,
        "adapter": {
            "path": str(adapter_path),
            "tensor_count": len(hashes),
            "hashes": hashes,
        },
        "runtime": {
            "elapsed_seconds": time.time() - started_at,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(args.device)),
            "peak_cuda_reserved_bytes": int(
                torch.cuda.max_memory_reserved(args.device)
            ),
            "python": sys.version,
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(args.device),
        },
        "interpretation_boundary": (
            "This one-example objective-fit smoke establishes technical adapter "
            "trainability only. It does not establish speaker identity, accent, "
            "prosody, generalisation, or perceptual sufficiency."
        ),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
