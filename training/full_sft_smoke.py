from __future__ import annotations

import argparse
import gc
import hashlib
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

from training.model_loading import (
    load_eager_config,
    load_tokenizer,
    load_training_model,
)
from training.supervised_example import build_supervised_example, example_summary

SENTINEL_PARAMETERS = {
    "backbone": "backbone_model.layers.0.self_attn.q_proj.weight",
    "depth_decoder": "depth_decoder.model.layers.0.self_attn.q_proj.weight",
    "text_projection": "text_encoder_proj.weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one Breeze full-SFT feasibility update"
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--speaker", default="S0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def family_for_name(name: str) -> str:
    if name.startswith("backbone_model."):
        return "backbone"
    if name.startswith("depth_decoder."):
        return "depth_decoder"
    if name.startswith(("text_encoder_proj.", "text_encoder_layer_projs.")):
        return "text_projection"
    if name.startswith("text_encoder."):
        return "text_encoder"
    if name.startswith("codec_model."):
        return "codec"
    return "other_synthesis"


def parameter_inventory(model: torch.nn.Module) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for name, parameter in model.named_parameters():
        family = family_for_name(name)
        row = result.setdefault(family, {"total": 0, "trainable": 0})
        row["total"] += parameter.numel()
        if parameter.requires_grad:
            row["trainable"] += parameter.numel()
    return result


def gradient_inventory(model: torch.nn.Module) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        family = family_for_name(name)
        row = result.setdefault(
            family,
            {
                "trainable_tensors": 0,
                "gradient_tensors": 0,
                "finite_gradient_tensors": 0,
                "nonzero_gradient_tensors": 0,
                "sum_squared_l2": 0.0,
                "max_abs": 0.0,
            },
        )
        row["trainable_tensors"] += 1
        gradient = parameter.grad
        if gradient is None:
            continue
        row["gradient_tensors"] += 1
        finite = bool(torch.isfinite(gradient).all().item())
        if finite:
            row["finite_gradient_tensors"] += 1
        nonzero = bool(torch.count_nonzero(gradient).item())
        if nonzero:
            row["nonzero_gradient_tensors"] += 1
        grad_float = gradient.detach().float()
        norm = float(torch.linalg.vector_norm(grad_float).item())
        row["sum_squared_l2"] += norm * norm
        row["max_abs"] = max(row["max_abs"], float(grad_float.abs().max().item()))
    for row in result.values():
        row["l2_norm"] = math.sqrt(row.pop("sum_squared_l2"))
    return result


def selected_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    parameters = dict(model.named_parameters())
    missing = [name for name in SENTINEL_PARAMETERS.values() if name not in parameters]
    if missing:
        raise KeyError(f"sentinel parameters missing: {missing}")
    return {
        family: parameters[name].detach().cpu().clone()
        for family, name in SENTINEL_PARAMETERS.items()
    }


def move_batch(
    example: dict[str, torch.Tensor], device: str
) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device) for name, tensor in example.items()}


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError(
            "the BF16 feasibility smoke requires an available CUDA device"
        )
    if not args.model_root.is_dir():
        raise FileNotFoundError(args.model_root)
    if not args.audio.is_file():
        raise FileNotFoundError(args.audio)
    if args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")

    checkpoint_dir = args.output_root / "checkpoint-step-1"
    checkpoint_partial = args.output_root / "checkpoint-step-1.partial"
    receipt_path = args.output_root / "full-sft-receipt.json"
    example_path = args.output_root / "example.pt"
    for collision in (checkpoint_dir, checkpoint_partial, receipt_path, example_path):
        if collision.exists():
            raise FileExistsError(
                f"refusing to overwrite existing artifact: {collision}"
            )
    args.output_root.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)

    source_root = Path(__file__).resolve().parents[1]
    source_revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    source_status = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--short"], text=True
    ).splitlines()

    config = load_eager_config(args.model_root)
    tokenizer = load_tokenizer(args.model_root)
    from qwen_tts import Qwen3TTSTokenizer

    audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
        str(args.model_root / "audio_tokenizer"), device_map=args.device
    )
    example = build_supervised_example(
        tokenizer,
        audio_tokenizer,
        config,
        audio_path=args.audio,
        transcript=args.transcript,
        speaker=args.speaker,
    )
    torch.save(example, example_path)
    prepared_summary = example_summary(example, config)
    del audio_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    model = load_training_model(args.model_root, device=args.device)
    model.gradient_checkpointing_enable()
    model.train()
    inventory = parameter_inventory(model)
    if inventory.get("backbone", {}).get("trainable", 0) == 0:
        raise RuntimeError("backbone has no trainable parameters")
    if inventory.get("depth_decoder", {}).get("trainable", 0) == 0:
        raise RuntimeError("depth decoder has no trainable parameters")

    batch = move_batch(example, args.device)
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        momentum=0.0,
        foreach=False,
    )
    optimizer.zero_grad(set_to_none=True)
    before = selected_tensors(model)

    started_at = time.time()
    outputs = model(**batch, use_cache=False, return_dict=True)
    loss = outputs.loss
    if loss is None or not bool(torch.isfinite(loss).item()):
        raise RuntimeError(f"non-finite or absent loss: {loss}")
    if outputs.backbone_loss is None or not bool(
        torch.isfinite(outputs.backbone_loss).item()
    ):
        raise RuntimeError(
            f"non-finite or absent backbone loss: {outputs.backbone_loss}"
        )
    if outputs.depth_decoder_loss is None or not bool(
        torch.isfinite(outputs.depth_decoder_loss).item()
    ):
        raise RuntimeError(
            f"non-finite or absent depth decoder loss: {outputs.depth_decoder_loss}"
        )
    loss.backward()
    gradients = gradient_inventory(model)
    for required_family in ("backbone", "depth_decoder"):
        row = gradients.get(required_family, {})
        if row.get("nonzero_gradient_tensors", 0) == 0:
            raise RuntimeError(f"{required_family} produced no nonzero gradients")
        if row.get("finite_gradient_tensors") != row.get("gradient_tensors"):
            raise RuntimeError(f"{required_family} produced non-finite gradients")

    optimizer.step()
    elapsed = time.time() - started_at
    after = selected_tensors(model)
    updates: dict[str, Any] = {}
    for family, before_tensor in before.items():
        after_tensor = after[family]
        changed = int(torch.count_nonzero(after_tensor.ne(before_tensor)).item())
        updates[family] = {
            "parameter": SENTINEL_PARAMETERS[family],
            "changed_elements": changed,
            "total_elements": before_tensor.numel(),
            "before_sha256": tensor_sha256(before_tensor),
            "after_sha256": tensor_sha256(after_tensor),
        }
    for required_family in ("backbone", "depth_decoder"):
        if updates[required_family]["changed_elements"] == 0:
            raise RuntimeError(
                f"{required_family} sentinel did not change after the SGD step; "
                "the BF16 update was not established"
            )

    checkpoint_partial.mkdir(parents=False, exist_ok=False)
    model.save_pretrained(
        checkpoint_partial,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer.save_pretrained(checkpoint_partial)
    checkpoint_partial.rename(checkpoint_dir)

    receipt = {
        "schema_version": 1,
        "status": "full_sft_step_complete_reload_pending",
        "scope": {
            "description": "one-example BF16 full-SFT feasibility smoke",
            "intentional_frozen_families": ["codec", "text_encoder"],
            "optimizer": "torch.optim.SGD",
            "optimizer_momentum": 0.0,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "source": {
            "repository": str(source_root),
            "revision": source_revision,
            "dirty_paths": source_status,
            "model_root": str(args.model_root),
            "model_index_sha256": sha256_file(
                args.model_root / "model.safetensors.index.json"
            ),
            "audio": str(args.audio),
            "audio_sha256": sha256_file(args.audio),
            "transcript": args.transcript,
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(args.device),
            "dtype": "bfloat16",
            "attention_implementation": "eager",
            "gradient_checkpointing": True,
            "elapsed_seconds_forward_backward_step": elapsed,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(args.device)),
            "peak_cuda_reserved_bytes": int(
                torch.cuda.max_memory_reserved(args.device)
            ),
        },
        "example": prepared_summary,
        "loss": {
            "total": float(loss.detach().float().cpu().item()),
            "backbone": float(outputs.backbone_loss.detach().float().cpu().item()),
            "depth_decoder": float(
                outputs.depth_decoder_loss.detach().float().cpu().item()
            ),
        },
        "parameters": inventory,
        "gradients": gradients,
        "updates": updates,
        "checkpoint": str(checkpoint_dir),
        "example_artifact": str(example_path),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
