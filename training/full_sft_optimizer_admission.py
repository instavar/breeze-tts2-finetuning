from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from training.full_sft_smoke import gradient_inventory, parameter_inventory
from training.model_loading import load_training_model
from training.real_data import sha256_file
from training.real_full_sft import (
    build_optimizer,
    checkpoint_role_hashes,
    verify_model_role,
)
from training.real_lora import (
    example_index,
    load_cache,
    lr_multiplier,
    move_example,
    validate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reload a full-SFT optimizer and perform one non-persistent update"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--validation-examples", type=int, default=8)
    return parser.parse_args()


def optimizer_args(configuration: dict[str, Any]) -> SimpleNamespace:
    multipliers = configuration["lr_multipliers"]
    return SimpleNamespace(
        optimizer=configuration["optimizer"],
        weight_decay=float(configuration["weight_decay"]),
        backbone_lr_multiplier=float(multipliers["backbone"]),
        depth_decoder_lr_multiplier=float(multipliers["depth_decoder"]),
        text_projection_lr_multiplier=float(multipliers["text_projection"]),
        other_synthesis_lr_multiplier=float(multipliers["other_synthesis"]),
    )


def main() -> int:
    args = parse_args()
    if args.output_receipt.exists():
        raise FileExistsError(args.output_receipt)
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("optimizer admission requires an available CUDA device")
    receipt = json.loads((args.checkpoint / "checkpoint-receipt.json").read_text())
    if checkpoint_role_hashes(args.checkpoint) != receipt["roles"]:
        raise RuntimeError("checkpoint role hashes do not match")
    verify_model_role(args.checkpoint)
    training_root = args.checkpoint.parent
    configuration = json.loads((training_root / "run-config.json").read_text())
    trainer_state = json.loads((args.checkpoint / "trainer-state.json").read_text())
    _cache_receipt, train_paths, validation_paths = load_cache(args.cache_root)

    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    model = load_training_model(args.checkpoint, device=args.device)
    model.gradient_checkpointing_enable()
    optimizer = build_optimizer(optimizer_args(configuration), model)
    optimizer_state = torch.load(
        args.checkpoint / "optimizer.pt", map_location="cpu", weights_only=True
    )
    optimizer.load_state_dict(optimizer_state)
    del optimizer_state

    before = validate(
        model,
        validation_paths,
        device=args.device,
        limit=args.validation_examples,
    )
    model.train()
    model.zero_grad(set_to_none=True)
    micro_step = int(trainer_state["micro_step"])
    path = train_paths[example_index(micro_step, len(train_paths), seed=configuration["seed"])]
    started_at = time.time()
    outputs = model(**move_example(path, args.device), use_cache=False, return_dict=True)
    if outputs.loss is None or not bool(torch.isfinite(outputs.loss).item()):
        raise RuntimeError(f"non-finite continuation loss: {outputs.loss}")
    outputs.loss.backward()
    gradients = gradient_inventory(model)
    if any(
        row["gradient_tensors"] != row["finite_gradient_tensors"]
        or row["nonzero_gradient_tensors"] == 0
        for row in gradients.values()
    ):
        raise RuntimeError(f"invalid continuation gradients: {gradients}")
    gradient_norm = float(
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            max_norm=float(configuration["max_gradient_norm"]),
        )
        .detach()
        .float()
        .cpu()
        .item()
    )
    if not math.isfinite(gradient_norm):
        raise RuntimeError("non-finite continuation gradient norm")
    learning_rate = float(configuration["learning_rate"]) * lr_multiplier(
        int(trainer_state["global_step"]),
        warmup_steps=int(configuration["warmup_steps"]),
        max_steps=max(
            int(
                configuration.get(
                    "schedule_horizon_steps", configuration["max_steps"]
                )
            ),
            int(trainer_state["global_step"]) + 1,
        ),
    )
    optimizer.step(learning_rate)
    after = validate(
        model,
        validation_paths,
        device=args.device,
        limit=args.validation_examples,
    )
    output = {
        "schema_version": 1,
        "status": "fresh_optimizer_reload_and_update_complete",
        "checkpoint": str(args.checkpoint),
        "checkpoint_receipt_sha256": sha256_file(
            args.checkpoint / "checkpoint-receipt.json"
        ),
        "optimizer": optimizer.receipt(),
        "parameters": parameter_inventory(model),
        "validation_before": before,
        "continuation_training_loss": {
            "total": float(outputs.loss.detach().float().cpu().item()),
            "backbone": float(outputs.backbone_loss.detach().float().cpu().item()),
            "depth_decoder": float(
                outputs.depth_decoder_loss.detach().float().cpu().item()
            ),
        },
        "continuation_learning_rate": learning_rate,
        "continuation_gradient_norm_before_clip": gradient_norm,
        "continuation_gradients": gradients,
        "validation_after": after,
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
        "boundary": (
            "This proves fresh-process optimizer-state reload and one additional "
            "non-persistent update. It is not a declared training resume because the "
            "source run had already reached its configured terminal step."
        ),
    }
    args.output_receipt.parent.mkdir(parents=True, exist_ok=True)
    args.output_receipt.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": output["status"], "receipt": str(args.output_receipt)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
