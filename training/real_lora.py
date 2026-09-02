from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.lora import (
    adapter_hashes,
    inject_lora,
    load_adapter,
    save_adapter,
    trainable_parameter_receipt,
)
from training.lora_study import gradient_receipt, loss_receipt
from training.model_loading import load_training_model
from training.real_data import sha256_file

VARIANT = "backbone_depth_projection"
_STOP_REQUESTED = False


def _request_stop(_signum: int, _frame: object) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the selected real Breeze LoRA")
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--validation-examples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--stop-after-step", type=int)
    return parser.parse_args()


def atomic_write_json(path: Path, value: Any) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def load_cache(cache_root: Path) -> tuple[dict[str, Any], list[Path], list[Path]]:
    receipt_path = cache_root / "cache-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "deterministic_supervised_cache_complete":
        raise RuntimeError(f"cache is not complete: {receipt.get('status')}")
    train = [cache_root / row["artifact"] for row in receipt["splits"]["train"]]
    validation = [
        cache_root / row["artifact"] for row in receipt["splits"]["validation"]
    ]
    if not all(path.is_file() for path in train + validation):
        raise FileNotFoundError("cache receipt points to missing example artifacts")
    return receipt, train, validation


def example_order(length: int, *, epoch: int, seed: int) -> list[int]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)
    return torch.randperm(length, generator=generator).tolist()


def example_index(micro_step: int, length: int, *, seed: int) -> int:
    epoch, offset = divmod(micro_step, length)
    return example_order(length, epoch=epoch, seed=seed)[offset]


def move_example(path: Path, device: str) -> dict[str, torch.Tensor]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    return {name: tensor.to(device) for name, tensor in value.items()}


def lr_multiplier(step: int, *, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def validate(
    model: torch.nn.Module,
    paths: list[Path],
    *,
    device: str,
    limit: int,
) -> dict[str, float]:
    if limit <= 0 or limit > len(paths):
        raise ValueError("validation limit must fit the cached validation split")
    sums = {"total": 0.0, "backbone": 0.0, "depth_decoder": 0.0}
    model.eval()
    with torch.no_grad():
        for path in paths[:limit]:
            losses = loss_receipt(
                model(**move_example(path, device), use_cache=False, return_dict=True)
            )
            for name, value in losses.items():
                sums[name] += value
    model.train()
    return {name: value / limit for name, value in sums.items()}


def rng_state() -> dict[str, Any]:
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(path: Path) -> None:
    value = torch.load(path, map_location="cpu", weights_only=True)
    torch.set_rng_state(value["torch_cpu"])
    torch.cuda.set_rng_state_all(value["torch_cuda"])


def checkpoint_role_hashes(checkpoint: Path) -> dict[str, str]:
    return {
        name: sha256_file(checkpoint / name)
        for name in (
            "adapter.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "rng.pt",
            "trainer-state.json",
        )
    }


def save_checkpoint(
    *,
    output_root: Path,
    global_step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state: dict[str, Any],
) -> Path:
    checkpoint = output_root / f"checkpoint-step-{global_step:06d}"
    partial = checkpoint.with_name(checkpoint.name + ".partial")
    if checkpoint.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {checkpoint}")
    partial.mkdir(parents=True, exist_ok=False)
    hashes = save_adapter(model, partial / "adapter.safetensors")
    if hashes != adapter_hashes(model):
        raise RuntimeError("adapter changed while saving checkpoint")
    torch.save(optimizer.state_dict(), partial / "optimizer.pt")
    torch.save(scheduler.state_dict(), partial / "scheduler.pt")
    torch.save(rng_state(), partial / "rng.pt")
    atomic_write_json(partial / "trainer-state.json", state)
    roles = checkpoint_role_hashes(partial)
    atomic_write_json(
        partial / "checkpoint-receipt.json",
        {
            "schema_version": 1,
            "status": "five_role_checkpoint_complete",
            "global_step": global_step,
            "roles": roles,
            "adapter_tensor_hashes": hashes,
        },
    )
    partial.rename(checkpoint)
    atomic_write_json(
        output_root / "latest.json",
        {
            "checkpoint": str(checkpoint),
            "global_step": global_step,
            "checkpoint_receipt_sha256": sha256_file(
                checkpoint / "checkpoint-receipt.json"
            ),
        },
    )
    return checkpoint


def run_configuration(
    args: argparse.Namespace, cache_receipt: dict[str, Any]
) -> dict[str, Any]:
    return {
        "variant": VARIANT,
        "rank": args.rank,
        "alpha": args.alpha,
        "max_steps": args.max_steps,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "save_every": args.save_every,
        "validation_examples": args.validation_examples,
        "seed": args.seed,
        "max_gradient_norm": args.max_gradient_norm,
        "cache_receipt_sha256": sha256_file(args.cache_root / "cache-receipt.json"),
        "train_manifest_sha256": cache_receipt["source"]["train_manifest_sha256"],
        "validation_manifest_sha256": cache_receipt["source"][
            "validation_manifest_sha256"
        ],
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("real LoRA training requires an available CUDA device")
    if args.max_steps <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("step counts must be positive")
    if args.save_every <= 0 or args.max_steps % args.save_every:
        raise ValueError("save-every must divide max-steps")
    if (
        args.stop_after_step is not None
        and not 0 < args.stop_after_step < args.max_steps
    ):
        raise ValueError("stop-after-step must be inside the training interval")
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)

    cache_receipt, train_paths, validation_paths = load_cache(args.cache_root)
    configuration = run_configuration(args, cache_receipt)
    source_root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty_paths = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--short"], text=True
    ).splitlines()
    if dirty_paths:
        raise RuntimeError(f"source repository must be clean: {dirty_paths}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    config_path = args.output_root / "run-config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != configuration:
            raise RuntimeError("resume configuration differs from the original run")
    else:
        if args.resume_checkpoint is not None:
            raise FileNotFoundError("resume requested before run-config.json exists")
        atomic_write_json(config_path, configuration)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.set_device(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)

    model = load_training_model(args.model_root, device=args.device)
    model.gradient_checkpointing_enable()
    families = inject_lora(
        model,
        variant=VARIANT,
        rank=args.rank,
        alpha=args.alpha,
        seed=args.seed,
    )
    parameters = trainable_parameter_receipt(model, families)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        eps=1e-8,
        foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_multiplier(
            step, warmup_steps=args.warmup_steps, max_steps=args.max_steps
        ),
    )

    global_step = 0
    micro_step = 0
    history: list[dict[str, Any]] = []
    initial_validation = None
    if args.resume_checkpoint is not None:
        checkpoint_receipt = json.loads(
            (args.resume_checkpoint / "checkpoint-receipt.json").read_text()
        )
        actual_roles = checkpoint_role_hashes(args.resume_checkpoint)
        if actual_roles != checkpoint_receipt["roles"]:
            raise RuntimeError("resume checkpoint role hashes do not match")
        load_adapter(model, args.resume_checkpoint / "adapter.safetensors")
        optimizer.load_state_dict(
            torch.load(
                args.resume_checkpoint / "optimizer.pt",
                map_location=args.device,
                weights_only=True,
            )
        )
        scheduler.load_state_dict(
            torch.load(
                args.resume_checkpoint / "scheduler.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        state = json.loads((args.resume_checkpoint / "trainer-state.json").read_text())
        global_step = int(state["global_step"])
        micro_step = int(state["micro_step"])
        history = list(state["history"])
        initial_validation = state["initial_validation"]
        restore_rng_state(args.resume_checkpoint / "rng.pt")
    else:
        initial_validation = validate(
            model,
            validation_paths,
            device=args.device,
            limit=args.validation_examples,
        )

    first_gradients = None
    started_at = time.time()
    model.train()
    while global_step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        accumulated = {"total": 0.0, "backbone": 0.0, "depth_decoder": 0.0}
        for _ in range(args.gradient_accumulation):
            path = train_paths[
                example_index(micro_step, len(train_paths), seed=args.seed)
            ]
            outputs = model(
                **move_example(path, args.device), use_cache=False, return_dict=True
            )
            losses = loss_receipt(outputs)
            for name, value in losses.items():
                accumulated[name] += value / args.gradient_accumulation
            (outputs.loss / args.gradient_accumulation).backward()
            micro_step += 1
        if first_gradients is None:
            first_gradients = gradient_receipt(model, families)
            for family, row in first_gradients.items():
                if row["gradient_tensors"] != row["finite_gradient_tensors"]:
                    raise RuntimeError(f"non-finite first gradients in {family}: {row}")
                if row["nonzero_gradient_tensors"] == 0:
                    raise RuntimeError(f"zero first gradients in {family}: {row}")
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                (
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                max_norm=args.max_gradient_norm,
            )
            .detach()
            .float()
            .cpu()
            .item()
        )
        if not math.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite gradient norm at step {global_step + 1}")
        optimizer.step()
        scheduler.step()
        global_step += 1
        row: dict[str, Any] = {
            "step": global_step,
            "micro_step": micro_step,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "gradient_norm_before_clip": gradient_norm,
            "training_loss": accumulated,
        }

        should_save = (
            global_step % args.save_every == 0 or global_step == args.max_steps
        )
        should_pause = args.stop_after_step == global_step or _STOP_REQUESTED
        if should_save or should_pause:
            row["validation_loss"] = validate(
                model,
                validation_paths,
                device=args.device,
                limit=args.validation_examples,
            )
        history.append(row)
        if should_save or should_pause:
            status = "paused_for_fresh_process_resume" if should_pause else "running"
            if global_step == args.max_steps:
                status = "training_complete"
            state = {
                "schema_version": 1,
                "status": status,
                "global_step": global_step,
                "micro_step": micro_step,
                "initial_validation": initial_validation,
                "history": history,
                "source_revision": revision,
                "trainable_parameters": parameters,
                "target_families": families,
                "first_step_gradients": first_gradients,
                "runtime": {
                    "elapsed_seconds_this_process": time.time() - started_at,
                    "peak_cuda_memory_bytes": int(
                        torch.cuda.max_memory_allocated(args.device)
                    ),
                    "peak_cuda_reserved_bytes": int(
                        torch.cuda.max_memory_reserved(args.device)
                    ),
                    "python": sys.version,
                    "torch": torch.__version__,
                    "device": torch.cuda.get_device_name(args.device),
                },
            }
            checkpoint = save_checkpoint(
                output_root=args.output_root,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                state=state,
            )
            print(
                json.dumps(
                    {
                        "status": status,
                        "step": global_step,
                        "checkpoint": str(checkpoint),
                    }
                ),
                flush=True,
            )
            if should_pause and global_step < args.max_steps:
                return 75

    final_checkpoint = args.output_root / f"checkpoint-step-{args.max_steps:06d}"
    atomic_write_json(
        args.output_root / "training-receipt.json",
        {
            "schema_version": 1,
            "status": "real_lora_training_complete",
            "source": {
                "repository": str(source_root),
                "revision": revision,
                "model_root": str(args.model_root),
                "model_index_sha256": sha256_file(
                    args.model_root / "model.safetensors.index.json"
                ),
            },
            "configuration": configuration,
            "trainable_parameters": parameters,
            "target_families": families,
            "initial_validation": initial_validation,
            "final_validation": history[-1]["validation_loss"],
            "final_checkpoint": str(final_checkpoint),
            "final_checkpoint_receipt_sha256": sha256_file(
                final_checkpoint / "checkpoint-receipt.json"
            ),
            "interpretation_boundary": (
                "This run establishes one bounded multi-example optimization and "
                "held-out objective result. Loss is not a substitute for blind "
                "speaker identity, accent, cadence, pronunciation, monotony, or "
                "listening-fatigue judgments."
            ),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
