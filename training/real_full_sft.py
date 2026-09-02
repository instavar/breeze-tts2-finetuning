from __future__ import annotations

import argparse
import json
import math
import random
import signal
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch

from training.full_sft_smoke import (
    family_for_name,
    gradient_inventory,
    parameter_inventory,
)
from training.model_loading import load_tokenizer, load_training_model
from training.real_data import sha256_file
from training.real_lora import (
    atomic_write_json,
    example_index,
    load_cache,
    lr_multiplier,
    move_example,
    restore_rng_state,
    rng_state,
    validate,
)

_STOP_REQUESTED = False
TRAINABLE_FAMILIES = (
    "backbone",
    "depth_decoder",
    "text_projection",
    "other_synthesis",
)
INTENTIONALLY_FROZEN_FAMILIES = ("codec", "text_encoder")


def _request_stop(_signum: int, _frame: object) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Breeze on real data with full synthesis-model SFT"
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument(
        "--schedule-horizon-steps",
        type=int,
        help="Cosine schedule horizon; defaults to max-steps.",
    )
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--validation-examples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--stop-after-step", type=int)
    parser.add_argument(
        "--optimizer",
        choices=("fp32_master_sgd", "adafactor"),
        default="fp32_master_sgd",
    )
    parser.add_argument("--backbone-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--depth-decoder-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--text-projection-lr-multiplier", type=float, default=1.0)
    parser.add_argument("--other-synthesis-lr-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--save-policy",
        choices=("none", "model_only", "final", "interval"),
        default="interval",
    )
    parser.add_argument(
        "--validation-every",
        type=int,
        default=0,
        help="Validate every N steps independently of checkpoint saving; 0 disables it.",
    )
    return parser.parse_args()


def family_lr_multipliers(args: argparse.Namespace) -> dict[str, float]:
    return {
        "backbone": float(args.backbone_lr_multiplier),
        "depth_decoder": float(args.depth_decoder_lr_multiplier),
        "text_projection": float(args.text_projection_lr_multiplier),
        "other_synthesis": float(args.other_synthesis_lr_multiplier),
    }


def schedule_horizon_steps(args: argparse.Namespace) -> int:
    return int(args.schedule_horizon_steps or args.max_steps)


class FP32MasterSGD:
    """State-free SGD with FP32 master weights for a BF16 training model."""

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        weight_decay: float,
        lr_multipliers: dict[str, float] | None = None,
    ) -> None:
        if weight_decay < 0:
            raise ValueError("weight decay must be non-negative")
        self.weight_decay = float(weight_decay)
        self.lr_multipliers = dict(lr_multipliers or {})
        self.entries: list[
            tuple[str, str, torch.nn.Parameter, torch.Tensor]
        ] = []
        for name, parameter in named_parameters:
            if not parameter.requires_grad:
                continue
            family = family_for_name(name)
            master = parameter.detach().to(dtype=torch.float32, copy=True)
            self.entries.append((name, family, parameter, master))
        if not self.entries:
            raise ValueError("no trainable parameters were supplied")

    @torch.no_grad()
    def step(self, learning_rate: float) -> None:
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning rate must be positive and finite")
        decay = 1.0 - learning_rate * self.weight_decay
        if decay < 0:
            raise ValueError("learning rate and weight decay imply negative decay")
        for _name, family, parameter, master in self.entries:
            gradient = parameter.grad
            if gradient is None:
                continue
            effective_learning_rate = learning_rate * self.lr_multipliers.get(
                family, 1.0
            )
            if self.weight_decay:
                family_decay = 1.0 - effective_learning_rate * self.weight_decay
                if family_decay < 0:
                    raise ValueError(
                        "effective learning rate and weight decay imply negative decay"
                    )
                master.mul_(family_decay)
            master.add_(gradient, alpha=-effective_learning_rate)
            parameter.copy_(master)

    def state_dict_cpu(self) -> dict[str, torch.Tensor]:
        return {
            name: master.detach().cpu()
            for name, _family, _parameter, master in self.entries
        }

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        expected = [name for name, _family, _parameter, _master in self.entries]
        if set(state) != set(expected):
            missing = sorted(set(expected) - set(state))
            extra = sorted(set(state) - set(expected))
            raise RuntimeError(
                f"FP32 master state names differ: missing={missing} extra={extra}"
            )
        for name, _family, parameter, master in self.entries:
            value = state[name]
            if value.shape != master.shape:
                raise RuntimeError(
                    f"FP32 master shape differs for {name}: "
                    f"expected={tuple(master.shape)} actual={tuple(value.shape)}"
                )
            master.copy_(value.to(device=master.device, dtype=torch.float32))
            parameter.copy_(master)

    def receipt(self) -> dict[str, Any]:
        elements = sum(
            master.numel() for _name, _family, _parameter, master in self.entries
        )
        return {
            "implementation": "FP32MasterSGD",
            "momentum": 0.0,
            "weight_decay": self.weight_decay,
            "parameter_tensors": len(self.entries),
            "parameter_elements": elements,
            "state_bytes": elements * torch.tensor([], dtype=torch.float32).element_size(),
            "lr_multipliers": self.lr_multipliers,
        }


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


class AdafactorOptimizer:
    """PyTorch Adafactor with explicit Breeze family learning-rate groups."""

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
        *,
        weight_decay: float,
        lr_multipliers: dict[str, float],
    ) -> None:
        grouped: dict[str, list[torch.nn.Parameter]] = {
            family: [] for family in TRAINABLE_FAMILIES
        }
        for name, parameter in named_parameters:
            if parameter.requires_grad:
                grouped[family_for_name(name)].append(parameter)
        self.lr_multipliers = dict(lr_multipliers)
        parameter_groups = [
            {
                "params": parameters,
                "lr": self.lr_multipliers[family],
                "family": family,
                "lr_multiplier": self.lr_multipliers[family],
            }
            for family, parameters in grouped.items()
            if parameters
        ]
        self.optimizer = torch.optim.Adafactor(
            parameter_groups,
            lr=1.0,
            weight_decay=weight_decay,
            foreach=False,
        )

    def step(self, learning_rate: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate * float(group["lr_multiplier"])
        self.optimizer.step()

    def state_dict_cpu(self) -> dict[str, Any]:
        return _cpu_tree(self.optimizer.state_dict())

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state)

    def receipt(self) -> dict[str, Any]:
        state_bytes = 0
        for state in self.optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    state_bytes += value.numel() * value.element_size()
        return {
            "implementation": "torch.optim.Adafactor",
            "weight_decay": self.optimizer.defaults["weight_decay"],
            "state_bytes": state_bytes,
            "lr_multipliers": self.lr_multipliers,
        }


def build_optimizer(
    args: argparse.Namespace, model: torch.nn.Module
) -> FP32MasterSGD | AdafactorOptimizer:
    multipliers = family_lr_multipliers(args)
    if args.optimizer == "fp32_master_sgd":
        return FP32MasterSGD(
            model.named_parameters(),
            weight_decay=args.weight_decay,
            lr_multipliers=multipliers,
        )
    if args.optimizer == "adafactor":
        return AdafactorOptimizer(
            model.named_parameters(),
            weight_decay=args.weight_decay,
            lr_multipliers=multipliers,
        )
    raise ValueError(f"unsupported optimizer: {args.optimizer}")


def regular_file_manifest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def checkpoint_role_hashes(checkpoint: Path) -> dict[str, str]:
    return {
        name: sha256_file(checkpoint / name)
        for name in (
            "model-role.json",
            "optimizer.pt",
            "scheduler.pt",
            "rng.pt",
            "trainer-state.json",
        )
    }


def verify_model_role(checkpoint: Path) -> None:
    role = json.loads((checkpoint / "model-role.json").read_text())
    actual = {
        name: sha256_file(checkpoint / name)
        for name in role["files"]
    }
    if actual != role["files"]:
        raise RuntimeError("checkpoint model files differ from model role manifest")


def save_checkpoint(
    *,
    output_root: Path,
    global_step: int,
    model: torch.nn.Module,
    optimizer: FP32MasterSGD | AdafactorOptimizer,
    scheduler_state: dict[str, Any],
    trainer_state: dict[str, Any],
    tokenizer_root: Path,
) -> Path:
    checkpoint = output_root / f"checkpoint-step-{global_step:06d}"
    partial = checkpoint.with_name(checkpoint.name + ".partial")
    if checkpoint.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {checkpoint}")
    partial.mkdir(parents=True, exist_ok=False)

    model.save_pretrained(partial, safe_serialization=True, max_shard_size="4GB")
    load_tokenizer(tokenizer_root).save_pretrained(partial)
    atomic_write_json(
        partial / "model-role.json",
        {
            "schema_version": 1,
            "status": "full_model_role_complete",
            "files": regular_file_manifest(partial),
        },
    )
    torch.save(optimizer.state_dict_cpu(), partial / "optimizer.pt")
    torch.save(scheduler_state, partial / "scheduler.pt")
    torch.save(rng_state(), partial / "rng.pt")
    atomic_write_json(partial / "trainer-state.json", trainer_state)
    roles = checkpoint_role_hashes(partial)
    atomic_write_json(
        partial / "checkpoint-receipt.json",
        {
            "schema_version": 1,
            "status": "five_role_full_sft_checkpoint_complete",
            "global_step": global_step,
            "roles": roles,
            "model_files": json.loads((partial / "model-role.json").read_text())["files"],
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


def save_model_export(
    *,
    output_root: Path,
    global_step: int,
    model: torch.nn.Module,
    tokenizer_root: Path,
) -> Path:
    export = output_root / f"model-step-{global_step:06d}"
    partial = export.with_name(export.name + ".partial")
    if export.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite model export: {export}")
    partial.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(partial, safe_serialization=True, max_shard_size="4GB")
    load_tokenizer(tokenizer_root).save_pretrained(partial)
    atomic_write_json(
        partial / "model-export-receipt.json",
        {
            "schema_version": 1,
            "status": "screening_model_export_complete",
            "global_step": global_step,
            "files": regular_file_manifest(partial),
            "boundary": (
                "This export is suitable for fresh inference and validation. It does "
                "not contain optimizer, scheduler, trainer, or RNG state and cannot "
                "establish exact training resume."
            ),
        },
    )
    partial.rename(export)
    return export


def run_configuration(
    args: argparse.Namespace, cache_receipt: dict[str, Any]
) -> dict[str, Any]:
    return {
        "adaptation": "full_sft",
        "trainable_scope": "all released synthesis parameters",
        "intentional_frozen_families": list(INTENTIONALLY_FROZEN_FAMILIES),
        "optimizer": args.optimizer,
        "max_steps": args.max_steps,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "schedule_horizon_steps": schedule_horizon_steps(args),
        "save_every": args.save_every,
        "validation_examples": args.validation_examples,
        "seed": args.seed,
        "max_gradient_norm": args.max_gradient_norm,
        "lr_multipliers": family_lr_multipliers(args),
        "save_policy": args.save_policy,
        "validation_every": args.validation_every,
        "cache_receipt_sha256": sha256_file(args.cache_root / "cache-receipt.json"),
        "train_manifest_sha256": cache_receipt["source"]["train_manifest_sha256"],
        "validation_manifest_sha256": cache_receipt["source"][
            "validation_manifest_sha256"
        ],
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("real full SFT requires an available CUDA device")
    if args.max_steps <= 0 or args.gradient_accumulation <= 0:
        raise ValueError("step counts must be positive")
    if args.save_every <= 0:
        raise ValueError("save-every must be positive")
    if args.save_policy == "interval" and args.max_steps % args.save_every:
        raise ValueError("save-every must divide max-steps for interval saving")
    if args.validation_every < 0:
        raise ValueError("validation-every must be non-negative")
    schedule_horizon = schedule_horizon_steps(args)
    if schedule_horizon < args.max_steps:
        raise ValueError("schedule-horizon-steps must be at least max-steps")
    if not 0 <= args.warmup_steps < schedule_horizon:
        raise ValueError("warmup-steps must be inside the schedule horizon")
    if args.learning_rate <= 0 or args.max_gradient_norm <= 0:
        raise ValueError("learning rate and gradient norm must be positive")
    multipliers = family_lr_multipliers(args)
    if any(not math.isfinite(value) or value <= 0 for value in multipliers.values()):
        raise ValueError("all family learning-rate multipliers must be positive")
    if args.stop_after_step is not None and args.save_policy not in {
        "final",
        "interval",
    }:
        raise ValueError("stop-after-step requires final or interval checkpoint saving")
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

    model_source = args.resume_checkpoint or args.model_root
    if args.resume_checkpoint is not None:
        receipt = json.loads(
            (args.resume_checkpoint / "checkpoint-receipt.json").read_text()
        )
        actual_roles = checkpoint_role_hashes(args.resume_checkpoint)
        if actual_roles != receipt["roles"]:
            raise RuntimeError("resume checkpoint role hashes do not match")
        verify_model_role(args.resume_checkpoint)

    model = load_training_model(model_source, device=args.device)
    model.gradient_checkpointing_enable()
    inventory = parameter_inventory(model)
    for family in TRAINABLE_FAMILIES:
        row = inventory.get(family, {})
        if row.get("trainable", 0) == 0:
            raise RuntimeError(f"required full-SFT family is not trainable: {family}")
    for family in INTENTIONALLY_FROZEN_FAMILIES:
        if inventory.get(family, {}).get("trainable", 0) != 0:
            raise RuntimeError(f"expected frozen family became trainable: {family}")

    optimizer = build_optimizer(args, model)
    global_step = 0
    micro_step = 0
    history: list[dict[str, Any]] = []
    initial_validation = None
    if args.resume_checkpoint is not None:
        master_state = torch.load(
            args.resume_checkpoint / "optimizer.pt",
            map_location="cpu",
            weights_only=True,
        )
        optimizer.load_state_dict(master_state)
        del master_state
        scheduler_state = torch.load(
            args.resume_checkpoint / "scheduler.pt",
            map_location="cpu",
            weights_only=True,
        )
        state = json.loads((args.resume_checkpoint / "trainer-state.json").read_text())
        if state["source_revision"] != revision:
            raise RuntimeError("source revision differs from checkpoint")
        global_step = int(state["global_step"])
        micro_step = int(state["micro_step"])
        history = list(state["history"])
        initial_validation = state["initial_validation"]
        if scheduler_state["last_completed_step"] != global_step:
            raise RuntimeError("scheduler state differs from trainer step")
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
        model.zero_grad(set_to_none=True)
        accumulated = {"total": 0.0, "backbone": 0.0, "depth_decoder": 0.0}
        for _ in range(args.gradient_accumulation):
            path = train_paths[
                example_index(micro_step, len(train_paths), seed=args.seed)
            ]
            outputs = model(
                **move_example(path, args.device), use_cache=False, return_dict=True
            )
            losses = {
                "total": float(outputs.loss.detach().float().cpu().item()),
                "backbone": float(
                    outputs.backbone_loss.detach().float().cpu().item()
                ),
                "depth_decoder": float(
                    outputs.depth_decoder_loss.detach().float().cpu().item()
                ),
            }
            if not all(math.isfinite(value) for value in losses.values()):
                raise RuntimeError(f"non-finite training loss: {losses}")
            for name, value in losses.items():
                accumulated[name] += value / args.gradient_accumulation
            (outputs.loss / args.gradient_accumulation).backward()
            micro_step += 1

        if first_gradients is None:
            first_gradients = gradient_inventory(model)
            for family in TRAINABLE_FAMILIES:
                row = first_gradients.get(family, {})
                if row.get("gradient_tensors") != row.get(
                    "finite_gradient_tensors"
                ):
                    raise RuntimeError(
                        f"non-finite first gradients in {family}: {row}"
                    )
                if row.get("nonzero_gradient_tensors", 0) == 0:
                    raise RuntimeError(f"zero first gradients in {family}: {row}")

        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                max_norm=args.max_gradient_norm,
            )
            .detach()
            .float()
            .cpu()
            .item()
        )
        if not math.isfinite(gradient_norm):
            raise RuntimeError(f"non-finite gradient norm at step {global_step + 1}")
        learning_rate = args.learning_rate * lr_multiplier(
            global_step,
            warmup_steps=args.warmup_steps,
            max_steps=schedule_horizon,
        )
        optimizer.step(learning_rate)
        global_step += 1
        row: dict[str, Any] = {
            "step": global_step,
            "micro_step": micro_step,
            "learning_rate": learning_rate,
            "gradient_norm_before_clip": gradient_norm,
            "training_loss": accumulated,
        }

        should_save = args.save_policy in {"final", "interval"} and (
            (args.save_policy == "interval" and global_step % args.save_every == 0)
            or global_step == args.max_steps
        )
        should_pause = args.stop_after_step == global_step or _STOP_REQUESTED
        should_validate = (
            should_save
            or should_pause
            or global_step == args.max_steps
            or (
                args.validation_every > 0
                and global_step % args.validation_every == 0
            )
        )
        if should_validate:
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
                "parameters": inventory,
                "optimizer": optimizer.receipt(),
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
                scheduler_state={
                    "schema_version": 1,
                    "last_completed_step": global_step,
                    "base_learning_rate": args.learning_rate,
                    "warmup_steps": args.warmup_steps,
                    "max_steps": schedule_horizon,
                    "training_max_steps": args.max_steps,
                },
                trainer_state=state,
                tokenizer_root=args.model_root,
            )
            print(
                json.dumps(
                    {"status": status, "step": global_step, "checkpoint": str(checkpoint)}
                ),
                flush=True,
            )
            if should_pause and global_step < args.max_steps:
                return 75

    final_checkpoint = (
        args.output_root / f"checkpoint-step-{args.max_steps:06d}"
        if args.save_policy in {"final", "interval"}
        else None
    )
    final_model_export = (
        save_model_export(
            output_root=args.output_root,
            global_step=args.max_steps,
            model=model,
            tokenizer_root=args.model_root,
        )
        if args.save_policy == "model_only"
        else None
    )
    final_validation = next(
        row["validation_loss"]
        for row in reversed(history)
        if "validation_loss" in row
    )
    final_checkpoint_receipt_sha256 = (
        sha256_file(final_checkpoint / "checkpoint-receipt.json")
        if final_checkpoint is not None
        else None
    )
    atomic_write_json(
        args.output_root / "training-receipt.json",
        {
            "schema_version": 1,
            "status": "real_multi_example_full_sft_complete",
            "source": {
                "repository": str(source_root),
                "revision": revision,
                "model_root": str(args.model_root),
                "model_index_sha256": sha256_file(
                    args.model_root / "model.safetensors.index.json"
                ),
            },
            "configuration": configuration,
            "parameters": inventory,
            "optimizer": optimizer.receipt(),
            "initial_validation": initial_validation,
            "history": history,
            "first_step_gradients": first_gradients,
            "final_validation": final_validation,
            "final_checkpoint": str(final_checkpoint) if final_checkpoint else None,
            "final_checkpoint_receipt_sha256": final_checkpoint_receipt_sha256,
            "final_model_export": (
                str(final_model_export) if final_model_export else None
            ),
            "final_model_export_receipt_sha256": (
                sha256_file(final_model_export / "model-export-receipt.json")
                if final_model_export
                else None
            ),
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
            "interpretation_boundary": (
                "This run establishes one bounded multi-example full-SFT optimization "
                "and held-out objective result. It does not establish speaker identity, "
                "accent, cadence, pronunciation, monotony, listening fatigue, or "
                "production fitness."
            ),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
