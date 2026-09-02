from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from training.real_lora import atomic_write_json

TRIAL_NAME = re.compile(r"^[a-z0-9][a-z0-9-]*$")
OPTIMIZERS = {"fp32_master_sgd", "adafactor"}
SAVE_POLICIES = {"none", "model_only", "final", "interval"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a frozen sequential Breeze full-SFT hyperparameter plan"
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def validate_plan(plan: dict[str, Any]) -> list[dict[str, Any]]:
    if plan.get("schema_version") != 1:
        raise ValueError("sweep plan schema_version must be 1")
    defaults = plan.get("defaults", {})
    trials = plan.get("trials")
    if not isinstance(defaults, dict) or not isinstance(trials, list) or not trials:
        raise ValueError("sweep plan requires defaults and a non-empty trials list")
    resolved: list[dict[str, Any]] = []
    names: set[str] = set()
    for raw_trial in trials:
        if not isinstance(raw_trial, dict):
            raise TypeError("every sweep trial must be an object")
        trial = {**defaults, **raw_trial}
        name = trial.get("name")
        if not isinstance(name, str) or not TRIAL_NAME.fullmatch(name):
            raise ValueError(f"invalid trial name: {name!r}")
        if name in names:
            raise ValueError(f"duplicate trial name: {name}")
        names.add(name)
        if trial.get("optimizer") not in OPTIMIZERS:
            raise ValueError(f"invalid optimizer for {name}: {trial.get('optimizer')}")
        if trial.get("save_policy", "none") not in SAVE_POLICIES:
            raise ValueError(f"invalid save policy for {name}")
        for key in (
            "max_steps",
            "gradient_accumulation",
            "learning_rate",
            "warmup_steps",
            "validation_examples",
            "max_gradient_norm",
            "backbone_lr_multiplier",
            "depth_decoder_lr_multiplier",
            "text_projection_lr_multiplier",
            "other_synthesis_lr_multiplier",
        ):
            if key not in trial:
                raise ValueError(f"trial {name} is missing {key}")
        if int(trial["max_steps"]) <= 0 or int(trial["gradient_accumulation"]) <= 0:
            raise ValueError(f"trial {name} has non-positive step settings")
        if not 0 <= int(trial["warmup_steps"]) < int(trial["max_steps"]):
            raise ValueError(f"trial {name} has invalid warmup_steps")
        horizon = int(trial.get("schedule_horizon_steps", trial["max_steps"]))
        if horizon < int(trial["max_steps"]):
            raise ValueError(f"trial {name} has a schedule horizon below max_steps")
        if not 0 <= int(trial["warmup_steps"]) < horizon:
            raise ValueError(f"trial {name} has warmup outside its schedule horizon")
        for key in (
            "learning_rate",
            "max_gradient_norm",
            "backbone_lr_multiplier",
            "depth_decoder_lr_multiplier",
            "text_projection_lr_multiplier",
            "other_synthesis_lr_multiplier",
        ):
            if float(trial[key]) <= 0:
                raise ValueError(f"trial {name} has non-positive {key}")
        resolved.append(trial)
    return resolved


def command_for_trial(
    trial: dict[str, Any],
    *,
    model_root: Path,
    cache_root: Path,
    output_root: Path,
    device: str,
) -> list[str]:
    trial_root = output_root / "trials" / trial["name"]
    command = [
        sys.executable,
        "-m",
        "training.real_full_sft",
        "--model-root",
        str(model_root),
        "--cache-root",
        str(cache_root),
        "--output-root",
        str(trial_root),
        "--device",
        device,
    ]
    options = {
        "max-steps": trial["max_steps"],
        "gradient-accumulation": trial["gradient_accumulation"],
        "learning-rate": trial["learning_rate"],
        "weight-decay": trial.get("weight_decay", 0.01),
        "warmup-steps": trial["warmup_steps"],
        "save-every": trial.get("save_every", trial["max_steps"]),
        "validation-examples": trial["validation_examples"],
        "seed": trial.get("seed", 42),
        "max-gradient-norm": trial["max_gradient_norm"],
        "optimizer": trial["optimizer"],
        "backbone-lr-multiplier": trial["backbone_lr_multiplier"],
        "depth-decoder-lr-multiplier": trial["depth_decoder_lr_multiplier"],
        "text-projection-lr-multiplier": trial["text_projection_lr_multiplier"],
        "other-synthesis-lr-multiplier": trial["other_synthesis_lr_multiplier"],
        "save-policy": trial.get("save_policy", "none"),
        "validation-every": trial.get("validation_every", trial["max_steps"]),
    }
    if "schedule_horizon_steps" in trial:
        options["schedule-horizon-steps"] = trial["schedule_horizon_steps"]
    for name, value in options.items():
        command.extend((f"--{name}", str(value)))
    return command


def summary_for_trial(trial: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    initial = receipt["initial_validation"]
    final = receipt["final_validation"]
    return {
        "name": trial["name"],
        "configuration": receipt["configuration"],
        "optimizer": receipt["optimizer"],
        "initial_validation": initial,
        "final_validation": final,
        "validation_delta": {
            key: float(final[key]) - float(initial[key])
            for key in ("total", "backbone", "depth_decoder")
        },
        "runtime": receipt["runtime"],
    }


def main() -> int:
    args = parse_args()
    plan = json.loads(args.plan.read_text())
    trials = validate_plan(plan)
    receipt_path = args.output_root / "sweep-receipt.json"
    if receipt_path.exists():
        raise FileExistsError(receipt_path)
    args.output_root.mkdir(parents=True, exist_ok=True)
    frozen_plan = args.output_root / "plan.json"
    if frozen_plan.exists():
        if json.loads(frozen_plan.read_text()) != plan:
            raise RuntimeError("existing frozen plan differs")
    else:
        atomic_write_json(frozen_plan, plan)

    started_at = time.time()
    summaries: list[dict[str, Any]] = []
    for trial in trials:
        trial_root = args.output_root / "trials" / trial["name"]
        trial_receipt = trial_root / "training-receipt.json"
        if trial_root.exists():
            if not trial_receipt.is_file():
                raise FileExistsError(
                    f"trial root exists without a complete receipt: {trial_root}"
                )
        else:
            command = command_for_trial(
                trial,
                model_root=args.model_root,
                cache_root=args.cache_root,
                output_root=args.output_root,
                device=args.device,
            )
            completed = subprocess.run(command, check=False)
            if completed.returncode != 0:
                atomic_write_json(
                    receipt_path,
                    {
                        "schema_version": 1,
                        "status": "sweep_failed",
                        "failed_trial": trial["name"],
                        "returncode": completed.returncode,
                        "completed_trials": summaries,
                    },
                )
                return completed.returncode
        summaries.append(summary_for_trial(trial, json.loads(trial_receipt.read_text())))
        atomic_write_json(
            args.output_root / "progress.json",
            {
                "schema_version": 1,
                "status": "sweep_running",
                "completed_trials": summaries,
            },
        )

    ranked = sorted(summaries, key=lambda row: row["validation_delta"]["total"])
    atomic_write_json(
        receipt_path,
        {
            "schema_version": 1,
            "status": "sweep_complete",
            "plan": str(frozen_plan),
            "elapsed_seconds": time.time() - started_at,
            "trials": summaries,
            "ranked_by_validation_total_delta": [row["name"] for row in ranked],
            "boundary": (
                "The ranking is a teacher-forced held-out screening result. It is not "
                "a perceptual voice-quality or production-readiness verdict."
            ),
        },
    )
    print(json.dumps({"status": "sweep_complete", "receipt": str(receipt_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
