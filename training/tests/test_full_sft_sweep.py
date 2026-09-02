from __future__ import annotations

from pathlib import Path

import pytest

from training.full_sft_sweep import command_for_trial, validate_plan


def base_plan() -> dict:
    return {
        "schema_version": 1,
        "defaults": {
            "max_steps": 50,
            "gradient_accumulation": 4,
            "warmup_steps": 5,
            "validation_examples": 64,
            "max_gradient_norm": 1.0,
            "backbone_lr_multiplier": 1.0,
            "depth_decoder_lr_multiplier": 1.0,
            "text_projection_lr_multiplier": 1.0,
            "other_synthesis_lr_multiplier": 1.0,
        },
        "trials": [
            {
                "name": "adafactor-1e-6",
                "optimizer": "adafactor",
                "learning_rate": 1e-6,
            }
        ],
    }


def test_validate_plan_resolves_defaults() -> None:
    trial = validate_plan(base_plan())[0]

    assert trial["name"] == "adafactor-1e-6"
    assert trial["max_steps"] == 50
    assert trial["gradient_accumulation"] == 4


def test_validate_plan_rejects_duplicate_names() -> None:
    plan = base_plan()
    plan["trials"].append(dict(plan["trials"][0]))

    with pytest.raises(ValueError, match="duplicate trial name"):
        validate_plan(plan)


def test_command_for_trial_uses_metric_only_defaults() -> None:
    trial = validate_plan(base_plan())[0]

    command = command_for_trial(
        trial,
        model_root=Path("/model"),
        cache_root=Path("/cache"),
        output_root=Path("/output"),
        device="cuda:0",
    )

    assert command[0]
    assert command[1:4] == ["-m", "training.real_full_sft", "--model-root"]
    assert command[command.index("--save-policy") + 1] == "none"
    assert command[command.index("--validation-every") + 1] == "50"


def test_validate_plan_accepts_model_only_finalist_export() -> None:
    plan = base_plan()
    plan["trials"][0]["save_policy"] = "model_only"

    trial = validate_plan(plan)[0]

    assert trial["save_policy"] == "model_only"


def test_command_for_trial_can_freeze_a_longer_schedule_horizon() -> None:
    plan = base_plan()
    plan["defaults"]["schedule_horizon_steps"] = 1000
    trial = validate_plan(plan)[0]

    command = command_for_trial(
        trial,
        model_root=Path("/model"),
        cache_root=Path("/cache"),
        output_root=Path("/output"),
        device="cuda:0",
    )

    assert command[command.index("--schedule-horizon-steps") + 1] == "1000"
