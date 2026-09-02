from __future__ import annotations

import json
from pathlib import Path

import torch

from training.real_full_sft import AdafactorOptimizer, FP32MasterSGD
from training.real_full_sft_verify import copy_inference_export


def test_fp32_master_sgd_updates_bf16_parameter() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.bfloat16))
    parameter.grad = torch.tensor([0.5, -0.25], dtype=torch.bfloat16)
    optimizer = FP32MasterSGD([("weight", parameter)], weight_decay=0.0)

    optimizer.step(0.1)

    assert torch.allclose(parameter.float(), torch.tensor([0.94921875, -1.9765625]))
    receipt = optimizer.receipt()
    assert receipt["parameter_tensors"] == 1
    assert receipt["parameter_elements"] == 2
    assert receipt["state_bytes"] == 8


def test_fp32_master_sgd_state_round_trip_preserves_hidden_precision() -> None:
    first = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    first.grad = torch.tensor([0.001], dtype=torch.bfloat16)
    optimizer = FP32MasterSGD([("weight", first)], weight_decay=0.0)
    optimizer.step(0.001)
    state = optimizer.state_dict_cpu()

    second = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    restored = FP32MasterSGD([("weight", second)], weight_decay=0.0)
    restored.load_state_dict(state)

    assert torch.equal(restored.state_dict_cpu()["weight"], state["weight"])
    assert second.item() == first.item()


def test_fp32_master_sgd_applies_family_learning_rate_multiplier() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    parameter.grad = torch.tensor([1.0], dtype=torch.bfloat16)
    optimizer = FP32MasterSGD(
        [("backbone_model.weight", parameter)],
        weight_decay=0.0,
        lr_multipliers={"backbone": 0.25},
    )

    optimizer.step(0.1)

    assert torch.allclose(parameter.float(), torch.tensor([0.9765625]))
    assert optimizer.receipt()["lr_multipliers"] == {"backbone": 0.25}


def test_adafactor_state_round_trip_preserves_family_groups() -> None:
    first = torch.nn.Parameter(torch.tensor([[1.0, -2.0]], dtype=torch.float32))
    optimizer = AdafactorOptimizer(
        [("backbone_model.weight", first)],
        weight_decay=0.01,
        lr_multipliers={
            "backbone": 0.25,
            "depth_decoder": 1.0,
            "text_projection": 1.0,
            "other_synthesis": 1.0,
        },
    )
    first.grad = torch.tensor([[0.5, -0.25]], dtype=torch.float32)
    optimizer.step(0.01)
    state = optimizer.state_dict_cpu()

    second = torch.nn.Parameter(first.detach().clone())
    restored = AdafactorOptimizer(
        [("backbone_model.weight", second)],
        weight_decay=0.01,
        lr_multipliers={
            "backbone": 0.25,
            "depth_decoder": 1.0,
            "text_projection": 1.0,
            "other_synthesis": 1.0,
        },
    )
    restored.load_state_dict(state)

    assert restored.receipt()["implementation"] == "torch.optim.Adafactor"
    assert restored.receipt()["state_bytes"] > 0
    assert restored.optimizer.param_groups[0]["family"] == "backbone"
    assert restored.optimizer.param_groups[0]["lr_multiplier"] == 0.25


def test_copy_inference_export_excludes_training_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n")
    (checkpoint / "model.safetensors").write_bytes(b"model")
    (checkpoint / "model-role.json").write_text(
        json.dumps(
            {
                "files": {
                    "config.json": "unused-in-copy-test",
                    "model.safetensors": "unused-in-copy-test",
                }
            }
        )
    )
    (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    (checkpoint / "trainer-state.json").write_text("{}\n")

    export = tmp_path / "export"
    copy_inference_export(checkpoint, export)

    assert (export / "config.json").is_file()
    assert (export / "model.safetensors").read_bytes() == b"model"
    assert (export / "model-role.json").is_file()
    assert not (export / "optimizer.pt").exists()
    assert not (export / "trainer-state.json").exists()
