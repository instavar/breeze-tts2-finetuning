from __future__ import annotations

import json

import pytest
import torch

from breeze_infer.adapter import (
    apply_adapter,
    load_adapter_manifest,
    sha256_file,
    validate_base_model,
)
from training.lora import inject_lora, save_adapter
from training.tests.test_lora import ToyModel


def _write_release(tmp_path, *, revision="a" * 40):
    model_root = tmp_path / "model"
    adapter_root = tmp_path / "adapter"
    model_root.mkdir()
    adapter_root.mkdir()
    (model_root / "config.json").write_text('{"model_type":"breeze"}\n')

    trained = ToyModel()
    inject_lora(
        trained,
        variant="backbone_depth_projection",
        rank=2,
        alpha=4,
        seed=7,
    )
    with torch.no_grad():
        for name, parameter in trained.named_parameters():
            if name.endswith("lora_B"):
                parameter.fill_(0.125)
    adapter_path = adapter_root / "adapter.safetensors"
    save_adapter(trained, adapter_path)
    manifest = {
        "schema_version": 1,
        "artifact_type": "breeze_lora_adapter",
        "base_model": {
            "id": "BreezeBlue/Breeze-TTS-2",
            "revision": revision,
            "files": {"config.json": sha256_file(model_root / "config.json")},
        },
        "adapter": {
            "file": "adapter.safetensors",
            "sha256": sha256_file(adapter_path),
        },
        "lora": {
            "variant": "backbone_depth_projection",
            "rank": 2,
            "alpha": 4,
            "seed": 7,
        },
    }
    (adapter_root / "adapter_config.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return model_root, adapter_root


def test_apply_adapter_validates_and_loads(tmp_path) -> None:
    model_root, adapter_root = _write_release(tmp_path)
    model = ToyModel()

    manifest = apply_adapter(
        model,
        model_root=model_root,
        adapter_root=adapter_root,
        base_revision="a" * 40,
    )

    assert manifest.rank == 2
    assert any(name.endswith("lora_A") for name, _ in model.named_parameters())


def test_base_revision_mismatch_fails_before_loading(tmp_path) -> None:
    model_root, adapter_root = _write_release(tmp_path)
    manifest = load_adapter_manifest(adapter_root)

    with pytest.raises(ValueError, match="base revision mismatch"):
        validate_base_model(model_root, manifest, base_revision="b" * 40)


def test_base_file_checksum_mismatch_fails(tmp_path) -> None:
    model_root, adapter_root = _write_release(tmp_path)
    manifest = load_adapter_manifest(adapter_root)
    (model_root / "config.json").write_text("changed\n")

    with pytest.raises(ValueError, match="base model checksum mismatch"):
        validate_base_model(model_root, manifest, base_revision="a" * 40)


def test_adapter_checksum_mismatch_fails(tmp_path) -> None:
    model_root, adapter_root = _write_release(tmp_path)
    manifest_path = adapter_root / "adapter_config.json"
    value = json.loads(manifest_path.read_text())
    value["adapter"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(value) + "\n")

    with pytest.raises(ValueError, match="adapter file checksum mismatch"):
        apply_adapter(
            ToyModel(),
            model_root=model_root,
            adapter_root=adapter_root,
            base_revision="a" * 40,
        )
