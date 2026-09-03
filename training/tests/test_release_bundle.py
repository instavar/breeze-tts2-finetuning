from __future__ import annotations

import json
from argparse import Namespace

import pytest
import torch
from safetensors.torch import save_file

from breeze_infer.adapter import sha256_file
from training.release_bundle import build_full_sft, build_lora


def _docs(tmp_path):
    card = tmp_path / "card.md"
    license_path = tmp_path / "model-license.txt"
    provenance = tmp_path / "provenance.json"
    card.write_text("---\nlicense: other\n---\n# Test\n")
    license_path.write_text("test model agreement\n")
    provenance.write_text('{"schema_version":1}\n')
    return card, license_path, provenance


def _base(tmp_path):
    root = tmp_path / "base"
    root.mkdir()
    (root / "config.json").write_text("{}\n")
    save_file({"model.weight": torch.ones(1)}, root / "model.safetensors")
    index = {"weight_map": {"model.weight": "model.safetensors"}}
    (root / "model.safetensors.index.json").write_text(json.dumps(index) + "\n")
    return root


def test_build_lora_copies_only_release_roles(tmp_path) -> None:
    base = _base(tmp_path)
    adapter = tmp_path / "adapter.safetensors"
    save_file({"layer.lora_A": torch.ones(1)}, adapter)
    card, license_path, provenance = _docs(tmp_path)
    output = tmp_path / "release"
    build_lora(
        Namespace(
            output=output,
            adapter=adapter,
            adapter_sha256=sha256_file(adapter),
            base_model_root=base,
            base_model_id="BreezeBlue/Breeze-TTS-2",
            base_revision="a" * 40,
            card=card,
            model_license=license_path,
            provenance=provenance,
        )
    )
    assert {path.name for path in output.iterdir()} == {
        "LICENSE",
        "NOTICE",
        "PROVENANCE.json",
        "README.md",
        "SHA256SUMS",
        "TENSOR_INVENTORY.json",
        "adapter.safetensors",
        "adapter_config.json",
    }


def test_build_full_sft_rejects_training_role(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model-role.json").write_text(
        json.dumps({"files": {"optimizer.pt": "0" * 64}}) + "\n"
    )
    (source / "optimizer.pt").write_bytes(b"not a model")
    card, license_path, provenance = _docs(tmp_path)

    with pytest.raises(ValueError, match="model role checksum mismatch"):
        build_full_sft(
            Namespace(
                output=tmp_path / "release",
                source=source,
                base_model_id="BreezeBlue/Breeze-TTS-2",
                base_revision="a" * 40,
                card=card,
                model_license=license_path,
                provenance=provenance,
            )
        )
