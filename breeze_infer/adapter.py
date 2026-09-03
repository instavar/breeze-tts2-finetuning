"""Validated loading for Instavar Breeze LoRA adapter releases."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from training.lora import inject_lora, load_adapter

MANIFEST_NAME = "adapter_config.json"


@dataclass(frozen=True)
class AdapterManifest:
    base_model_id: str
    base_revision: str
    base_files: dict[str, str]
    adapter_file: str
    adapter_sha256: str
    variant: str
    rank: int
    alpha: float
    seed: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_adapter_manifest(adapter_root: Path) -> AdapterManifest:
    path = adapter_root / MANIFEST_NAME
    value: dict[str, Any] = json.loads(path.read_text())
    if value.get("schema_version") != 1:
        raise ValueError("unsupported adapter manifest schema")
    if value.get("artifact_type") != "breeze_lora_adapter":
        raise ValueError("manifest is not a Breeze LoRA adapter")

    base = value.get("base_model")
    adapter = value.get("adapter")
    lora = value.get("lora")
    if not all(isinstance(item, dict) for item in (base, adapter, lora)):
        raise ValueError("adapter manifest sections are incomplete")

    adapter_file = str(adapter["file"])
    if Path(adapter_file).name != adapter_file:
        raise ValueError("adapter file must be a top-level filename")
    base_files = base.get("files", {})
    if not isinstance(base_files, dict) or not base_files:
        raise ValueError("base_model.files must contain at least one checksum")
    for relative_name, expected_hash in base_files.items():
        if Path(relative_name).is_absolute() or ".." in Path(relative_name).parts:
            raise ValueError(f"unsafe base model path: {relative_name}")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"invalid base model checksum: {relative_name}")

    return AdapterManifest(
        base_model_id=str(base["id"]),
        base_revision=str(base["revision"]),
        base_files={str(name): str(digest) for name, digest in base_files.items()},
        adapter_file=adapter_file,
        adapter_sha256=str(adapter["sha256"]),
        variant=str(lora["variant"]),
        rank=int(lora["rank"]),
        alpha=float(lora["alpha"]),
        seed=int(lora["seed"]),
    )


def _revision_from_snapshot_path(model_root: Path) -> str | None:
    parts = model_root.resolve().parts
    for index, part in enumerate(parts[:-1]):
        if part == "snapshots":
            return parts[index + 1]
    marker = model_root / "BASE_MODEL_REVISION"
    if marker.is_file():
        return marker.read_text().strip()
    return None


def validate_base_model(
    model_root: Path,
    manifest: AdapterManifest,
    *,
    base_revision: str | None,
) -> None:
    observed_revision = base_revision or _revision_from_snapshot_path(model_root)
    if not observed_revision:
        raise ValueError(
            "base revision cannot be inferred; pass --base-revision or add "
            "BASE_MODEL_REVISION to the model directory"
        )
    if observed_revision != manifest.base_revision:
        raise ValueError(
            f"base revision mismatch: expected {manifest.base_revision}, "
            f"got {observed_revision}"
        )
    for relative_name, expected_hash in manifest.base_files.items():
        path = model_root / relative_name
        if not path.is_file():
            raise FileNotFoundError(f"base model file is missing: {relative_name}")
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise ValueError(f"base model checksum mismatch: {relative_name}")


def apply_adapter(
    model: torch.nn.Module,
    *,
    model_root: Path,
    adapter_root: Path,
    base_revision: str | None,
) -> AdapterManifest:
    manifest = load_adapter_manifest(adapter_root)
    validate_base_model(model_root, manifest, base_revision=base_revision)
    adapter_path = adapter_root / manifest.adapter_file
    if not adapter_path.is_file():
        raise FileNotFoundError(f"adapter file is missing: {adapter_path}")
    if sha256_file(adapter_path) != manifest.adapter_sha256:
        raise ValueError("adapter file checksum mismatch")
    inject_lora(
        model,
        variant=manifest.variant,
        rank=manifest.rank,
        alpha=manifest.alpha,
        seed=manifest.seed,
    )
    load_adapter(model, adapter_path)
    model.eval()
    return manifest
