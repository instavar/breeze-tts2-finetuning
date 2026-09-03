"""Build allowlisted, auditable model-release directories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from safetensors import safe_open

from breeze_infer.adapter import sha256_file

FORBIDDEN_NAMES = {
    "checkpoint-receipt.json",
    "optimizer.pt",
    "rng.pt",
    "scheduler.pt",
    "trainer-state.json",
}
TEXT_SUFFIXES = {".json", ".md", ".txt"}
FORBIDDEN_TEXT = ("/mnt/work/", "/mnt/ext4", "/Users/", "FEMALE_01", "female01")
NOTICE_TEXT = """Breeze TTS 2 is licensed under the BreezeBlue Research and Non-Commercial License Agreement. Copyright (c) 2026 RESONIA, INC. All Rights Reserved.

Derived from Breeze TTS 2 by BreezeBlue and licensed for research and non-commercial use only.

Instavar modified the base model through supervised adaptation. Instavar is not affiliated with or endorsed by BreezeBlue or RESONIA, INC.
"""


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file() or source.is_symlink():
        raise FileNotFoundError(f"release source must be a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    destination.chmod(0o644)


def base_file_hashes(model_root: Path) -> dict[str, str]:
    index_path = model_root / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shard_names = sorted(set(index["weight_map"].values()))
    names = ["config.json", "model.safetensors.index.json", *shard_names]
    return {name: sha256_file(model_root / name) for name in names}


def copy_release_documents(
    *, output: Path, card: Path, model_license: Path, provenance: dict[str, Any]
) -> None:
    copy_file(card, output / "README.md")
    copy_file(model_license, output / "LICENSE")
    (output / "NOTICE").write_text(NOTICE_TEXT)
    (output / "NOTICE").chmod(0o644)
    write_json(output / "PROVENANCE.json", provenance)


def safetensors_inventory(path: Path) -> dict[str, Any]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "tensor_count": len(keys),
        "tensor_key_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(),
    }


def validate_and_finalize(partial: Path, output: Path) -> None:
    for path in partial.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"release bundle contains a symlink: {path}")
        if not path.is_file():
            continue
        if path.name in FORBIDDEN_NAMES:
            raise ValueError(f"release bundle contains training state: {path.name}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(errors="replace")
            for marker in FORBIDDEN_TEXT:
                if marker in text:
                    raise ValueError(f"release text contains private marker {marker}: {path}")

    tensor_inventory = [
        safetensors_inventory(path)
        for path in sorted(partial.rglob("*.safetensors"))
    ]
    write_json(partial / "TENSOR_INVENTORY.json", tensor_inventory)
    checksum_paths = [
        path for path in sorted(partial.rglob("*")) if path.is_file()
    ]
    lines = [
        f"{sha256_file(path)}  {path.relative_to(partial).as_posix()}"
        for path in checksum_paths
    ]
    (partial / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    for path in partial.rglob("*"):
        if path.is_file():
            path.chmod(0o644)
    os.replace(partial, output)


def create_output(output: Path) -> Path:
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"refusing to overwrite release output: {output}")
    partial.mkdir(parents=True)
    return partial


def refresh_model_card(args: argparse.Namespace) -> None:
    checksums_path = args.bundle / "SHA256SUMS"
    rows: list[tuple[str, str]] = []
    for line in checksums_path.read_text().splitlines():
        digest, relative_name = line.split("  ", 1)
        rows.append((digest, relative_name))
    checksum_map = {name: digest for digest, name in rows}
    if "README.md" not in checksum_map:
        raise ValueError("release checksum manifest does not contain README.md")
    current_card = args.bundle / "README.md"
    if sha256_file(current_card) != checksum_map["README.md"]:
        raise ValueError("current model card differs from the release checksum manifest")
    replacement = args.card.read_text()
    for marker in FORBIDDEN_TEXT:
        if marker in replacement:
            raise ValueError(f"replacement model card contains private marker {marker}")

    partial_card = current_card.with_name("README.md.partial")
    partial_card.write_text(replacement)
    partial_card.chmod(0o644)
    os.replace(partial_card, current_card)
    new_digest = sha256_file(current_card)
    checksum_lines = [
        f"{new_digest if name == 'README.md' else digest}  {name}"
        for digest, name in rows
    ]
    partial_checksums = checksums_path.with_name("SHA256SUMS.partial")
    partial_checksums.write_text("\n".join(checksum_lines) + "\n")
    partial_checksums.chmod(0o644)
    os.replace(partial_checksums, checksums_path)
    print(args.bundle)


def build_lora(args: argparse.Namespace) -> None:
    partial = create_output(args.output)
    expected_adapter_hash = args.adapter_sha256.lower()
    observed_adapter_hash = sha256_file(args.adapter)
    if observed_adapter_hash != expected_adapter_hash:
        raise ValueError("source adapter checksum does not match the selected checkpoint")
    copy_file(args.adapter, partial / "adapter.safetensors")
    base_files = base_file_hashes(args.base_model_root)
    manifest = {
        "schema_version": 1,
        "artifact_type": "breeze_lora_adapter",
        "base_model": {
            "id": args.base_model_id,
            "revision": args.base_revision,
            "files": base_files,
        },
        "adapter": {
            "file": "adapter.safetensors",
            "sha256": observed_adapter_hash,
        },
        "lora": {
            "variant": "backbone_depth_projection",
            "rank": 8,
            "alpha": 16.0,
            "seed": 42,
        },
    }
    write_json(partial / "adapter_config.json", manifest)
    provenance = json.loads(args.provenance.read_text())
    provenance["artifact"] = {
        "kind": "lora_adapter",
        "selected_checkpoint_step": 500,
        "sha256": observed_adapter_hash,
    }
    provenance["base_model"] = {
        "id": args.base_model_id,
        "revision": args.base_revision,
        "files": base_files,
    }
    copy_release_documents(
        output=partial,
        card=args.card,
        model_license=args.model_license,
        provenance=provenance,
    )
    validate_and_finalize(partial, args.output)


def build_full_sft(args: argparse.Namespace) -> None:
    partial = create_output(args.output)
    model_role_path = args.source / "model-role.json"
    role = json.loads(model_role_path.read_text())
    files: dict[str, str] = role["files"]
    for relative_name, expected_hash in files.items():
        source = args.source / relative_name
        if sha256_file(source) != expected_hash:
            raise ValueError(f"model role checksum mismatch: {relative_name}")
        copy_file(source, partial / relative_name)

    audio_root = args.source / "audio_tokenizer"
    allowed_audio_files = {
        "config.json",
        "configuration.json",
        "model.safetensors",
        "preprocessor_config.json",
    }
    observed_audio_files = {
        path.relative_to(audio_root).as_posix()
        for path in audio_root.rglob("*")
        if path.is_file()
    }
    if observed_audio_files != allowed_audio_files:
        raise ValueError(
            "audio tokenizer allowlist differs: "
            f"expected={sorted(allowed_audio_files)} "
            f"observed={sorted(observed_audio_files)}"
        )
    for relative_name in sorted(allowed_audio_files):
        copy_file(audio_root / relative_name, partial / "audio_tokenizer" / relative_name)

    provenance = json.loads(args.provenance.read_text())
    provenance["artifact"] = {
        "kind": "full_sft_inference_checkpoint",
        "selected_checkpoint_step": 750,
        "model_files": files,
    }
    provenance["base_model"] = {
        "id": args.base_model_id,
        "revision": args.base_revision,
    }
    copy_release_documents(
        output=partial,
        card=args.card,
        model_license=args.model_license,
        provenance=provenance,
    )
    validate_and_finalize(partial, args.output)


def common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--model-license", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--base-model-id", default="BreezeBlue/Breeze-TTS-2")
    parser.add_argument("--base-revision", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    lora = subparsers.add_parser("lora")
    common(lora)
    lora.add_argument("--adapter", type=Path, required=True)
    lora.add_argument("--adapter-sha256", required=True)
    lora.add_argument("--base-model-root", type=Path, required=True)
    lora.set_defaults(run=build_lora)
    full = subparsers.add_parser("full-sft")
    common(full)
    full.add_argument("--source", type=Path, required=True)
    full.set_defaults(run=build_full_sft)
    refresh = subparsers.add_parser("refresh-card")
    refresh.add_argument("--bundle", type=Path, required=True)
    refresh.add_argument("--card", type=Path, required=True)
    refresh.set_defaults(run=refresh_model_card)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.run(args)
    result_path = getattr(args, "output", None) or getattr(args, "bundle", None)
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
