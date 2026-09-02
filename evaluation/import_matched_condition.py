from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from evaluation.generate_matched import atomic_json, sha256_file


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    partial = path.with_name(path.name + ".partial")
    with partial.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import a hash-verified matched Breeze condition"
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-model-id", required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--target-model-id", required=True)
    args = parser.parse_args()

    source_receipt_path = args.source_root / f"{args.source_model_id}-receipt.json"
    source_receipt = json.loads(source_receipt_path.read_text())
    if source_receipt.get("status") != "matched_generation_complete":
        raise RuntimeError("source matched generation is incomplete")
    source_manifest = args.source_root / "prompt-manifest.json"
    target_manifest = args.target_root / "prompt-manifest.json"
    args.target_root.mkdir(parents=True, exist_ok=True)
    if target_manifest.exists():
        if sha256_file(target_manifest) != sha256_file(source_manifest):
            raise RuntimeError("source and target prompt manifests differ")
    else:
        shutil.copy2(source_manifest, target_manifest)

    source_metrics = (
        args.source_root / "metrics" / f"{args.source_model_id}.jsonl"
    )
    rows = [
        json.loads(line)
        for line in source_metrics.read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != int(source_receipt["conditions"]):
        raise RuntimeError("source metrics count differs from source receipt")

    target_audio = args.target_root / "audio" / args.target_model_id
    target_metrics = args.target_root / "metrics" / f"{args.target_model_id}.jsonl"
    target_receipt = args.target_root / f"{args.target_model_id}-receipt.json"
    if target_metrics.exists() or target_receipt.exists():
        raise FileExistsError("target metrics or receipt already exists")
    target_audio.mkdir(parents=True, exist_ok=True)
    target_metrics.parent.mkdir(parents=True, exist_ok=True)

    imported = []
    for row in rows:
        source = Path(row["output"])
        expected_hash = row["sha256"]
        if not source.is_file() or sha256_file(source) != expected_hash:
            raise RuntimeError(f"source audio differs from receipt: {source}")
        destination = target_audio / source.name
        if destination.exists():
            if sha256_file(destination) != expected_hash:
                raise RuntimeError(f"target audio collision differs: {destination}")
        else:
            shutil.copy2(source, destination)
        imported.append(
            {
                **row,
                "model_id": args.target_model_id,
                "output": str(destination),
                "imported_from": {
                    "run_root": str(args.source_root),
                    "model_id": args.source_model_id,
                    "receipt_sha256": sha256_file(source_receipt_path),
                },
            }
        )

    atomic_jsonl(target_metrics, imported)
    atomic_json(
        target_receipt,
        {
            "schema_version": 1,
            "status": "matched_generation_complete",
            "model_id": args.target_model_id,
            "conditions": len(imported),
            "metrics": str(target_metrics),
            "imported_from": {
                "run_root": str(args.source_root),
                "model_id": args.source_model_id,
                "receipt_sha256": sha256_file(source_receipt_path),
                "prompt_manifest_sha256": sha256_file(source_manifest),
            },
        },
    )
    print(
        json.dumps(
            {
                "status": "matched_condition_imported",
                "source_model_id": args.source_model_id,
                "target_model_id": args.target_model_id,
                "conditions": len(imported),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
