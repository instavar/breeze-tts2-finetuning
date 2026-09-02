from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def checkpoint_rows(training_root: Path) -> list[dict[str, Any]]:
    rows = []
    for checkpoint in sorted(training_root.glob("checkpoint-step-*")):
        trainer_state_path = checkpoint / "trainer-state.json"
        checkpoint_receipt_path = checkpoint / "checkpoint-receipt.json"
        if not trainer_state_path.is_file() or not checkpoint_receipt_path.is_file():
            continue
        trainer_state = json.loads(trainer_state_path.read_text())
        history = trainer_state.get("history", [])
        if not history or "validation_loss" not in history[-1]:
            continue
        rows.append(
            {
                "checkpoint": str(checkpoint),
                "global_step": int(trainer_state["global_step"]),
                "validation_loss": history[-1]["validation_loss"],
                "trainer_state_sha256": sha256_file(trainer_state_path),
                "checkpoint_receipt_sha256": sha256_file(checkpoint_receipt_path),
            }
        )
    if not rows:
        raise RuntimeError("no complete checkpoint validation records found")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select the Breeze checkpoint with minimum held-out total loss"
    )
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = checkpoint_rows(args.training_root)
    selected = min(rows, key=lambda row: row["validation_loss"]["total"])
    output = args.output or args.training_root / "selected-checkpoint.json"
    atomic_json(
        output,
        {
            "schema_version": 1,
            "status": "validation_selected_checkpoint",
            "criterion": "minimum held-out teacher-forced total loss",
            "selected": selected,
            "candidates": rows,
            "interpretation_boundary": (
                "Teacher-forced validation loss selects a checkpoint for further "
                "evaluation. It does not establish speaker identity, accent, cadence, "
                "naturalness, or listening quality."
            ),
        },
    )
    print(json.dumps({"status": "selected", **selected}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
