from __future__ import annotations

import json
from pathlib import Path

from training.select_checkpoint import checkpoint_rows


def write_checkpoint(root: Path, step: int, total: float) -> None:
    checkpoint = root / f"checkpoint-step-{step:06d}"
    checkpoint.mkdir()
    (checkpoint / "trainer-state.json").write_text(
        json.dumps(
            {
                "global_step": step,
                "history": [
                    {
                        "validation_loss": {
                            "backbone": total - 4.0,
                            "depth_decoder": 4.0,
                            "total": total,
                        }
                    }
                ],
            }
        )
    )
    (checkpoint / "checkpoint-receipt.json").write_text(
        json.dumps({"status": "five_role_checkpoint_complete"})
    )


def test_checkpoint_rows_exposes_validation_and_hashes(tmp_path: Path) -> None:
    write_checkpoint(tmp_path, 250, 5.4)
    write_checkpoint(tmp_path, 500, 5.2)

    rows = checkpoint_rows(tmp_path)

    assert [row["global_step"] for row in rows] == [250, 500]
    assert rows[1]["validation_loss"]["total"] == 5.2
    assert len(rows[1]["trainer_state_sha256"]) == 64
    assert len(rows[1]["checkpoint_receipt_sha256"]) == 64
