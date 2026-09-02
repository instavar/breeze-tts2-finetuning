from __future__ import annotations

import json
import sys

from evaluation.generate_matched import sha256_file
from evaluation.import_matched_condition import main


def test_import_matched_condition_preserves_audio_and_provenance(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    audio = source / "audio" / "adapted" / "prompt__seed-42.wav"
    metrics = source / "metrics" / "adapted.jsonl"
    audio.parent.mkdir(parents=True)
    metrics.parent.mkdir(parents=True)
    audio.write_bytes(b"frozen-wave-bytes")
    (source / "prompt-manifest.json").write_text(
        json.dumps({"schema_version": 1, "conditions": []}) + "\n"
    )
    metrics.write_text(
        json.dumps(
            {
                "status": "success",
                "model_id": "adapted",
                "condition_id": "prompt",
                "output": str(audio),
                "sha256": sha256_file(audio),
            }
        )
        + "\n"
    )
    (source / "adapted-receipt.json").write_text(
        json.dumps(
            {
                "status": "matched_generation_complete",
                "model_id": "adapted",
                "conditions": 1,
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "import-matched",
            "--source-root",
            str(source),
            "--source-model-id",
            "adapted",
            "--target-root",
            str(target),
            "--target-model-id",
            "lora",
        ],
    )

    assert main() == 0
    imported = target / "audio" / "lora" / audio.name
    assert imported.read_bytes() == audio.read_bytes()
    receipt = json.loads((target / "lora-receipt.json").read_text())
    assert receipt["status"] == "matched_generation_complete"
    assert receipt["imported_from"]["model_id"] == "adapted"
