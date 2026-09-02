from __future__ import annotations

import json
import stat
import wave
from pathlib import Path

from evaluation.build_blind_pack import main


def write_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16)


def test_builds_opaque_pack_and_private_mode(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    (run_root / "prompt-manifest.json").parent.mkdir(parents=True)
    (run_root / "prompt-manifest.json").write_text(
        json.dumps(
            {"conditions": [{"condition_id": "neutral", "text": "Hello there."}]}
        )
    )
    write_wav(run_root / "audio" / "base" / "neutral__seed-42.wav")
    write_wav(run_root / "audio" / "adapted" / "neutral__seed-42.wav")
    output = tmp_path / "blind"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_blind_pack",
            "--run-root",
            str(run_root),
            "--output-root",
            str(output),
        ],
    )

    assert main() == 0
    public = json.loads((output / "manifest.json").read_text())
    private = json.loads((output / "private-key.json").read_text())
    assert len(public["samples"]) == 2
    assert all("model_id" not in row for row in public["samples"])
    assert {row["model_id"] for row in private["samples"]} == {"base", "adapted"}
    assert stat.S_IMODE((output / "private-key.json").stat().st_mode) == 0o600
