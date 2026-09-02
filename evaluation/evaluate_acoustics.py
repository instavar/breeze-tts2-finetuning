from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(
        (args.run_root / "prompt-manifest.json").read_text(encoding="utf-8")
    )
    target_words = {
        row["condition_id"]: len(
            re.findall(r"[a-z0-9]+", re.sub(r"\([^)]*\)", " ", row["text"].lower()))
        )
        for row in manifest["conditions"]
    }
    output_path = args.run_root / "metrics" / "acoustic-results.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["sample_id"]] = row

    audio_files = sorted(
        path
        for path in (args.run_root / "audio").glob("*/*.wav")
        if not path.name.startswith(".")
    )
    for audio_path in audio_files:
        sample_id = f"{audio_path.parent.name}/{audio_path.stem}"
        audio_hash = sha256(audio_path)
        if sample_id in completed:
            if completed[sample_id]["audio_sha256"] != audio_hash:
                raise RuntimeError(
                    f"audio changed after acoustic extraction: {sample_id}"
                )
            print(f"skip verified {sample_id}", flush=True)
            continue
        audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        audio = audio - float(np.mean(audio))
        duration = len(audio) / sample_rate
        rms_frames = librosa.feature.rms(
            y=audio, frame_length=2048, hop_length=512
        ).squeeze()
        silence_fraction = (
            float(np.mean(rms_frames < 0.01)) if rms_frames.size else None
        )
        f0 = librosa.yin(
            audio, fmin=65, fmax=400, sr=sample_rate, frame_length=2048, hop_length=512
        )
        voiced = f0[np.isfinite(f0) & (f0 >= 65) & (f0 <= 400)]
        median_f0 = float(np.median(voiced)) if voiced.size else math.nan
        semitones = (
            12.0 * np.log2(voiced / median_f0)
            if voiced.size and median_f0 > 0
            else np.array([])
        )
        condition_id = audio_path.name.split("__seed-", 1)[0]
        row = {
            "sample_id": sample_id,
            "model_id": audio_path.parent.name,
            "condition_id": condition_id,
            "audio_path": str(audio_path),
            "audio_sha256": audio_hash,
            "sample_rate": sample_rate,
            "duration_seconds": duration,
            "peak_absolute": float(np.max(np.abs(audio))) if audio.size else None,
            "rms": float(np.sqrt(np.mean(np.square(audio)))) if audio.size else None,
            "silence_frame_fraction_below_minus_40_dbfs": silence_fraction,
            "median_f0_hz": finite_or_none(median_f0),
            "f0_semitone_std": finite_or_none(float(np.std(semitones)))
            if semitones.size
            else None,
            "seconds_per_target_word": duration / target_words[condition_id]
            if target_words[condition_id]
            else None,
            "interpretation_boundary": "These deterministic timing and pitch diagnostics cannot rank naturalness, cadence quality, accent, monotony, or listening fatigue.",
        }
        append_jsonl(output_path, row)
        completed[sample_id] = row
        print(f"completed {sample_id} duration={duration:.3f}s", flush=True)


if __name__ == "__main__":
    main()
