from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any, mode: int | None = None) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    if mode is not None:
        partial.chmod(mode)
    os.replace(partial, path)


def collect_audio(run_root: Path) -> list[Path]:
    audio = sorted(
        path
        for path in (run_root / "audio").glob("*/*.wav")
        if not path.name.startswith(".")
    )
    if not audio:
        raise RuntimeError("no matched audio found")
    return audio


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an opaque blind-listening pack for matched Breeze audio"
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prompt_manifest = json.loads((args.run_root / "prompt-manifest.json").read_text())
    prompts = {
        row["condition_id"]: row["text"] for row in prompt_manifest["conditions"]
    }
    audio = collect_audio(args.run_root)
    expected_count = len(prompts) * len({path.parent.name for path in audio})
    if len(audio) != expected_count:
        raise RuntimeError(
            f"incomplete matched pack: found {len(audio)}, expected {expected_count}"
        )

    output_root = args.output_root or args.run_root / "blind-pack"
    audio_root = output_root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    shuffled = list(audio)
    random.Random(args.seed).shuffle(shuffled)

    public_rows = []
    private_rows = []
    for index, source in enumerate(shuffled, start=1):
        opaque_id = f"sample-{index:03d}"
        condition_id = source.name.split("__seed-", 1)[0]
        destination = audio_root / f"{opaque_id}.wav"
        source_hash = sha256_file(source)
        if destination.exists():
            if sha256_file(destination) != source_hash:
                raise RuntimeError(f"blind artifact changed: {destination}")
        else:
            shutil.copyfile(source, destination)
        public_rows.append(
            {
                "sample_id": opaque_id,
                "condition_id": condition_id,
                "prompt": prompts[condition_id],
                "audio": f"audio/{destination.name}",
                "audio_sha256": source_hash,
            }
        )
        private_rows.append(
            {
                "sample_id": opaque_id,
                "condition_id": condition_id,
                "model_id": source.parent.name,
                "source": str(source),
                "source_sha256": source_hash,
            }
        )

    atomic_json(
        output_root / "manifest.json",
        {
            "schema_version": 1,
            "status": "blind_pack_complete",
            "seed": args.seed,
            "samples": public_rows,
            "rating_dimensions": [
                "speaker_identity",
                "singaporean_accent",
                "cadence",
                "pronunciation",
                "long_form_monotony",
                "listening_fatigue",
                "overall_naturalness",
            ],
            "interpretation_boundary": (
                "The public manifest intentionally hides model labels. Decode the "
                "private key only after ratings are frozen."
            ),
        },
    )
    private_key = output_root / "private-key.json"
    atomic_json(
        private_key,
        {
            "schema_version": 1,
            "status": "private_blind_key",
            "samples": private_rows,
        },
        mode=0o600,
    )
    private_key.chmod(0o600)
    print(
        json.dumps(
            {
                "status": "blind_pack_complete",
                "samples": len(public_rows),
                "output_root": str(output_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
