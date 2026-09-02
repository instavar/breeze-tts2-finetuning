from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from training.model_loading import load_eager_config, load_tokenizer
from training.supervised_example import build_supervised_example, example_summary


@dataclass(frozen=True)
class ManifestRow:
    line_number: int
    audio: Path
    text: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_row_key(row: ManifestRow, seed: int) -> str:
    payload = f"{seed}\0{row.audio}\0{row.text}".encode()
    return hashlib.sha256(payload).hexdigest()


def read_manifest(path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            audio = Path(value["audio"])
            text = str(value["text"]).strip()
            if not text:
                raise ValueError(f"empty transcript at {path}:{line_number}")
            if not audio.is_file():
                raise FileNotFoundError(
                    f"missing audio at {path}:{line_number}: {audio}"
                )
            rows.append(ManifestRow(line_number, audio, text))
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    return rows


def select_rows(rows: list[ManifestRow], *, limit: int, seed: int) -> list[ManifestRow]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    if limit > len(rows):
        raise ValueError(f"limit {limit} exceeds manifest rows {len(rows)}")
    return sorted(rows, key=lambda row: stable_row_key(row, seed))[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic Breeze supervised-example cache"
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=1024)
    parser.add_argument("--validation-limit", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speaker", default="S0")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def build_split(
    *,
    split: str,
    rows: list[ManifestRow],
    output_root: Path,
    tokenizer: Any,
    audio_tokenizer: Any,
    config: Any,
    speaker: str,
) -> list[dict[str, Any]]:
    split_root = output_root / split
    split_root.mkdir(parents=True, exist_ok=False)
    receipt_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        example = build_supervised_example(
            tokenizer,
            audio_tokenizer,
            config,
            audio_path=row.audio,
            transcript=row.text,
            speaker=speaker,
        )
        artifact = split_root / f"{index:06d}.pt"
        torch.save(example, artifact)
        receipt_rows.append(
            {
                "index": index,
                "manifest_line": row.line_number,
                "artifact": str(artifact.relative_to(output_root)),
                "audio": str(row.audio),
                "audio_sha256": sha256_file(row.audio),
                "transcript_sha256": hashlib.sha256(row.text.encode()).hexdigest(),
                "example": example_summary(example, config),
            }
        )
        if (index + 1) % 100 == 0 or index + 1 == len(rows):
            print(f"{split}: prepared {index + 1}/{len(rows)}", flush=True)
    return receipt_rows


def main() -> int:
    args = parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to overwrite cache: {args.output_root}")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("real-data preparation requires an available CUDA device")

    partial_root = args.output_root.with_name(args.output_root.name + ".partial")
    if partial_root.exists():
        raise FileExistsError(f"stale partial cache requires review: {partial_root}")
    partial_root.mkdir(parents=True, exist_ok=False)

    source_root = Path(__file__).resolve().parents[1]
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty_paths = subprocess.check_output(
        ["git", "-C", str(source_root), "status", "--short"], text=True
    ).splitlines()
    if dirty_paths:
        raise RuntimeError(f"source repository must be clean: {dirty_paths}")

    train_rows = select_rows(
        read_manifest(args.train_manifest), limit=args.train_limit, seed=args.seed
    )
    validation_rows = select_rows(
        read_manifest(args.validation_manifest),
        limit=args.validation_limit,
        seed=args.seed + 1,
    )

    started_at = time.time()
    torch.cuda.set_device(args.device)
    config = load_eager_config(args.model_root)
    tokenizer = load_tokenizer(args.model_root)
    from qwen_tts import Qwen3TTSTokenizer

    audio_tokenizer = Qwen3TTSTokenizer.from_pretrained(
        str(args.model_root / "audio_tokenizer"), device_map=args.device
    )
    splits = {
        "train": build_split(
            split="train",
            rows=train_rows,
            output_root=partial_root,
            tokenizer=tokenizer,
            audio_tokenizer=audio_tokenizer,
            config=config,
            speaker=args.speaker,
        ),
        "validation": build_split(
            split="validation",
            rows=validation_rows,
            output_root=partial_root,
            tokenizer=tokenizer,
            audio_tokenizer=audio_tokenizer,
            config=config,
            speaker=args.speaker,
        ),
    }
    del audio_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    receipt = {
        "schema_version": 1,
        "status": "deterministic_supervised_cache_complete",
        "source": {
            "repository": str(source_root),
            "revision": revision,
            "model_root": str(args.model_root),
            "train_manifest": str(args.train_manifest),
            "train_manifest_sha256": sha256_file(args.train_manifest),
            "validation_manifest": str(args.validation_manifest),
            "validation_manifest_sha256": sha256_file(args.validation_manifest),
        },
        "selection": {
            "seed": args.seed,
            "train_limit": args.train_limit,
            "validation_limit": args.validation_limit,
            "speaker": args.speaker,
        },
        "splits": splits,
        "runtime": {
            "elapsed_seconds": time.time() - started_at,
            "device": torch.cuda.get_device_name(args.device),
        },
        "interpretation_boundary": (
            "The cache binds selected manifest rows to supervised tensors. It does "
            "not establish transcript correctness, contributor identity, model "
            "quality, or split generalisation."
        ),
    }
    receipt_path = partial_root / "cache-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    partial_root.rename(args.output_root)
    print(json.dumps({"status": receipt["status"], "output": str(args.output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
