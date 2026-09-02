from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import soundfile as sf
import torch

from breeze_infer.runtime import (
    load_runtime,
    set_all_seeds,
    update_generation_config_for_breeze,
)
from breeze_infer.templates import get_template, prepare_inputs
from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

REFERENCE_TRANSCRIPT = (
    "He was a head shorter than his companion, of almost delicate physique."
)


@dataclass(frozen=True)
class Condition:
    condition_id: str
    text: str
    instruction: str


CONDITIONS = (
    Condition(
        "clone_neutral",
        "The morning meeting starts at nine, and I will send the revised notes before lunch.",
        "Speak clearly and naturally.",
    ),
    Condition(
        "clone_singapore_ordinary",
        "The train was delayed near Tanjong Pagar, so Mei Lin took the bus home after work.",
        "Speak clearly and naturally.",
    ),
    Condition(
        "clone_local_pronunciation",
        "Sze Min met Farhan at Jalan Membina before they bought chicken rice at the hawker centre. She felt paiseh after forgetting his kopi order.",
        "Speak clearly and naturally.",
    ),
    Condition(
        "clone_cadence",
        "At first, nothing happened. Then the phone rang twice, the lift doors opened, and everyone turned around. No one spoke. After a moment, Priya laughed and the room relaxed.",
        "Speak clearly and naturally, preserving natural changes in pace and emphasis.",
    ),
    Condition(
        "clone_longform",
        "On Saturday morning, I left home earlier than usual because the forecast warned of heavy rain. The station was already crowded, but the train arrived on time and the journey into town was quiet. I reviewed my notes, answered two messages, and watched the neighbourhoods pass outside the window. By the time I reached the office, the clouds had cleared. We spent the next hour checking the presentation, shortening a few crowded slides, and rehearsing the difficult transitions. The discussion with the client was calm and practical. They asked careful questions, we explained the trade-offs, and everyone agreed on the next steps. After lunch, I walked back through the park instead of taking the bus. It was a small change, but the slower route gave me time to think about what had gone well and what I would improve next time.",
        "Speak clearly and naturally. Vary pace and emphasis across sentences without sounding theatrical.",
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one matched Breeze base or adapted condition"
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--mode", choices=("reference", "reference_free"), required=True
    )
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.reference_audio.is_file():
        raise FileNotFoundError(args.reference_audio)
    audio_root = args.run_root / "audio" / args.model_id
    metrics_root = args.run_root / "metrics"
    audio_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_root / f"{args.model_id}.jsonl"

    manifest = {
        "schema_version": 1,
        "reference_audio": str(args.reference_audio),
        "reference_audio_sha256": sha256_file(args.reference_audio),
        "reference_transcript": REFERENCE_TRANSCRIPT,
        "seed": args.seed,
        "conditions": [asdict(condition) for condition in CONDITIONS],
        "interpretation_boundary": (
            "The four matched paths isolate model adaptation and inference-time "
            "reference use. They do not establish human identity or quality without "
            "blind review."
        ),
    }
    manifest_path = args.run_root / "prompt-manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("existing prompt manifest differs")
    else:
        atomic_json(manifest_path, manifest)

    completed: dict[str, dict[str, Any]] = {}
    if metrics_path.exists():
        for line in metrics_path.read_text().splitlines():
            row = json.loads(line)
            if row.get("status") == "success":
                completed[row["condition_id"]] = row

    tokenizer, model, audio_tokenizer = load_runtime(
        args.model_root,
        device=args.device,
        attn_implementation="eager",
    )
    update_generation_config_for_breeze(model)
    runtime_config = FastStreamingConfig(
        max_new_tokens=1500,
        max_seq_len=2048,
        collect_timing=True,
        fast_all=False,
        repetition_penalty=1.1,
    )

    for condition in CONDITIONS:
        output = audio_root / f"{condition.condition_id}__seed-{args.seed}.wav"
        previous = completed.get(condition.condition_id)
        if previous is not None:
            if not output.is_file() or sha256_file(output) != previous["sha256"]:
                raise RuntimeError(f"completed artifact changed: {output}")
            continue
        if output.exists():
            raise FileExistsError(f"unreceipted output requires review: {output}")

        request = {
            "id": condition.condition_id,
            "text": condition.text,
            "instruction": condition.instruction,
            "speaker": "S0",
        }
        template = "tts_instruction"
        if args.mode == "reference":
            request["ref_audio_path"] = str(args.reference_audio)
            request["ref_text"] = REFERENCE_TRANSCRIPT
            template = "ref_edit_tata"

        set_all_seeds(args.seed)
        inputs = prepare_inputs(
            tokenizer,
            audio_tokenizer,
            model,
            [request],
            get_template(template),
            guidance_scale=1.0,
            guidance_scale_ref=None,
            guidance_scale_ins=None,
        )
        runtime = FastBreezeStreamingRuntime(
            model, audio_tokenizer, runtime_config, tokenizer=tokenizer
        )
        partial = output.with_name(output.name + ".partial")
        if partial.exists():
            raise FileExistsError(f"stale partial output requires review: {partial}")
        started_at = time.perf_counter()
        first_audio = None
        chunks = 0
        frames = 0
        torch.cuda.reset_peak_memory_stats(args.device)
        with sf.SoundFile(
            partial,
            mode="x",
            format="WAV",
            samplerate=runtime.sample_rate,
            channels=1,
            subtype="PCM_16",
        ) as handle:
            for chunk in runtime.iter_audio_chunks(
                inputs, request_id=condition.condition_id
            ):
                if first_audio is None and chunk.audio.size:
                    first_audio = time.perf_counter() - started_at
                handle.write(chunk.audio)
                chunks += 1
                frames += int(chunk.codec_frames)
        partial.rename(output)
        info = sf.info(output)
        row = {
            "status": "success",
            "model_id": args.model_id,
            "mode": args.mode,
            "condition_id": condition.condition_id,
            "seed": args.seed,
            "output": str(output),
            "sha256": sha256_file(output),
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "duration_seconds": info.duration,
            "generation_seconds": time.perf_counter() - started_at,
            "time_to_first_audio_seconds": first_audio,
            "chunk_count": chunks,
            "codec_frames": frames,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(args.device)),
        }
        append_jsonl(metrics_path, row)
        print(
            f"completed {args.model_id}/{condition.condition_id} "
            f"duration={info.duration:.3f}s",
            flush=True,
        )

    atomic_json(
        args.run_root / f"{args.model_id}-receipt.json",
        {
            "schema_version": 1,
            "status": "matched_generation_complete",
            "model_id": args.model_id,
            "mode": args.mode,
            "model_root": str(args.model_root),
            "conditions": len(CONDITIONS),
            "metrics": str(metrics_path),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
