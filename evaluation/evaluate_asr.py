from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel

LEXICAL_PROBES = ("tanjong pagar", "sze min", "jalan membina", "paiseh")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(text: str) -> list[str]:
    text = re.sub(r"\([^)]*\)", " ", text.lower())
    return re.findall(r"[a-z0-9]+", text)


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_word in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_word in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_word != hyp_word),
                )
            )
        previous = current
    return previous[-1]


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def condition_id(path: Path) -> str:
    return path.name.split("__seed-", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(
        (args.run_root / "prompt-manifest.json").read_text(encoding="utf-8")
    )
    targets = {row["condition_id"]: row["text"] for row in manifest["conditions"]}
    output_path = args.run_root / "metrics" / "asr-results.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["sample_id"]] = row

    model = WhisperModel(
        "large-v3",
        device="cpu",
        compute_type="int8",
        local_files_only=True,
    )
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
                raise RuntimeError(f"audio changed after ASR: {sample_id}")
            print(f"skip verified {sample_id}", flush=True)
            continue
        cid = condition_id(audio_path)
        target = targets[cid]
        segments, info = model.transcribe(
            str(audio_path),
            language="en",
            beam_size=5,
            vad_filter=False,
        )
        hypothesis = " ".join(segment.text.strip() for segment in segments).strip()
        target_words = normalize(target)
        hypothesis_words = normalize(hypothesis)
        distance = edit_distance(target_words, hypothesis_words)
        normalized_hypothesis = " ".join(hypothesis_words)
        exact_probe_hits = {
            probe: probe in normalized_hypothesis
            for probe in LEXICAL_PROBES
            if probe in " ".join(target_words)
        }
        terminal_words = target_words[-5:]
        terminal_overlap = (
            sum(word in hypothesis_words[-12:] for word in terminal_words)
            / len(terminal_words)
            if terminal_words
            else None
        )
        row = {
            "sample_id": sample_id,
            "model_id": audio_path.parent.name,
            "condition_id": cid,
            "audio_path": str(audio_path),
            "audio_sha256": audio_hash,
            "asr_model": "Systran/faster-whisper-large-v3",
            "asr_device": "cpu",
            "asr_compute_type": "int8",
            "detected_language": info.language,
            "detected_language_probability": info.language_probability,
            "target_text": target,
            "hypothesis": hypothesis,
            "target_word_count": len(target_words),
            "hypothesis_word_count": len(hypothesis_words),
            "word_count_ratio": len(hypothesis_words) / len(target_words)
            if target_words
            else None,
            "word_error_count": distance,
            "wer": distance / len(target_words) if target_words else None,
            "exact_lexical_probe_hits": exact_probe_hits,
            "terminal_five_word_overlap": terminal_overlap,
            "interpretation_boundary": "ASR recognition evidence is not pronunciation, accent, identity, or naturalness evidence.",
        }
        append_jsonl(output_path, row)
        completed[sample_id] = row
        print(f"completed {sample_id} wer={row['wer']:.4f}", flush=True)


if __name__ == "__main__":
    main()
