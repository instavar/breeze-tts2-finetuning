from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torchaudio

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

from speechbrain.inference.speaker import EncoderClassifier


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def embedding(classifier: EncoderClassifier, path: Path) -> torch.Tensor:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    signal = torch.from_numpy(audio).to(classifier.device)
    signal = classifier.audio_normalizer(signal, sample_rate)
    value = (
        classifier.encode_batch(signal.unsqueeze(0)).squeeze().detach().cpu().float()
    )
    return torch.nn.functional.normalize(value, dim=0)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--reference-audio", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    args = parser.parse_args()

    output_path = args.run_root / "metrics" / "ecapa-results.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if output_path.exists():
        for line in output_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["sample_id"]] = row

    classifier = EncoderClassifier.from_hparams(
        source=str(args.model_dir),
        run_opts={"device": "cpu"},
        overrides={"pretrained_path": str(args.model_dir)},
    )
    reference_embedding = embedding(classifier, args.reference_audio)
    reference_hash = sha256(args.reference_audio)
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
                raise RuntimeError(f"audio changed after ECAPA extraction: {sample_id}")
            print(f"skip verified {sample_id}", flush=True)
            continue
        candidate = embedding(classifier, audio_path)
        similarity = float(torch.dot(reference_embedding, candidate).item())
        condition_id = audio_path.name.split("__seed-", 1)[0]
        row = {
            "sample_id": sample_id,
            "model_id": audio_path.parent.name,
            "condition_id": condition_id,
            "audio_path": str(audio_path),
            "audio_sha256": audio_hash,
            "reference_audio": str(args.reference_audio),
            "reference_audio_sha256": reference_hash,
            "extractor": "speechbrain/spkrec-ecapa-voxceleb",
            "extractor_revision": args.model_revision,
            "extractor_model_dir": str(args.model_dir),
            "device": "cpu",
            "trusted_model_checkpoints": True,
            "cosine_similarity": similarity,
            "reference_role": (
                "diagnostic_not_target"
                if condition_id == "voice_design_singapore"
                else "target_voice_reference"
            ),
            "interpretation_boundary": "ECAPA similarity is a bounded speaker proxy, not blind identity, accent, cadence, or quality evidence.",
        }
        append_jsonl(output_path, row)
        completed[sample_id] = row
        print(f"completed {sample_id} similarity={similarity:.4f}", flush=True)


if __name__ == "__main__":
    main()
