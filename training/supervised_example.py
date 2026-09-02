from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from breeze_infer.templates import _prepare_one

IGNORE_INDEX = -100


def normalize_audio_tokens(audio_tokens: torch.Tensor) -> torch.Tensor:
    """Normalize discrete codec IDs to the dtype required by embedding and labels."""

    if audio_tokens.ndim != 3:
        raise ValueError(
            "audio tokens must have shape (batch, frames, codebooks), "
            f"got shape={tuple(audio_tokens.shape)}"
        )
    if audio_tokens.is_floating_point() or audio_tokens.is_complex():
        raise TypeError(f"audio tokens must be integer IDs, got {audio_tokens.dtype}")
    return audio_tokens.to(dtype=torch.long)


def make_supervised_labels(input_ids: torch.Tensor, model_config: Any) -> torch.Tensor:
    """Build the 2D label contract expected by Breeze's merge helper.

    Breeze expands these labels to all codebooks inside
    ``_merge_input_ids_with_input_values``. Audio placeholders become the
    actual target codec IDs, while text remains ignored. The model inserts its
    backbone EOS target at the audio EOS position during that expansion.
    """

    if input_ids.ndim != 2:
        raise ValueError(
            f"input_ids must be rank 2, got shape={tuple(input_ids.shape)}"
        )

    audio_token_id = int(model_config.audio_token_id)
    audio_eos_token_id = int(model_config.audio_eos_token_id)
    audio_mask = input_ids.eq(audio_token_id)
    eos_mask = input_ids.eq(audio_eos_token_id)

    if int(audio_mask.sum().item()) == 0:
        raise ValueError("supervised example contains no audio placeholder tokens")
    if int(eos_mask.sum().item()) != input_ids.shape[0]:
        raise ValueError(
            "each supervised example must contain exactly one audio EOS token; "
            f"batch={input_ids.shape[0]} eos_count={int(eos_mask.sum().item())}"
        )

    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[audio_mask] = audio_token_id
    return labels


def build_supervised_example(
    tokenizer: Any,
    audio_tokenizer: Any,
    model_config: Any,
    *,
    audio_path: str | Path,
    transcript: str,
    speaker: str = "S0",
) -> dict[str, torch.Tensor]:
    """Create one teacher-forced text-to-audio example from an exact transcript."""

    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(audio_path)
    transcript = transcript.strip()
    if not transcript:
        raise ValueError("transcript must not be empty")

    prefix = speaker if speaker.startswith("[") else f"[{speaker}]"
    prepared = _prepare_one(
        tokenizer,
        audio_tokenizer,
        model_config,
        [
            {"type": "text", "text": f"{prefix}{transcript}"},
            {
                "type": "audio",
                "audio_path": str(audio_path),
                "append_eos": True,
            },
        ],
    )

    input_ids = prepared["input_ids"]
    audio_tokens = normalize_audio_tokens(prepared["audio_tokens"])
    labels = make_supervised_labels(input_ids, model_config)
    placeholder_count = int(input_ids.eq(int(model_config.audio_token_id)).sum().item())
    frame_count = int(audio_tokens.shape[1])
    if placeholder_count != frame_count:
        raise ValueError(
            "audio placeholder and target frame counts differ: "
            f"placeholders={placeholder_count} frames={frame_count}"
        )
    if audio_tokens.shape[-1] != int(model_config.num_codebooks):
        raise ValueError(
            "unexpected codebook count: "
            f"expected={model_config.num_codebooks} actual={audio_tokens.shape[-1]}"
        )

    return {
        "input_ids": input_ids.cpu(),
        "attention_mask": prepared["attention_mask"].cpu(),
        "text_ids_mask": prepared["text_ids_mask"].cpu(),
        "text_ids_len": prepared["text_ids_len"].cpu(),
        "input_values": audio_tokens.cpu(),
        "labels": labels.cpu(),
    }


def example_summary(
    example: dict[str, torch.Tensor], model_config: Any
) -> dict[str, Any]:
    input_ids = example["input_ids"]
    labels = example["labels"]
    return {
        "sequence_length": int(input_ids.shape[1]),
        "audio_frames": int(example["input_values"].shape[1]),
        "num_codebooks": int(example["input_values"].shape[2]),
        "supervised_audio_positions": int(
            labels.eq(int(model_config.audio_token_id)).sum().item()
        ),
        "ignored_positions": int(labels.eq(IGNORE_INDEX).sum().item()),
        "audio_eos_positions": int(
            input_ids.eq(int(model_config.audio_eos_token_id)).sum().item()
        ),
    }
