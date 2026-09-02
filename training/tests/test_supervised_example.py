from types import SimpleNamespace

import pytest
import torch

from training.real_data import ManifestRow, select_rows
from training.real_lora import example_index, lr_multiplier
from training.supervised_example import (
    IGNORE_INDEX,
    make_supervised_labels,
    normalize_audio_tokens,
)

CONFIG = SimpleNamespace(audio_token_id=20, audio_eos_token_id=21)


def test_labels_supervise_audio_placeholders_only() -> None:
    input_ids = torch.tensor([[3, 4, 20, 20, 21]])
    labels = make_supervised_labels(input_ids, CONFIG)
    assert labels.tolist() == [[IGNORE_INDEX, IGNORE_INDEX, 20, 20, IGNORE_INDEX]]


def test_labels_require_audio_placeholders() -> None:
    with pytest.raises(ValueError, match="no audio placeholder"):
        make_supervised_labels(torch.tensor([[3, 4, 21]]), CONFIG)


def test_labels_require_one_eos_per_example() -> None:
    with pytest.raises(ValueError, match="exactly one audio EOS"):
        make_supervised_labels(torch.tensor([[3, 20, 21, 21]]), CONFIG)


def test_labels_require_rank_two_input() -> None:
    with pytest.raises(ValueError, match="rank 2"):
        make_supervised_labels(torch.tensor([3, 20, 21]), CONFIG)


def test_audio_token_ids_are_normalized_to_long() -> None:
    tokens = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.int16)
    normalized = normalize_audio_tokens(tokens)
    assert normalized.dtype == torch.long
    assert normalized.tolist() == tokens.tolist()


def test_audio_token_ids_reject_floating_point_values() -> None:
    with pytest.raises(TypeError, match="integer IDs"):
        normalize_audio_tokens(torch.zeros((1, 2, 3), dtype=torch.float32))


def test_manifest_selection_is_seeded_and_stable(tmp_path) -> None:
    rows = [
        ManifestRow(index, tmp_path / f"{index}.wav", f"row {index}")
        for index in range(1, 8)
    ]
    first = select_rows(rows, limit=4, seed=42)
    second = select_rows(list(reversed(rows)), limit=4, seed=42)
    assert first == second
    assert select_rows(rows, limit=4, seed=43) != first


def test_example_schedule_is_resume_derivable() -> None:
    uninterrupted = [example_index(step, 7, seed=42) for step in range(20)]
    resumed = [example_index(step, 7, seed=42) for step in range(9)] + [
        example_index(step, 7, seed=42) for step in range(9, 20)
    ]
    assert uninterrupted == resumed
    assert sorted(uninterrupted[:7]) == list(range(7))


def test_learning_rate_multiplier_warms_then_decays() -> None:
    values = [lr_multiplier(step, warmup_steps=2, max_steps=10) for step in range(10)]
    assert values[0] == pytest.approx(0.5)
    assert values[1] == pytest.approx(1.0)
    assert values[2] == pytest.approx(1.0)
    assert values[-1] < values[2]
