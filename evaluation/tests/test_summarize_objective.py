from __future__ import annotations

import pytest

from evaluation.summarize_objective import PAIRINGS, parse_pairings


def test_parse_pairings_keeps_legacy_defaults() -> None:
    assert parse_pairings([]) == PAIRINGS


def test_parse_pairings_accepts_repeated_comparisons() -> None:
    assert parse_pairings(["full:base", "full:lora"]) == (
        ("full", "base"),
        ("full", "lora"),
    )


@pytest.mark.parametrize("value", ["missing", ":base", "full:", "a:b:c"])
def test_parse_pairings_rejects_malformed_value(value: str) -> None:
    with pytest.raises(ValueError):
        parse_pairings([value])


def test_parse_pairings_rejects_duplicates() -> None:
    with pytest.raises(ValueError):
        parse_pairings(["full:base", "full:base"])
