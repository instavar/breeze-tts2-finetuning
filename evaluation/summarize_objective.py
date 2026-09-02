from __future__ import annotations

import argparse
import json
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

PAIRINGS = (
    ("adapted-reference", "base-reference"),
    ("adapted-reference-free", "base-reference-free"),
)


def parse_pairings(values: list[str]) -> tuple[tuple[str, str], ...]:
    if not values:
        return PAIRINGS
    result = []
    for value in values:
        adapted, separator, base = value.partition(":")
        if not separator or not adapted or not base or ":" in base:
            raise ValueError(f"pair must use adapted:base form: {value}")
        result.append((adapted, base))
    if len(set(result)) != len(result):
        raise ValueError("pair arguments must be unique")
    return tuple(result)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def bootstrap_mean_ci(
    values: list[float], seed: int = 42, samples: int = 10_000
) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    rng = random.Random(seed)
    draws = sorted(mean([rng.choice(values) for _ in values]) for _ in range(samples))
    return [draws[int(samples * 0.025)], draws[int(samples * 0.975)]]


def keyed(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["model_id"], row["condition_id"]): row for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize bounded objective diagnostics for Breeze matched audio"
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--pair",
        action="append",
        default=[],
        help="paired comparison in adapted:base form; repeat for multiple pairs",
    )
    args = parser.parse_args()
    pairings = parse_pairings(args.pair)

    metrics_root = args.run_root / "metrics"
    asr_rows = read_jsonl(metrics_root / "asr-results.jsonl")
    ecapa_rows = read_jsonl(metrics_root / "ecapa-results.jsonl")
    acoustic_rows = read_jsonl(metrics_root / "acoustic-results.jsonl")
    expected = len(list((args.run_root / "audio").glob("*/*.wav")))
    if expected <= 0:
        raise RuntimeError("matched audio is absent")
    if not (len(asr_rows) == len(ecapa_rows) == len(acoustic_rows) == expected):
        raise RuntimeError(
            "objective metrics are incomplete: "
            f"asr={len(asr_rows)} ecapa={len(ecapa_rows)} "
            f"acoustic={len(acoustic_rows)} expected={expected}"
        )

    by_model: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in asr_rows:
        by_model[row["model_id"]]["wer"].append(float(row["wer"]))
        by_model[row["model_id"]]["word_count_ratio"].append(
            float(row["word_count_ratio"])
        )
        by_model[row["model_id"]]["terminal_five_word_overlap"].append(
            float(row["terminal_five_word_overlap"])
        )
    for row in ecapa_rows:
        by_model[row["model_id"]]["ecapa_cosine_similarity"].append(
            float(row["cosine_similarity"])
        )
    for row in acoustic_rows:
        by_model[row["model_id"]]["duration_seconds"].append(
            float(row["duration_seconds"])
        )
        by_model[row["model_id"]]["f0_semitone_std"].append(
            float(row["f0_semitone_std"])
        )

    model_summary = {
        model_id: {
            metric: {
                "mean": mean(values),
                "bootstrap_95_ci": bootstrap_mean_ci(values),
                "n_prompts": len(values),
            }
            for metric, values in metrics.items()
        }
        for model_id, metrics in sorted(by_model.items())
    }

    asr_index = keyed(asr_rows)
    ecapa_index = keyed(ecapa_rows)
    acoustic_index = keyed(acoustic_rows)
    pairwise = []
    for adapted, base in pairings:
        condition_ids = sorted(
            condition for model_id, condition in asr_index if model_id == adapted
        )
        base_condition_ids = sorted(
            condition for model_id, condition in asr_index if model_id == base
        )
        if not condition_ids or condition_ids != base_condition_ids:
            raise RuntimeError(
                f"paired conditions differ: adapted={adapted} base={base}"
            )
        metrics = {
            "wer_delta_adapted_minus_base": [
                float(asr_index[(adapted, condition)]["wer"])
                - float(asr_index[(base, condition)]["wer"])
                for condition in condition_ids
            ],
            "ecapa_delta_adapted_minus_base": [
                float(ecapa_index[(adapted, condition)]["cosine_similarity"])
                - float(ecapa_index[(base, condition)]["cosine_similarity"])
                for condition in condition_ids
            ],
            "duration_delta_seconds_adapted_minus_base": [
                float(acoustic_index[(adapted, condition)]["duration_seconds"])
                - float(acoustic_index[(base, condition)]["duration_seconds"])
                for condition in condition_ids
            ],
        }
        pairwise.append(
            {
                "adapted": adapted,
                "base": base,
                "n_paired_prompts": len(condition_ids),
                "metrics": {
                    name: {
                        "mean": mean(values),
                        "bootstrap_95_ci": bootstrap_mean_ci(values),
                        "per_prompt": dict(zip(condition_ids, values, strict=True)),
                    }
                    for name, values in metrics.items()
                },
            }
        )

    output = {
        "schema_version": 1,
        "status": "objective_diagnostics_complete",
        "models": model_summary,
        "paired_differences": pairwise,
        "interpretation_boundary": (
            "These five-prompt objective diagnostics and bootstrap intervals are "
            "descriptive evidence for this matched pack only. They do not establish "
            "speaker identity, Singaporean accent, cadence quality, pronunciation, "
            "naturalness, long-form monotony, or listening fatigue."
        ),
    }
    atomic_json(args.run_root / "objective-summary.json", output)
    print(json.dumps({"status": output["status"], "models": len(model_summary)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
