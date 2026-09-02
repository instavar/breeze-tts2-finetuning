# Evaluation guide

## Matched comparison

Compare the base checkpoint, LoRA export, and full-SFT export with identical
prompts, seeds, generation settings, and reference conditions. Keep reference
and reference-free conditions separate. Matching inputs reduces avoidable
variation but does not by itself prove that artifacts or runtime behavior are
equivalent.

`evaluation.generate_matched` writes hashes, prompt metadata, timing, and audio
paths. Use a separate output root for every study.

## Objective diagnostics

- `evaluation.evaluate_asr` measures transcript error through an ASR model.
- `evaluation.evaluate_ecapa` measures embedding similarity to a reference.
- `evaluation.evaluate_acoustics` measures duration, pitch, energy, and pauses.
- `evaluation.summarize_objective` creates paired summaries and descriptive
  bootstrap confidence intervals.

These are bounded diagnostics. Low word error does not prove natural speech.
High embedding similarity does not prove human identity or consent. Acoustic
statistics do not establish pleasant cadence or low listening fatigue.

## Blind listening

`evaluation.build_blind_pack` randomizes opaque sample identifiers and writes a
public manifest plus a mode-`0600` private key. Record all ratings before reading
the key.

Rate at least these dimensions separately:

- speaker identity;
- accent and regional pronunciation;
- cadence and timing;
- local names and words;
- long-form monotony; and
- listening fatigue.

Do not collapse them into one preference score. A model can sound smoother while
being less similar to the intended speaker, or sound regionally plausible while
matching the wrong identity.

## Interpretation boundary

A single speaker, seed, GPU, model revision, or prompt set is a case study. Use
multiple seeds and held-out speakers before making a broader model claim. Treat
automated metrics and blind human ratings as evidence with different failure
modes, not as interchangeable judges.
