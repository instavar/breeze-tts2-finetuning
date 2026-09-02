#!/usr/bin/env bash
set -euo pipefail

: "${BREEZE_MODEL_ROOT:?Set BREEZE_MODEL_ROOT to a Breeze TTS 2 checkpoint directory}"
: "${BREEZE_TRAIN_MANIFEST:?Set BREEZE_TRAIN_MANIFEST to a JSONL manifest}"
: "${BREEZE_VALIDATION_MANIFEST:?Set BREEZE_VALIDATION_MANIFEST to a JSONL manifest}"
: "${BREEZE_CACHE_ROOT:?Set BREEZE_CACHE_ROOT to a new output directory}"

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

exec python -m training.real_data \
  --model-root "$BREEZE_MODEL_ROOT" \
  --train-manifest "$BREEZE_TRAIN_MANIFEST" \
  --validation-manifest "$BREEZE_VALIDATION_MANIFEST" \
  --output-root "$BREEZE_CACHE_ROOT" \
  "$@"
