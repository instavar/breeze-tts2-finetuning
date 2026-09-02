#!/usr/bin/env bash
set -euo pipefail

: "${BREEZE_MODEL_ROOT:?Set BREEZE_MODEL_ROOT to a Breeze TTS 2 checkpoint directory}"
: "${BREEZE_CACHE_ROOT:?Set BREEZE_CACHE_ROOT to a completed supervised cache}"
: "${BREEZE_RUN_ROOT:?Set BREEZE_RUN_ROOT to a new output directory}"

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

exec python -m training.real_full_sft \
  --model-root "$BREEZE_MODEL_ROOT" \
  --cache-root "$BREEZE_CACHE_ROOT" \
  --output-root "$BREEZE_RUN_ROOT" \
  "$@"
