#!/usr/bin/env bash
set -euo pipefail

: "${BREEZE_MODEL_ROOT:?Set BREEZE_MODEL_ROOT to a Breeze TTS 2 checkpoint directory}"
: "${BREEZE_TRAIN_AUDIO:?Set BREEZE_TRAIN_AUDIO to one authorized audio file}"
: "${BREEZE_TRAIN_TRANSCRIPT:?Set BREEZE_TRAIN_TRANSCRIPT to its exact transcript}"
: "${BREEZE_RUN_ROOT:?Set BREEZE_RUN_ROOT to a new output directory}"
BREEZE_DEVICE=${BREEZE_DEVICE:-cuda:0}

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

python -m training.full_sft_smoke \
  --model-root "$BREEZE_MODEL_ROOT" \
  --audio "$BREEZE_TRAIN_AUDIO" \
  --transcript "$BREEZE_TRAIN_TRANSCRIPT" \
  --output-root "$BREEZE_RUN_ROOT" \
  --device "$BREEZE_DEVICE" \
  "$@"

exec python -m training.reload_smoke \
  --run-root "$BREEZE_RUN_ROOT" \
  --device "$BREEZE_DEVICE"
