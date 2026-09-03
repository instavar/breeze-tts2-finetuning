#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

for required in LICENSE NOTICE MODEL_LICENSE.md README.md; do
  test -s "$required" || {
    echo "missing required release file: $required" >&2
    exit 1
  }
done

for forbidden in '*.wav' '*.mp3' '*.flac' '*.pt' '*.pth' '*.ckpt' '*.bin' '*.safetensors' '*.onnx' '*.npy' '*.npz'; do
  if git ls-files "$forbidden" | grep -q .; then
    echo "tracked model, audio, or tensor artifact matches $forbidden" >&2
    exit 1
  fi
done

# Exclude the two scanners that must name the markers they reject.
if git grep -I -n -E '/Users/[^/]+/|/home/[^/]+/|/mnt/(work|ext4)|FEMALE_01|female01' -- . \
  ':!scripts/check_public_tree.sh' \
  ':!training/release_bundle.py'; then
  echo "tracked source contains a private path or study identifier" >&2
  exit 1
fi

echo "public source contract passed"
