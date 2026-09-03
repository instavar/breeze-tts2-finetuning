# Model release operations

This procedure builds new release directories from selected training artifacts.
It never edits or removes the source checkpoint.

## Required gates

Before building a release:

1. Bind the artifact to an exact base-model revision and agreement version.
2. Confirm the release remains non-commercial research.
3. Confirm consent, recording rights, dataset terms, and attribution for the
   exact source-file hashes in a private rights record.
4. Select the checkpoint from validation and matched evaluation evidence.
5. Obtain a current, complete copy of the governing model agreement from the
   pinned official model repository.

## LoRA bundle

```bash
python -m training.release_bundle lora \
  --adapter /runs/lora/checkpoint-step-000500/adapter.safetensors \
  --adapter-sha256 <selected-checkpoint-adapter-sha256> \
  --base-model-root /models/Breeze-TTS-2 \
  --base-revision <exact-base-revision> \
  --model-license /agreements/Breeze-TTS-2-LICENSE \
  --card release/lora/README.md \
  --provenance release/lora/PROVENANCE.json \
  --output /releases/sg-narration-lora-r8
```

The builder hashes the base config, weight index, and every referenced base
weight shard. Those hashes become part of the adapter manifest and are checked
when the adapter is loaded.

## Full-SFT bundle

The source directory must contain a verified `model-role.json` plus the bundled
audio tokenizer. Training-state files may remain beside them because the builder
copies only the model-role allowlist and the exact audio-tokenizer allowlist.

```bash
python -m training.release_bundle full-sft \
  --source /runs/full-sft/checkpoint-step-000750 \
  --base-revision <exact-base-revision> \
  --model-license /agreements/Breeze-TTS-2-LICENSE \
  --card release/full-sft/README.md \
  --provenance release/full-sft/PROVENANCE.json \
  --output /releases/sg-narration-full-sft
```

## Private staging and clean-room verification

Create the Hub repositories as private, upload each generated directory, and
download it into a new cache and output directory on a host that cannot resolve
the original experiment paths. Verify:

- every `SHA256SUMS` entry;
- every sharded-model index target exists and is non-empty;
- the adapter rejects the wrong base revision and loads the pinned base;
- the full-SFT checkpoint loads in a fresh process;
- each release synthesizes the same disclosed smoke prompt; and
- no source audio, generated private audio, blind key, credential, absolute
  private path, optimizer, scheduler, random state, or trainer state is present.

Only change visibility after these checks pass. Publish LoRA first, then the
full-SFT checkpoint as a separate decision. Visibility changes do not revoke
copies that were already downloaded.
