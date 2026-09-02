# Training guide

## Training surface

The upstream release exposes inference. This companion reconstructs supervised
training targets from an exact transcript and target audio:

1. tokenize the transcript with the checkpoint tokenizer;
2. encode the audio through the released audio tokenizer;
3. place ignore labels on text positions;
4. place the audio marker on audio placeholder positions; and
5. let the released model expand those positions into the 16 codebook targets
   and the backbone end-of-sequence target.

The codec and text encoder remain frozen. LoRA targets the synthesis backbone,
depth decoder, and the text, depth-input, and output projections. Full SFT trains
the released synthesis parameters while keeping those same two components
frozen.

## Admission sequence

Use these gates in order:

1. `scripts/run_full_sft_smoke.sh` proves one finite BF16 update and a fresh
   reload on one example.
2. `scripts/prepare_cache.sh` creates deterministic train and validation tensor
   caches from disjoint manifests.
3. `scripts/run_lora.sh` or `scripts/run_full_sft.sh` runs the selected path.
4. Select a checkpoint from held-out validation history.
5. Run the corresponding fresh-process verification and export.
6. Generate matched audio and freeze blind ratings before decoding identities.

Passing the feasibility gate proves that labels, forward loss, gradients,
updates, serialization, and reload are connected. It does not prove useful
adaptation, speaker similarity, naturalness, or convergence.

## LoRA lifecycle

`training.real_lora` writes atomic five-role checkpoints containing the adapter,
optimizer, scheduler, random-number state, and receipt. SIGTERM requests a clean
stop at a checkpoint boundary. Resume requires an explicit checkpoint and a new
output root.

After validation selection, `training.real_lora_verify` reloads the adapter in a
fresh process, evaluates held-out examples, merges the adapted linear layers,
and exports a separate merged package. `training.real_lora_merged_smoke` then
reloads that package independently.

## Full-SFT lifecycle

`training.real_full_sft` supports:

- FP32-master SGD for a low-state optimizer on a 24 GB GPU;
- Adafactor as an alternative low-memory optimizer;
- separate learning-rate multipliers for the backbone, depth decoder, text
  projection, and remaining synthesis parameters;
- an independent cosine schedule horizon;
- validation without checkpoint export;
- interval, final, model-only, or no-save policies; and
- explicit resume from a complete checkpoint.

Full-SFT exports are large. Check local disk capacity before enabling interval
checkpoints and never point two runs at the same output root.

## Sweep plans

`training.full_sft_sweep` reads a JSON object with `defaults` and `trials`.
Every trial receives its own directory and command receipt. The plans in
`training/plans/` are examples from one bounded experiment, not universal
recommendations. Re-test learning rates when the dataset, batch construction,
optimizer, schedule horizon, or model revision changes.

## Data and consent

The tools accept arbitrary manifest paths but do not grant rights to any audio,
speaker, transcript, model, or output. Dataset availability is not proof of
permission to create or distribute a reusable voice model. Keep the rights and
consent record outside Git and bind it to the dataset version used for training.
