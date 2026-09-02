# Instavar Breeze TTS 2 fine-tuning

An independent, source-only companion for LoRA and full supervised fine-tuning
of [Breeze TTS 2](https://github.com/breezeblue-ai/breeze-tts).

This repository adds the training and evaluation surface that is not included in
the upstream inference release. It does not contain model weights, adapters,
checkpoints, training audio, or generated speech.

Instavar is not affiliated with or endorsed by BreezeBlue or RESONIA, INC.

## What is implemented

| Capability | Implementation |
| --- | --- |
| Supervised labels | Reconstructs text and 16-codebook audio targets from an exact transcript and target WAV |
| Feasibility gate | One BF16 forward and backward step, finite losses, nonzero family gradients, changed weights, and fresh-process reload |
| LoRA | Backbone, depth decoder, text projection, depth-input projection, and output projection targets |
| LoRA lifecycle | Trainable-parameter receipt, checkpoint, resume, adapter export, merge, and fresh-process merged reload |
| Full SFT | BF16 synthesis model with frozen codec and text encoder, FP32-master SGD or Adafactor, family learning-rate multipliers, and bounded checkpoints |
| Full-SFT sweeps | JSON plans, multiple seeds, independent schedule horizons, validation-only trials, and lightweight finalist exports |
| Evaluation | Matched prompts and seeds, ASR, speaker similarity, acoustic diagnostics, confidence intervals, and an opaque blind-listening pack |
| Safety and reproducibility | Refuses output overwrite, hashes source audio and artifacts, records Git state, writes atomic receipts, and handles SIGTERM at checkpoint boundaries |

These tools were exercised on an NVIDIA RTX 3090 Ti. That establishes the
documented code paths on one 24 GB CUDA system, not general convergence or model
quality on every dataset or GPU.

## Install

Use Linux, Python 3.11, and a CUDA GPU. The dependency pins match the upstream
runtime.

```bash
git clone https://github.com/instavar/breeze-tts2-finetuning.git
cd breeze-tts2-finetuning
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Obtain Breeze TTS 2 separately from its official distribution after reading and
accepting its model agreement. Do not add the checkpoint to this repository.

## Dataset manifest

Create separate train and validation JSONL files. Each row has an absolute audio
path and its exact transcript:

```json
{"audio":"/data/speaker/0001.wav","text":"The exact words spoken in this file."}
```

Use clean, single-speaker recordings and only material for which you have all
necessary rights and consent. Keep the validation recordings disjoint from the
training recordings.

## Prepare deterministic targets

```bash
export BREEZE_MODEL_ROOT=/models/Breeze-TTS-2
export BREEZE_TRAIN_MANIFEST=/data/train.jsonl
export BREEZE_VALIDATION_MANIFEST=/data/validation.jsonl
export BREEZE_CACHE_ROOT=/runs/cache-v1

bash scripts/prepare_cache.sh --train-limit 1024 --validation-limit 128
```

The cache stores model-ready tensors and a receipt containing hashes and source
revision information. Treat it as sensitive if the source dataset is sensitive.

## Run LoRA

```bash
export BREEZE_RUN_ROOT=/runs/lora-r8
bash scripts/run_lora.sh \
  --rank 8 \
  --alpha 16 \
  --max-steps 1000 \
  --gradient-accumulation 4 \
  --learning-rate 2e-4 \
  --save-every 250
```

Resume by using a new output root and passing the prior checkpoint explicitly:

```bash
bash scripts/run_lora.sh \
  --resume-checkpoint /runs/lora-r8/checkpoint-step-000250 \
  --max-steps 1000
```

Adapter verification and merge are separate so a failed verification cannot
overwrite the training run:

```bash
python -m training.real_lora_verify \
  --model-root "$BREEZE_MODEL_ROOT" \
  --cache-root "$BREEZE_CACHE_ROOT" \
  --training-root /runs/lora-r8 \
  --merged-output /runs/lora-r8-merged

python -m training.real_lora_merged_smoke \
  --cache-root "$BREEZE_CACHE_ROOT" \
  --training-root /runs/lora-r8 \
  --merged-model /runs/lora-r8-merged
```

## Run full SFT

Start with the one-example feasibility gate before committing to a real run:

```bash
export BREEZE_TRAIN_AUDIO=/data/speaker/0001.wav
export BREEZE_TRAIN_TRANSCRIPT='The exact words spoken in this file.'
export BREEZE_RUN_ROOT=/runs/full-sft-smoke
bash scripts/run_full_sft_smoke.sh
```

Then run multi-example full SFT:

```bash
export BREEZE_RUN_ROOT=/runs/full-sft
bash scripts/run_full_sft.sh \
  --optimizer fp32_master_sgd \
  --max-steps 1000 \
  --gradient-accumulation 4 \
  --learning-rate 2e-5 \
  --save-every 250
```

See [training details](docs/TRAINING.md) before choosing an optimizer or copying
one of the included sweep plans.

## Evaluate before selecting

Do not choose a checkpoint from training loss alone. The repository supports:

- validation-based checkpoint selection;
- matched base, LoRA, and full-SFT generation;
- ASR word-error diagnostics;
- ECAPA speaker-similarity diagnostics;
- pitch, energy, pause, and duration diagnostics;
- bootstrap confidence intervals; and
- blind listening packs with a mode-`0600` private key.

See [evaluation](docs/EVALUATION.md) for the evaluation sequence and what each
measurement does not prove.

## License boundary

Repository source is provided under Apache License 2.0. Breeze TTS 2 model
materials, derivative models, adapters, checkpoints, and self-hosted outputs are
governed separately by the BreezeBlue Research and Non-Commercial License.

The source license does not grant a right to download, use, distribute, or use
Breeze model materials commercially. See [model license boundary](MODEL_LICENSE.md)
and [release boundaries](docs/RELEASE_BOUNDARIES.md).

## Acknowledgments

- [BreezeBlue](https://breezeblue.ai/) for Breeze TTS 2 and its upstream PyTorch runtime
- [Qwen](https://github.com/QwenLM/Qwen3-TTS) for the Qwen3-TTS audio tokenizer used by Breeze
- The open-source libraries listed in `requirements.txt`

## Citation

See [`CITATION.cff`](CITATION.cff).
