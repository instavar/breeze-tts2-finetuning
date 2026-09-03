---
license: other
license_name: breezeblue-research-and-non-commercial-license-v1.1
license_link: LICENSE
base_model: BreezeBlue/Breeze-TTS-2
base_model_relation: finetune
pipeline_tag: text-to-speech
language:
  - en
tags:
  - text-to-speech
  - full-finetune
  - singapore-english
  - non-commercial
---

# Instavar SG Narration Full SFT

Derived from Breeze TTS 2 by BreezeBlue and licensed for research and
non-commercial use only.

This is an independent research checkpoint trained by Instavar on a consented,
single-speaker subset of Singapore's National Speech Corpus. It is not an
official BreezeBlue release and is not endorsed by BreezeBlue or RESONIA, INC.

## Artifact

- Base: `BreezeBlue/Breeze-TTS-2`
- Pinned base revision: `799624c0b4a1daa8db6d28bbd9850043c0270734`
- Selected checkpoint: step 750 of 1,000
- Updated synthesis parameters: 2,387,151,872
- Frozen during training: text encoder and audio codec
- Optimizer: FP32-master SGD

This repository contains inference roles only. Optimizer, scheduler, random
state, trainer state, source recordings, caches, and private receipts are not
included.

## Use

Use the companion source toolkit:

```bash
python infer.py /models/sg-narration-full-sft \
  --text "The train arrives in five minutes." \
  --output output.wav
```

Source and documentation:
[instavar/breeze-tts2-finetuning](https://github.com/instavar/breeze-tts2-finetuning)

## Evaluation

In a matched reference-free comparison, this checkpoint reached mean ECAPA
speaker similarity 0.6973 and WER 0.0467. The LoRA adapter reached 0.6810 and the
same WER. The paired ECAPA interval crossed zero, so the objective study did not
establish a winner.

One listener preferred LoRA for cadence and long-form listening, with three
short-prompt ties. Both releases mispronounced a tested local word. These results
do not establish faithful identity cloning, general Singaporean-accent coverage,
production fitness, or broad listener preference.

## Licence and responsible use

The included BreezeBlue agreement permits research and non-commercial use only.
Do not use this artifact for a product, paid service, client work, advertising,
revenue generation, production deployment, impersonation, deception, or voice
cloning without legally sufficient consent and recording rights.

Review `LICENSE`, `NOTICE`, `PROVENANCE.json`, and `SHA256SUMS` before use.
