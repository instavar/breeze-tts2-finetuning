---
license: other
license_name: breezeblue-research-and-non-commercial-license-v1.1
license_link: LICENSE
base_model: BreezeBlue/Breeze-TTS-2
base_model_relation: adapter
pipeline_tag: text-to-speech
language:
  - en
tags:
  - text-to-speech
  - lora
  - singapore-english
  - non-commercial
extra_gated_prompt: >-
  This derivative is available only for research and non-commercial use under
  the included BreezeBlue agreement. Gating records an acknowledgement but does
  not replace the agreement or supply voice and recording rights.
extra_gated_fields:
  I will use this derivative only for research or non-commercial purposes: checkbox
  I will not use it for impersonation, deception, or non-consensual voice cloning: checkbox
  I have the rights and consent required for every reference voice and recording I use: checkbox
---

# Instavar SG Narration LoRA R8

Derived from Breeze TTS 2 by BreezeBlue and licensed for research and
non-commercial use only.

This is an independent research adapter trained by Instavar on a consented,
single-speaker subset of Singapore's National Speech Corpus. It is not an
official BreezeBlue release and is not endorsed by BreezeBlue or RESONIA, INC.

## Artifact

- Base: `BreezeBlue/Breeze-TTS-2`
- Pinned base revision: `799624c0b4a1daa8db6d28bbd9850043c0270734`
- Adapter: rank 8, alpha 16
- Selected checkpoint: step 500 of 1,000
- Trainable parameters: 12,092,448
- Adapted families: semantic backbone, depth decoder, text projection,
  depth-input projection, and output projection

The adapter manifest pins the base revision and checksums. The loader refuses a
revision or file mismatch.

## Use

Use the companion source toolkit:

```bash
python infer.py /models/Breeze-TTS-2 \
  --adapter /models/sg-narration-lora-r8 \
  --base-revision 799624c0b4a1daa8db6d28bbd9850043c0270734 \
  --text "The train arrives in five minutes." \
  --output output.wav
```

Source and documentation:
[instavar/breeze-tts2-finetuning](https://github.com/instavar/breeze-tts2-finetuning)

## Evaluation

In a matched reference-free comparison, this adapter reached mean ECAPA speaker
similarity 0.6810 and WER 0.0467. The selected full-SFT checkpoint reached 0.6973
and the same WER. The paired ECAPA interval crossed zero, so the objective study
did not establish a winner.

One listener preferred this adapter for cadence and long-form listening, with
three short-prompt ties. Both releases mispronounced a tested local word. These
results do not establish faithful identity cloning, general Singaporean-accent
coverage, production fitness, or broad listener preference.

## Licence and responsible use

The included BreezeBlue agreement permits research and non-commercial use only.
Do not use this artifact for a product, paid service, client work, advertising,
revenue generation, production deployment, impersonation, deception, or voice
cloning without legally sufficient consent and recording rights.

Review `LICENSE`, `NOTICE`, `PROVENANCE.json`, and `SHA256SUMS` before use.
