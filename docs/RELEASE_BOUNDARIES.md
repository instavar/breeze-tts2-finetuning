# Release boundaries

## Source repository

This repository is intended to remain source-only. The public-tree check rejects
common audio, tensor, adapter, and checkpoint extensions, as well as known
private path and study identifiers.

Keep these outside Git:

- model weights and model-specific tokenizer or codec weights;
- LoRA adapters, full-SFT checkpoints, merged models, and optimizer states;
- generated speech and source recordings;
- tensor caches and evaluation artifacts;
- credentials, private paths, blind keys, and consent records.

## Model artifacts

The BreezeBlue model agreement governs model materials and derivatives
separately from this repository's source license. Before distributing an
adapter, fine-tune, merge, quantization, or distillation:

1. identify the exact source model revision and governing agreement version;
2. confirm the purpose is permitted by that agreement;
3. include every required agreement, notice, attribution, provenance, and
   modification statement;
4. avoid names or presentation that imply official status or endorsement;
5. verify rights and explicit consent for every speaker and recording; and
6. obtain separate written authorization for any commercial purpose.

An organization's public branding or first-mover benefit can affect the
commercial-purpose analysis. Do not infer that a public non-monetized download
is automatically non-commercial.

## v0.1.0 source release and current model-artifact boundary

Checked on 2026-09-02:

- the public source repository is based on upstream source revision
  `d76819fa9c042c045e2e0cb9b6285795f677ff90`;
- the official model repository reported revision
  `799624c0b4a1daa8db6d28bbd9850043c0270734` and agreement version 1.1;
- the Apache-licensed source toolkit was approved for public release without
  model materials; and
- public LoRA, full-SFT, merged, distilled, or other derivative weights were
  held back.

The source-only scope of `v0.1.0` remains unchanged. A later review clarified
that version 1.1 expressly permits distribution of LoRA and full-SFT
Derivative Models for research or non-commercial purposes under Section 4.
Publishing ordinary, non-monetized research through an organization's account
does not by itself establish that commercial benefit is the release's primary
purpose. Product, hosted-service, customer, production, advertising, revenue,
or other business-directed use remains outside that free grant.

The training material used in the checked research runs is from the IMDA
National Speech Corpus. IMDA distributes that corpus under the Singapore Open
Data Licence, which permits use, modification, adaptation, and distribution,
subject to attribution. The corpus is therefore not treated as a blocker to a
research or non-commercial derivative-model release.

Model artifacts still belong in separate model repositories rather than this
source repository. Before publishing one, build a fresh allowlisted bundle and
verify it from a private staging repository. A LoRA bundle should contain only
the selected adapter, its configuration, a sanitized provenance manifest, the
complete BreezeBlue agreement, the required NOTICE, and a model card. A
full-SFT bundle should contain only inference-required model, tokenizer, codec,
and configuration files plus the same legal and provenance documents. Exclude
optimizer, scheduler, trainer, RNG, source-audio, generated-audio, cache, log,
credential, blind-key, and private-path artifacts.

The safest operational order is to validate and release the smaller LoRA
adapter first, then publish the full-SFT package after a clean independent
download and inference check. Platform gating can reduce accidental access and
record acknowledgement, but it does not change the governing agreement.

Use `training.release_bundle` and [the model release runbook](RELEASING_MODELS.md).
Never upload a training checkpoint directory directly.

This document is operational guidance, not legal advice.
