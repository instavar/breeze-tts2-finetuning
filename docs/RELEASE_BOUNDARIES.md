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

## v0.1.0 source release decision

Checked on 2026-09-02:

- the public source repository is based on upstream source revision
  `d76819fa9c042c045e2e0cb9b6285795f677ff90`;
- the official model repository reported revision
  `799624c0b4a1daa8db6d28bbd9850043c0270734` and agreement version 1.1;
- the Apache-licensed source toolkit was approved for public release without
  model materials; and
- public LoRA, full-SFT, merged, distilled, or other derivative weights were
  held back.

The weight hold has two independent reasons. The stated first-mover Instavar
branding purpose may be an indirect business benefit under version 1.1, so it
requires written BreezeBlue authorization before public distribution. A public
weight release also needs a training corpus with separately verified voice,
recording, and redistribution rights. A technically successful private run is
not evidence that either release gate has passed.

## Subsequent research-artifact decision

The project later narrowed the proposed publication to a non-monetized research
release rather than a product, hosted service, client deliverable, advertising
campaign, or production deployment. Organization hosting and incidental public
reputation do not by themselves make business benefit the primary purpose. The
speaker subset was also confirmed as National Speech Corpus material covered by
the Singapore Open Data Licence, with the project's separate consent and rights
record retained privately.

Under that narrowed purpose, version 1.1 supports distributing the checked LoRA
and full-SFT derivatives if every agreement, notice, provenance, naming, rights,
sanitization, and non-commercial restriction is satisfied. This is an
artifact-specific decision, not a general removal of the release gate for future
models, datasets, checkpoints, or purposes.

Use `training.release_bundle` and [the model release runbook](RELEASING_MODELS.md).
Never upload a training checkpoint directory directly.

This document is operational guidance, not legal advice.
