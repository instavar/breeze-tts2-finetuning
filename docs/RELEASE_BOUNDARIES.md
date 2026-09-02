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

This document is operational guidance, not legal advice.
