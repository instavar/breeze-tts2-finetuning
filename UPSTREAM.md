# Upstream provenance

This source tree was prepared from the official Breeze TTS 2 PyTorch repository:

- Repository: <https://github.com/breezeblue-ai/breeze-tts>
- Base revision: `d76819fa9c042c045e2e0cb9b6285795f677ff90`
- Base date checked: 2026-09-02

That revision includes upstream pull request 9, which fixes nested attention
selection and repeated streaming-runtime cache dtype. Instavar's training and
evaluation modules were then applied to that current base.

The Breeze model checkpoint is distributed separately. This repository does not
contain or mirror it.
