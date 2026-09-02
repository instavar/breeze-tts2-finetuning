# Contributing

Contributions are welcome when they improve the source-only training or
evaluation toolkit.

Before opening a pull request:

1. Keep model weights, adapters, checkpoints, audio, tensor caches, credentials,
   and personal data out of Git.
2. Use synthetic or properly licensed fixtures in tests.
3. Add or update focused tests for behavioral changes.
4. Run `python -m pytest -q` and `python -m ruff check .`.
5. Identify modifications to copied upstream files prominently.
6. Describe the tested GPU, dependency versions, dataset scope, and important
   untested boundaries for new empirical claims.

Do not submit voice recordings or reusable voice models unless you have all
necessary rights and explicit consent for that distribution.
